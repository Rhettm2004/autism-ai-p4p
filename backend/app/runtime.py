import asyncio
import hashlib
import json
import logging
import yaml
from app.corpus import verify_corpus
from app.errors import ServiceError
from app.schemas import ChatResponse, ChatMetadata, Source
from app.adapters.llama_cpp import LlamaCppAdapter

log = logging.getLogger(__name__)

class ChatRuntime:
    def __init__(self, settings, adapter=None):
        self.settings = settings
        self.adapter = adapter or LlamaCppAdapter(settings.model_urls, settings.timeout_seconds)
        self.router = self.retriever = None
        self.training_texts = set()
        self.components = {'router': 'not_initialized', 'corpus': 'not_initialized', 'prompts': 'not_initialized'}
        self.component_details = {}
        self.corpus_hash = self.training_hash = self.prompt_hash = ''
        self._lock = asyncio.Lock()
        self.cfg = None

    async def initialize(self):
        # Each component is attempted independently so health reports all blockers.
        await asyncio.to_thread(self._initialize)

    def _initialize(self):
        root = self.settings.root
        try:
            from src.chat import load_condition_prompt, DemoSettings
            self.cfg = yaml.safe_load((root / 'config/runtime.yaml').read_text())
            if (self.cfg['router_config'], self.cfg['router_track'], self.cfg['router_dataset']) != (
                'embeddings_word', 'topic', 'extended'):
                raise ValueError('Unsupported research profile')
            r = self.cfg['retrieval']
            if r['backend'] != 'tfidf' or r['max_per_source'] != 2:
                raise ValueError('Unsupported retrieval profile')
            self.turn_settings = DemoSettings(use_router=True, use_rag=True,
                top_k=r['top_k'], expand_neighbours=r['expand_neighbours'],
                concise=self.cfg['concise'], cite=self.cfg['cite'])
            self.condition = load_condition_prompt(self.cfg['condition'])
            self.application = yaml.safe_load((root / 'config/application_prompts.yaml').read_text())
            self.prompt_hash = hashlib.sha256((
                (root / 'config/prompts.yaml').read_bytes() +
                (root / 'config/application_prompts.yaml').read_bytes()).strip()).hexdigest()
            self.components['prompts'] = 'ready'
        except Exception:
            log.exception('Prompt configuration unavailable')
            self.components['prompts'] = 'configuration_unavailable'
            self.component_details['prompts'] = 'configuration_unavailable'
        try:
            from src.chat import load_demo_router
            from src.router_eval import BENCHMARK_PATHS, ROUTER_PROMPTS_PATH
            training_paths = [*BENCHMARK_PATHS, ROUTER_PROMPTS_PATH]
            if any(not p.is_file() for p in training_paths):
                raise FileNotFoundError('Required research training data missing')
            self.router, self.training_texts = load_demo_router()
            self.training_hash = hashlib.sha256(b''.join(p.read_bytes() for p in training_paths)).hexdigest()
            self.components['router'] = 'ready'
        except Exception:
            log.exception('Research router unavailable; no replacement selected')
            self.components['router'] = 'router_unavailable'
            self.component_details['router'] = 'router_unavailable'
        try:
            from src.retrieval import Retriever
            passages, self.corpus_hash = verify_corpus(root)
            self.retriever = Retriever().index(passages)
            self.components['corpus'] = 'ready'
            readiness = json.loads((root / 'data/corpus/readiness.json').read_text())
            excluded = readiness.get('excluded_sources') or []
            if excluded:
                self.component_details['corpus_policy'] = (
                    f'reduced_corpus_{len(excluded)}_sources_excluded')
        except Exception as exc:
            log.warning('Corpus unavailable: %s', exc)
            self.components['corpus'] = 'corpus_unavailable'
            reason = str(exc)
            self.component_details['corpus'] = reason if reason in {
                'corpus_not_prepared', 'corpus_incomplete',
                'corpus_fingerprint_mismatch', 'corpus_source_mismatch',
                'corpus_policy_invalid',
            } else 'corpus_unavailable'

    async def health(self):
        model_ready = await self.adapter.ready(self.settings.default_model)
        ready = all(v == 'ready' for v in self.components.values()) and model_ready
        return ready, {'status': 'ready' if ready else 'unavailable', 'api_version': 1,
            'runtime_profile': 'app_v1', 'model': self.settings.default_model,
            'components': {**self.components, 'model': 'ready' if model_ready else 'model_unavailable'},
            'details': self.component_details}

    @staticmethod
    def _conversation_messages(request, current_user):
        history = [message.model_dump() for message in request.history]
        if (history and history[-1]['role'] == 'user'
                and history[-1]['content'] == request.message):
            history.pop()

        # Mistral's llama.cpp template requires the transcript after the
        # system prompt to start with a user and strictly alternate roles.
        while history and history[0]['role'] == 'assistant':
            history.pop(0)
        history.append({'role': 'user', 'content': current_user})

        alternating = []
        for message in history:
            if alternating and alternating[-1]['role'] == message['role']:
                alternating[-1]['content'] += '\n\n' + message['content']
            else:
                alternating.append(dict(message))
        return alternating

    def _prepare(self, request):
        from src.chat import prepare_turn
        turn = prepare_turn(request.message, self.turn_settings, self.condition,
                            self.router, self.training_texts, self.retriever)
        if not turn.hits:
            raise ServiceError('retrieval_empty', 'Supporting evidence is unavailable.')
        # Upstream system prompt is left intact as the prefix. Context is data in
        # a user message, not a second user-controlled system instruction.
        system = turn.system_prompt + '\n\n' + self.application['contract']
        context = json.dumps(request.screening_context.model_dump(), ensure_ascii=False)
        messages = [{'role': 'system', 'content': system}]
        current_user = ('<read_only_screening_context>\n' + context +
            '\n</read_only_screening_context>\n\nUser message:\n' + request.message)
        messages.extend(self._conversation_messages(request, current_user))
        return turn, messages

    async def chat(self, request):
        if not all(v == 'ready' for v in self.components.values()):
            raise ServiceError('runtime_unavailable', 'The assistant requires its configured router and source corpus.')
        if self._lock.locked():
            raise ServiceError('capacity_exhausted', 'The assistant is busy. Please retry.')
        async with self._lock:
            try:
                turn, messages = await asyncio.to_thread(self._prepare, request)
            except ServiceError:
                raise
            except Exception as exc:
                raise ServiceError('retrieval_unavailable', 'Supporting evidence could not be prepared.') from exc
            try:
                result = await asyncio.wait_for(self.adapter.generate(messages, request.model),
                                                timeout=self.settings.timeout_seconds + 2)
            except asyncio.TimeoutError as exc:
                raise ServiceError('model_timeout', 'The assistant took too long to respond.', 504) from exc
            return ChatResponse(request_id=request.request_id, session_id=request.session_id,
                context_revision=request.screening_context.revision, response=result.text,
                route=turn.route_info['route'], model=request.model,
                sources=[Source(number=s['number'], title=s['name'], url=s['url'],
                    authority=s['authority'], low_authority=s['low_authority'],
                    passage_ids=s['passage_ids']) for s in turn.sources],
                metadata=ChatMetadata(runtime_profile='app_v1', router_config='embeddings_word',
                    route_decided_by=turn.route_info['decided_by'], rag_used=bool(turn.hits),
                    corpus_sha256=self.corpus_hash, prompts_sha256=self.prompt_hash,
                    training_sha256=self.training_hash,
                    application_prompt_version=self.application['version'],
                    model_identity=self.settings.model_identities.get(request.model, request.model),
                    finish_reason=result.finish_reason))

    async def close(self):
        await self.adapter.close()
