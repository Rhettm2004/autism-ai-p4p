import asyncio
import hashlib
import json
import logging
import re
import yaml
from app.corpus import verify_corpus
from app.errors import ServiceError
from app.schemas import (ChatResponse, ChatMetadata, CommandResult, Source,
                         StartScreeningAction)
from app.adapters.llama_cpp import LlamaCppAdapter

log = logging.getLogger(__name__)

class ChatRuntime:
    _explicit_start = re.compile(
        r'\b(start|begin|take|do|complete)\b.{0,24}\b(screening|questionnaire|test)\b|'
        r'\b(screening|questionnaire|test)\b.{0,24}\b(start|begin|take|do|complete)\b',
        re.IGNORECASE,
    )
    _affirmatives = {
        'yes', 'yes please', 'yeah', 'yep', 'sure', 'okay', 'ok',
        'lets start', "let's start", 'ready', "i'm ready", 'i am ready',
    }
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
            if self.cfg.get('prompt_profile') not in ('rayaan_chat_exact', 'app_context_v1'):
                raise ValueError('Unsupported prompt profile')
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
            prompt_bytes = (root / 'config/prompts.yaml').read_bytes()
            if self.cfg['prompt_profile'] == 'app_context_v1':
                prompt_bytes += (root / 'config/application_prompts.yaml').read_bytes()
            self.prompt_hash = hashlib.sha256(prompt_bytes.strip()).hexdigest()
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
        from src.chat import DemoSettings, prepare_turn
        settings = DemoSettings(
            use_router=request.options.router,
            use_rag=request.options.rag,
            cite=request.options.cite,
            concise=request.options.concise,
            top_k=self.turn_settings.top_k,
            expand_neighbours=self.turn_settings.expand_neighbours,
        )
        turn = prepare_turn(request.message, settings, self.condition,
                            self.router, self.training_texts, self.retriever)
        if settings.use_rag and not turn.hits:
            raise ServiceError('retrieval_empty', 'Supporting evidence is unavailable.')
        if self.cfg['prompt_profile'] == 'rayaan_chat_exact':
            # Match scripts/chat.py: each turn is independent and the model sees
            # the prepared research prompt plus the exact question, byte for byte.
            messages = [
                {'role': 'system', 'content': turn.system_prompt},
                {'role': 'user', 'content': request.message},
            ]
        else:
            system = turn.system_prompt + '\n\n' + self.application['contract']
            context = json.dumps(request.screening_context.model_dump(), ensure_ascii=False)
            messages = [{'role': 'system', 'content': system}]
            current_user = ('<read_only_screening_context>\n' + context +
                '\n</read_only_screening_context>\n\nUser message:\n' + request.message)
            messages.extend(self._conversation_messages(request, current_user))
        return turn, messages

    def _metadata(self, request, turn=None, finish_reason=None, invalid_citations=None):
        route_info = turn.route_info if turn is not None else None
        exact = self.cfg['prompt_profile'] == 'rayaan_chat_exact'
        return ChatMetadata(runtime_profile='app_v1', prompt_profile=self.cfg['prompt_profile'],
            router_config='embeddings_word',
            route_decided_by=(route_info or {}).get('decided_by', 'command_or_disabled'),
            rag_used=bool(turn and turn.hits), corpus_sha256=self.corpus_hash,
            prompts_sha256=self.prompt_hash, training_sha256=self.training_hash,
            application_prompt_version=None if exact else self.application['version'],
            model_identity=self.settings.model_identities.get(request.model, request.model),
            finish_reason=finish_reason,
            invalid_citations=invalid_citations or [])

    @staticmethod
    def _sources(turn=None, sources=None):
        sources = sources if sources is not None else turn.sources
        return [Source(number=s['number'], title=s['name'], url=s['url'],
            authority=s['authority'], low_authority=s['low_authority'],
            passage_ids=s['passage_ids'], cited=bool(s.get('cited', False)))
            for s in sources]

    def _command_response(self, request, text, name, arg=None, options=None,
                          turn=None, executed_question=None):
        route = ((turn.route_info or {}).get('route') if turn else None) or 'general_knowledge'
        return ChatResponse(request_id=request.request_id, session_id=request.session_id,
            context_revision=request.screening_context.revision, response=text,
            route=route, model=request.model, sources=self._sources(turn) if turn else [],
            options=options or request.options,
            command=CommandResult(name=name, arg=arg, executed_question=executed_question),
            metadata=self._metadata(request, turn))

    @classmethod
    def _wants_to_start_screening(cls, request):
        if request.screening_context.stage != 'welcome':
            return False
        normalized = ' '.join(request.message.lower().strip().split())
        if normalized == '/start':
            return True
        if any(phrase in normalized for phrase in (
            "don't start", 'do not start', 'not ready', 'no screening',
            "don't want", 'do not want',
        )):
            return False
        if cls._explicit_start.search(normalized):
            return True
        simple = normalized.strip(' .!?')
        if simple not in cls._affirmatives:
            return False
        previous_assistant = next(
            (message.content.lower() for message in reversed(request.history)
             if message.role == 'assistant'),
            '',
        )
        return ('start a screening' in previous_assistant
                or 'begin a screening' in previous_assistant)

    def _start_screening_response(self, request):
        return ChatResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            context_revision=request.screening_context.revision,
            response='Of course. Let’s begin with a few details to select the appropriate questionnaire.',
            route='screening_guidance',
            model=request.model,
            sources=[],
            options=request.options,
            action=StartScreeningAction(
                type='start_screening',
                expected_context_revision=request.screening_context.revision,
            ),
            metadata=self._metadata(request),
        )

    def _handle_command(self, request):
        from src.chat import COMMANDS, load_examples, parse_command
        try:
            command = parse_command(request.message)
        except ValueError as exc:
            return self._command_response(request, str(exc), 'invalid'), request
        if command is None:
            return None, request
        if command.name == 'help':
            text = '\n'.join([
                'Available commands:',
                '/examples — list Rayaan’s scripted demo questions',
                '/ex N — ask demo question N',
                '/prompt — show the prompt built for the previous question',
                '/cite on|off — toggle experimental inline citation instructions',
                '/concise on|off — toggle concise chat-style instructions',
                '/router on|off — toggle route-conditioned guidance',
                '/rag on|off — toggle retrieval',
                '/start — begin a screening from the welcome conversation',
                '/quit or /exit — explain how to leave the web demo',
            ])
            return self._command_response(request, text, command.name), request
        if command.name == 'examples':
            examples = load_examples()
            text = 'Rayaan’s scripted demo questions:\n\n' + '\n\n'.join(
                f"{index}. {example['question']}\n   {example['shows'].strip()}\n"
                f"   From: {example['from']}"
                for index, example in enumerate(examples, start=1))
            return self._command_response(request, text, command.name), request
        if command.name in ('quit', 'exit'):
            text = ('This command exits Rayaan’s terminal demo. In the web app, '
                    'close the tab or stop ./run_app.sh with Ctrl+C.')
            return self._command_response(request, text, command.name), request
        if command.name in ('cite', 'concise', 'router', 'rag'):
            field = command.name
            options = request.options.model_copy(update={field: command.arg == 'on'})
            note = ''
            if options.cite and not options.rag:
                note = '\nCitations need retrieval; /cite has no effect while /rag is off.'
            text = f'/{command.name} {command.arg}{note}'
            return self._command_response(
                request, text, command.name, command.arg, options=options), request
        if command.name == 'prompt':
            prior = next((m.content for m in reversed(request.history)
                          if m.role == 'user' and not m.content.startswith('/')), None)
            if prior is None:
                return self._command_response(request, 'No previous question yet.', 'prompt'), request
            prompt_request = request.model_copy(update={'message': prior})
            turn, messages = self._prepare(prompt_request)
            text = ('System prompt for the previous question:\n\n' + messages[0]['content']
                    + '\n\nQUESTION\n' + prior)
            return self._command_response(
                request, text, 'prompt', turn=turn, executed_question=prior), request
        if command.name == 'ex':
            examples = load_examples()
            if command.arg > len(examples):
                text = f'There are {len(examples)} demo questions; use /examples.'
                return self._command_response(request, text, 'ex', command.arg), request
            question = examples[command.arg - 1]['question']
            return None, request.model_copy(update={'message': question})
        # Keep this exhaustive if Rayaan adds another command.
        available = ', '.join(f'/{name}' for name in COMMANDS)
        return self._command_response(request, f'Unsupported command. Available: {available}', 'invalid'), request

    async def chat(self, request):
        if not all(v == 'ready' for v in self.components.values()):
            raise ServiceError('runtime_unavailable', 'The assistant requires its configured router and source corpus.')
        if self._wants_to_start_screening(request):
            return self._start_screening_response(request)
        handled, effective_request = self._handle_command(request)
        if handled is not None:
            return handled
        if self._lock.locked():
            raise ServiceError('capacity_exhausted', 'The assistant is busy. Please retry.')
        async with self._lock:
            try:
                turn, messages = await asyncio.to_thread(self._prepare, effective_request)
            except ServiceError:
                raise
            except Exception as exc:
                raise ServiceError('retrieval_unavailable', 'Supporting evidence could not be prepared.') from exc
            try:
                result = await asyncio.wait_for(self.adapter.generate(messages, request.model),
                                                timeout=self.settings.timeout_seconds + 2)
            except asyncio.TimeoutError as exc:
                raise ServiceError('model_timeout', 'The assistant took too long to respond.', 504) from exc

            response_text = result.text
            response_sources = turn.sources
            invalid_citations = []
            if effective_request.options.cite and turn.sources:
                from src.chat import order_sources_by_citation, renumber_citations
                valid = {source['number'] for source in turn.sources}
                response_text, mapping, invalid_citations = renumber_citations(
                    result.text, valid)
                response_sources = [
                    source for source in order_sources_by_citation(
                        turn.sources, mapping)
                    if source['cited']
                ]

            return ChatResponse(request_id=request.request_id, session_id=request.session_id,
                context_revision=request.screening_context.revision, response=response_text,
                route=(turn.route_info or {}).get('route') or 'general_knowledge',
                model=request.model, sources=self._sources(sources=response_sources),
                options=effective_request.options,
                command=(CommandResult(name='ex', executed_question=effective_request.message)
                         if effective_request is not request else None),
                metadata=self._metadata(request, turn, result.finish_reason,
                                        invalid_citations))

    async def close(self):
        await self.adapter.close()
