import httpx
from app.adapters.base import GenerationResult
from app.errors import ServiceError

class LlamaCppAdapter:
    def __init__(self, urls: dict[str, str], timeout: float = 60,
                 client: httpx.AsyncClient | None = None):
        self.urls = {k: v.rstrip('/') for k, v in urls.items()}
        self.timeout = timeout
        self._owned = client is None
        self.client = client or httpx.AsyncClient(timeout=timeout)

    async def ready(self, model: str) -> bool:
        try:
            response = await self.client.get(f'{self.urls[model]}/health', timeout=2)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def generate(self, messages, model):
        if model not in self.urls:
            raise ServiceError('unsupported_model', 'Unsupported model.', 422, False)
        # llama.cpp defaults differ from Transformers. These are an explicit app
        # profile, NOT equivalence to Rayaan's completion-only logits processors.
        body = dict(model=model, messages=messages, stream=False, temperature=0.1,
                    max_tokens=256, top_k=50, top_p=1.0, min_p=0.0,
                    repeat_penalty=1.0, presence_penalty=0.0, frequency_penalty=0.0)
        try:
            response = await self.client.post(f'{self.urls[model]}/v1/chat/completions',
                                              json=body, timeout=self.timeout)
        except httpx.TimeoutException as exc:
            raise ServiceError('model_timeout', 'The assistant took too long to respond.', 504) from exc
        except httpx.HTTPError as exc:
            raise ServiceError('model_unavailable', 'The model server is unavailable.') from exc
        if response.status_code >= 400:
            raise ServiceError('model_unavailable', 'The model server returned an error.')
        try:
            choice = response.json()['choices'][0]
            message = choice['message']
            content = message['content']
            # Tool calls and reasoning are not application actions. Never expose them.
            if message.get('tool_calls') or not isinstance(content, str) or not content.strip():
                raise ValueError('Missing visible answer')
            if any(marker in content.lower() for marker in ('<think>', '</think>', '<analysis>')):
                raise ValueError('Unexpected reasoning channel')
            finish = choice.get('finish_reason')
            if finish is not None and not isinstance(finish, str):
                raise ValueError('Invalid finish reason')
            return GenerationResult(content.strip(), finish)
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ServiceError('invalid_model_response', 'The model response could not be read.', 502) from exc

    async def close(self):
        if self._owned:
            await self.client.aclose()
