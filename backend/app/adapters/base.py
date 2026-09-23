from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class GenerationResult:
    text: str
    finish_reason: str | None = None

class ModelAdapter(Protocol):
    async def ready(self, model: str) -> bool: ...
    async def generate(self, messages: list[dict[str, str]], model: str) -> GenerationResult: ...
    async def close(self) -> None: ...
