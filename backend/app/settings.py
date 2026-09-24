from dataclasses import dataclass, field
from pathlib import Path
import os
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    model_urls: dict[str, str] = field(default_factory=lambda: {
        'mistral': 'http://127.0.0.1:8080', 'llama': 'http://127.0.0.1:8081'})
    model_identities: dict[str, str] = field(default_factory=dict)
    default_model: str = 'mistral'
    cors_origins: tuple[str, ...] = ('http://localhost:3000',)
    timeout_seconds: float = 60

    @classmethod
    def load(cls):
        load_dotenv(ROOT / '.env', override=False)
        cfg = yaml.safe_load((ROOT / 'config/runtime.yaml').read_text())
        urls = {name: os.getenv(f'AUTISM_AI_{name.upper()}_URL', value['url'])
                for name, value in cfg['models'].items()}
        default = os.getenv('AUTISM_AI_DEFAULT_MODEL', 'mistral')
        if default not in urls:
            raise ValueError('Unsupported default model')
        for url in urls.values():
            from urllib.parse import urlparse
            parsed = urlparse(url)
            if parsed.scheme not in ('http', 'https') or not parsed.hostname:
                raise ValueError('Model URL must be HTTP(S)')
        return cls(model_urls=urls,
                   model_identities={n: v['identity'] for n, v in cfg['models'].items()},
                   default_model=default,
                   cors_origins=tuple(x.strip() for x in os.getenv(
                       'AUTISM_AI_CORS_ORIGINS', 'http://localhost:3000').split(',') if x.strip()))
