from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT_PATH = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT_PATH / ".env"

# Load local project .env only.
if ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=False)


def _env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip()


def _env_int(name: str, default: int) -> int:
    raw = _env_str(name, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    PROJECT_ROOT: str
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str
    OPENAI_MODEL: str
    SKILLHUB_API_KEY: str
    SKILLHUB_BASE_URL: str
    SKILLHUB_TIMEOUT_SECONDS: int


settings = Settings(
    PROJECT_ROOT=str(PROJECT_ROOT_PATH),
    OPENAI_API_KEY=_env_str("OPENAI_API_KEY", ""),
    OPENAI_BASE_URL=_env_str("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    OPENAI_MODEL=_env_str("OPENAI_MODEL", "gpt-4o-mini"),
    SKILLHUB_API_KEY=_env_str("SKILLHUB_API_KEY", ""),
    SKILLHUB_BASE_URL=_env_str("SKILLHUB_BASE_URL", "https://www.skillhub.club/api/v1"),
    SKILLHUB_TIMEOUT_SECONDS=_env_int("SKILLHUB_TIMEOUT_SECONDS", 20),
)
