"""프로젝트 환경 설정 — .env 로드 및 환경 변수 접근을 중앙화한다.

사용법:
    from src.utils import config

    api_key = config.anthropic_api_key
    model = config.anthropic_model
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# 프로젝트 루트의 .env 를 자동 로드 (중복 호출 무시)
_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"  # env_config.py -> utils -> src -> root
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)
else:
    load_dotenv()


class Config:
    """환경 변수 접근을 위한 중앙 설정 객체.

    각 속성은 대응하는 환경 변수를 읽는다.
    필수 키가 없으면 ImportError 대신 지연 실패로 빈 문자열을 반환하므로
    호출부에서 명시적으로 검사한다.
    """

    @property
    def anthropic_api_key(self) -> str:
        """ANTHROPIC_API_KEY — Claude API 호출용."""
        return os.environ.get("ANTHROPIC_API_KEY", "")

    @property
    def anthropic_model(self) -> str:
        """ANTHROPIC_MODEL — 사용할 Claude 모델명."""
        return os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    @property
    def deepseek_api_key(self) -> str:
        """DEEPSEEK_API_KEY — DeepSeek API 호출용."""
        return os.environ.get("DEEPSEEK_API_KEY", "")

    @property
    def ollama_base_url(self) -> str:
        """OLLAMA_BASE_URL — Ollama 서버 주소."""
        return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

    @property
    def ollama_embed_model(self) -> str:
        """OLLAMA_EMBED_MODEL — Ollama 임베딩 모델명."""
        return os.environ.get("OLLAMA_EMBED_MODEL", "mxbai-embed-large:335m")


config = Config()
