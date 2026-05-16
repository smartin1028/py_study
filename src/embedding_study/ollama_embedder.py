"""Ollama 로컬 API 기반 임베딩 구현체.

사용 전:
    brew install ollama  # 또는 https://ollama.com
    ollama pull nomic-embed-text
    ollama serve          # localhost:11434
"""

from __future__ import annotations

import numpy as np
import requests

from src.embedding_study._interface import Embedder
from src.embedding_study.basics import cosine_similarity_manual
from src.utils import config


class OllamaEmbedder(Embedder):
    """Ollama 로컬 API를 사용하는 임베딩 구현체.

    기본값은 .env의 OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL 환경 변수로 설정한다.

    Usage:
        embedder = OllamaEmbedder()
        vectors = embedder.encode(["hello world", "안녕하세요"])
        sim = embedder.similarity(vectors[0], vectors[1])
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model if model is not None else config.ollama_embed_model
        self._base_url = (base_url if base_url is not None else config.ollama_base_url).rstrip("/")
        self._dim: int | None = None

    def encode(self, texts: list[str]) -> np.ndarray:
        resp = requests.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": texts},
            timeout=30,
        )
        resp.raise_for_status()
        embeddings = np.array(resp.json()["embeddings"], dtype=np.float32)

        if self._dim is None and embeddings.size > 0:
            self._dim = embeddings.shape[1]

        return embeddings

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return cosine_similarity_manual(a, b)

    @property
    def dim(self) -> int:
        if self._dim is None:
            dummy = self.encode([""])
            self._dim = dummy.shape[1]
        return self._dim
