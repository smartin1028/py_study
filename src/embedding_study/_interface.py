"""Public interfaces for embedding_study package."""

import abc

import numpy as np


class Embedder(abc.ABC):
    """텍스트를 벡터로 변환하는 인터페이스."""

    @abc.abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """텍스트 리스트를 임베딩 벡터 배열로 변환한다.

        Returns:
            shape=(len(texts), dim) 인 2차원 float 배열.
        """
        ...

    @abc.abstractmethod
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """두 벡터 간 코사인 유사도를 계산한다."""
        ...

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """임베딩 차원 수."""
        ...


class VectorStore(abc.ABC):
    """임베딩 벡터를 저장하고 검색하는 인터페이스."""

    @abc.abstractmethod
    def add(self, ids: list[str], texts: list[str], vectors: np.ndarray) -> None:
        """벡터와 메타데이터를 저장한다."""
        ...

    @abc.abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[str, str, float]]:
        """쿼리 벡터와 가장 유사한 문서를 검색한다.

        Returns:
            [(id, text, score), ...] 형태의 리스트. score는 내림차순.
        """
        ...
