"""임베딩 생성과 유사도 계산 기초.

이 모듈은 sentence-transformers를 사용해 텍스트를 벡터로 변환하고,
벡터 간 유사도를 계산하는 기본기를 다룬다.

실행 시간: 최초 실행 시 모델을 다운로드하므로 수 초~수십 초 소요.
"""

from __future__ import annotations

import logging
import os
import warnings

# 모델 로딩 시 불필요한 progress bar와 경고를 끈다.
# 모델은 최초 1회만 다운로드되며 이후 로컬 캐시에서 즉시 로딩된다.
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)

import numpy as np
from sentence_transformers import SentenceTransformer

from src.embedding_study._interface import Embedder


# Sentence-BERT 경량 모델. 384차원, 80MB, CPU에서도 빠르게 동작한다.
# 최초 사용 시 자동으로 다운로드된다.
_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SentenceEmbedder(Embedder):
    """sentence-transformers 기반 임베딩 구현체.

    Usage:
        embedder = SentenceEmbedder()
        vectors = embedder.encode(["hello world", "안녕하세요"])
        sim = embedder.similarity(vectors[0], vectors[1])
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        # 코사인 유사도 = (a·b) / (||a|| × ||b||)
        # 정규화된 벡터끼리는 내적만으로도 코사인 유사도를 얻을 수 있다.
        a_norm = a / (np.linalg.norm(a) + 1e-10)
        b_norm = b / (np.linalg.norm(b) + 1e-10)
        return float(np.dot(a_norm, b_norm))

    @property
    def dim(self) -> int:
        return self._model.get_embedding_dimension()


def demonstrate_embedding_basics() -> dict:
    """임베딩 기본 개념을 시연한다.

    Returns:
        {
            "vector_shape": 벡터의 shape,
            "vector_preview": 벡터 앞 5개 값 (실제로는 384차원),
            "similar_meaning": 유사한 의미의 문장 쌍 유사도,
            "different_meaning": 다른 의미의 문장 쌍 유사도,
        }
    """
    embedder = SentenceEmbedder()

    # all-MiniLM-L6-v2는 영어 전용 모델이므로 영어 문장으로 시연한다.
    # 다국어 임베딩이 필요하면 intfloat/multilingual-e5-large 모델을 사용한다.
    texts = [
        "The weather is beautiful today",
        "It is sunny and warm outside with clear skies",
        "Python is a popular programming language for data science",
    ]
    vectors = embedder.encode(texts)

    # 2. 벡터 정보
    vector_info = {
        "vector_shape": str(vectors.shape),
        "vector_dim": embedder.dim,
        "vector_preview": [round(float(v), 6) for v in vectors[0][:5]],
    }

    # 3. 유사도 비교 — 의미가 가까운 문장 vs 먼 문장
    sim_similar = embedder.similarity(vectors[0], vectors[1])  # 둘 다 날씨 이야기
    sim_different = embedder.similarity(vectors[0], vectors[2])  # 날씨 vs 프로그래밍

    return {
        **vector_info,
        "similar_meaning": round(sim_similar, 4),    # 높은 값 (0.5~0.9)
        "different_meaning": round(sim_different, 4),  # 낮은 값 (0.0~0.4)
        "texts": texts,
    }


def cosine_similarity_manual(a: np.ndarray, b: np.ndarray) -> float:
    """코사인 유사도를 직접 계산하는 방법 (학습용).

    cos(θ) = (A·B) / (|A| × |B|)

    - 1에 가까울수록 두 벡터의 방향이 같다 (의미가 유사).
    - 0에 가까울수록 직교한다 (의미가 무관).
    - -1에 가까울수록 반대 방향이다 (거의 쓰이지 않음).
    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return float(dot / (norm_a * norm_b + 1e-10))
