"""Tests for DatabaseVectorStore — requires PostgreSQL + pgvector.

사전 준비:
    docker run -d --name pytool-pgvector -p 5432:5432 \
        -e POSTGRES_PASSWORD=pytool -e POSTGRES_DB=pytool \
        pgvector/pgvector:pg17

    또는 로컬 PostgreSQL에 pgvector 확장 설치:
        CREATE EXTENSION vector;
"""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import create_engine, text


# 연결 가능한 경우에만 테스트 실행
def _pgvector_available() -> bool:
    try:
        engine = create_engine("postgresql://postgres:pytool@localhost:5432/pytool")
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


DB_URL = "postgresql://postgres:pytool@localhost:5432/pytool"
reason = "PostgreSQL+pgvector not available (docker run pgvector/pgvector:pg17)"

pytestmark = pytest.mark.skipif(not _pgvector_available(), reason=reason)


@pytest.fixture
def store():
    """Create a fresh DatabaseVectorStore for each test."""
    from src.embedding_study.db_vector_store import DatabaseVectorStore

    store = DatabaseVectorStore(DB_URL, dim=4)
    # Clean up before each test
    with store._engine.begin() as conn:
        conn.execute(text("DELETE FROM embeddings"))
    return store


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------


def test_add_and_search_returns_similar_documents(store):
    # Given
    ids = ["doc1", "doc2", "doc3"]
    texts = ["apple fruit", "banana fruit", "computer technology"]
    vectors = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # apple (fruit)
            [0.9, 0.1, 0.0, 0.0],  # banana (fruit, similar to apple)
            [0.0, 0.0, 1.0, 0.0],  # computer (technology, different)
        ],
        dtype=np.float64,
    )

    # When
    store.add(ids, texts, vectors)

    # Then — "fruit"에 가까운 쿼리는 apple, banana 순서로 반환되어야 함
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    results = store.search(query, top_k=2)

    assert len(results) == 2
    assert results[0][0] == "doc1"  # apple (exact match)
    assert results[0][1] == "apple fruit"
    assert results[1][0] == "doc2"  # banana (similar)
    assert results[0][2] > results[1][2]  # 첫 번째가 더 높은 유사도


def test_search_returns_empty_for_empty_store(store):
    # When
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    results = store.search(query, top_k=5)

    # Then
    assert results == []


def test_len_returns_count(store):
    # Given
    vectors = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    store.add(["doc1"], ["hello"], vectors)

    # When
    length = len(store)

    # Then
    assert length == 1


def test_upsert_updates_existing_document(store):
    # Given
    vectors = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
    store.add(["doc1"], ["hello"], vectors)

    # When — 같은 ID로 다른 텍스트, 다른 벡터를 덮어쓰기
    new_vectors = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
    store.add(["doc1"], ["computer"], new_vectors)

    # Then
    assert len(store) == 1
    query = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    results = store.search(query, top_k=1)
    assert results[0][1] == "computer"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_add_raises_on_dimension_mismatch(store):
    # Given — store는 dim=4인데 3차원 벡터 전달
    vectors = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

    # When / Then
    with pytest.raises(ValueError, match="dim mismatch"):
        store.add(["doc1"], ["hello"], vectors)


def test_add_raises_on_non_2d_vectors(store):
    # Given — 1차원 벡터
    vectors = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # When / Then
    with pytest.raises(ValueError, match="2-dimensional"):
        store.add(["doc1"], ["hello"], vectors)


def test_add_raises_on_length_mismatch(store):
    # Given
    vectors = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)

    # When / Then
    with pytest.raises(ValueError, match="same length"):
        store.add(["doc1", "doc2"], ["only one text"], vectors)


# ---------------------------------------------------------------------------
# Similarity ordering
# ---------------------------------------------------------------------------


def test_search_orders_by_cosine_similarity_descending(store):
    # Given — 세 문서의 방향이 점점 멀어짐
    ids = ["center", "near", "far"]
    texts = ["center doc", "near doc", "far doc"]
    vectors = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],  # center
            [0.7, 0.7, 0.0, 0.0],  # near: center와 45도 → cosine ≈ 0.7
            [0.0, 1.0, 0.0, 0.0],  # far: center와 90도 → cosine ≈ 0.0
        ],
        dtype=np.float64,
    )
    store.add(ids, texts, vectors)

    # When
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    results = store.search(query, top_k=3)

    # Then — 유사도 내림차순
    assert results[0][0] == "center"  # 가장 가까움
    assert results[2][0] == "far"  # 가장 멂
    scores = [r[2] for r in results]
    assert scores == sorted(scores, reverse=True)
