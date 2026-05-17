"""pgvector 기반 영구 벡터 저장소.

InMemoryVectorStore와 동일한 VectorStore 인터페이스를 구현하며,
임베딩 벡터를 PostgreSQL + pgvector에 영구 저장한다.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy import Engine, create_engine, text

from src.embedding_study._interface import VectorStore


class DatabaseVectorStore(VectorStore):
    """PostgreSQL + pgvector 기반 벡터 저장소.

    Usage:
        store = DatabaseVectorStore("postgresql://localhost:5432/pytool", dim=384)
        store.add(["doc1"], ["hello world"], vectors)
        results = store.search(query_vector, top_k=5)
    """

    def __init__(self, connection_string: str, dim: int = 384) -> None:
        self._engine = create_engine(connection_string)
        self._dim = dim
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create pgvector extension and table if not exists."""
        with self._engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(
                text(
                    f"""CREATE TABLE IF NOT EXISTS embeddings (
                        id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        vector vector({self._dim})
                    )"""
                )
            )
            # IVFFlat 인덱스는 데이터가 쌓인 후 생성해야 하지만,
            # 소규모 데모에서는 미리 생성해도 무방하다.
            conn.execute(
                text(
                    """CREATE INDEX IF NOT EXISTS embeddings_vector_idx
                    ON embeddings USING ivfflat (vector vector_cosine_ops)"""
                )
            )

    def add(self, ids: list[str], texts: list[str], vectors: np.ndarray) -> None:
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2-dimensional")
        if len(ids) != len(texts) or len(ids) != len(vectors):
            raise ValueError("ids, texts, vectors must have same length")
        if vectors.shape[1] != self._dim:
            raise ValueError(
                f"vector dim mismatch: expected {self._dim}, got {vectors.shape[1]}"
            )

        with self._engine.begin() as conn:
            for id_, text_, vec in zip(ids, texts, vectors):
                vec_literal = "[" + ",".join(str(float(v)) for v in vec) + "]"
                conn.execute(
                    text(
                        "INSERT INTO embeddings (id, text, vector) "
                        "VALUES (:id, :text, :vector::vector) "
                        "ON CONFLICT (id) DO UPDATE SET text=:text, vector=:vector::vector"
                    ),
                    {"id": id_, "text": text_, "vector": vec_literal},
                )

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> list[tuple[str, str, float]]:
        vec_literal = "[" + ",".join(str(float(v)) for v in query_vector) + "]"

        with self._engine.connect() as conn:
            result = conn.execute(
                text(
                    """SELECT id, text, 1 - (vector <=> :query::vector) AS similarity
                    FROM embeddings
                    ORDER BY vector <=> :query::vector
                    LIMIT :top_k"""
                ),
                {"query": vec_literal, "top_k": top_k},
            )
            return [(row[0], row[1], float(row[2])) for row in result]

    def __len__(self) -> int:
        with self._engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM embeddings"))
            return result.scalar_one()
