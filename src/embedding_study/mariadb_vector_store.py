"""MariaDB 11.7 VECTOR 기반 영구 벡터 저장소.

VectorStore ABC를 구현하며, MariaDB의 네이티브 VECTOR 타입과
VEC_DISTANCE_COSINE 함수를 사용한다.

요구사항:
    - MariaDB 11.7 이상 (VECTOR 타입 지원)
    - pymysql Python 드라이버

MariaDB 11.7 Docker 실행:
    docker run -d --name mariadb-vector \
        -e MARIADB_ROOT_PASSWORD=password \
        -e MARIADB_DATABASE=pytool \
        -p 3306:3306 \
        mariadb:11.7
"""

from __future__ import annotations

import numpy as np
import pymysql

from src.embedding_study._interface import VectorStore


class MariaDBVectorStore(VectorStore):
    """MariaDB 11.7 VECTOR 기반 벡터 저장소.

    Usage:
        store = MariaDBVectorStore(
            host="localhost", user="root", password="password",
            database="pytool", dim=384,
        )
        store.add(["doc1"], ["hello world"], vectors)
        results = store.search(query_vector, top_k=5)
    """

    def __init__(
        self,
        host: str = "localhost",
        user: str = "root",
        password: str = "",
        database: str = "pytool",
        port: int = 3306,
        dim: int = 384,
    ) -> None:
        self._conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port,
        )
        self._dim = dim
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create table with VECTOR column and index if not exists."""
        with self._conn.cursor() as cur:
            cur.execute(
                f"""CREATE TABLE IF NOT EXISTS embeddings (
                    id VARCHAR(256) PRIMARY KEY,
                    text TEXT NOT NULL,
                    vector VECTOR({self._dim}) NOT NULL,
                    VECTOR INDEX (vector) M=16 DISTANCE=cosine
                )"""
            )
        self._conn.commit()

    def _to_vec_text(self, vec: np.ndarray) -> str:
        """Convert numpy array to Vec_FromText compatible string."""
        return "[" + ",".join(str(float(v)) for v in vec) + "]"

    def add(self, ids: list[str], texts: list[str], vectors: np.ndarray) -> None:
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2-dimensional")
        if len(ids) != len(texts) or len(ids) != len(vectors):
            raise ValueError("ids, texts, vectors must have same length")
        if vectors.shape[1] != self._dim:
            raise ValueError(
                f"vector dim mismatch: expected {self._dim}, got {vectors.shape[1]}"
            )

        with self._conn.cursor() as cur:
            for id_, text_, vec in zip(ids, texts, vectors):
                vec_text = self._to_vec_text(vec)
                cur.execute(
                    """INSERT INTO embeddings (id, text, vector)
                    VALUES (%s, %s, Vec_FromText(%s))
                    ON DUPLICATE KEY UPDATE
                        text = VALUES(text), vector = Vec_FromText(%s)""",
                    (id_, text_, vec_text, vec_text),
                )
        self._conn.commit()

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> list[tuple[str, str, float]]:
        vec_text = self._to_vec_text(query_vector)

        with self._conn.cursor() as cur:
            # VEC_DISTANCE_COSINE returns cosine distance [0, 2].
            # Cosine similarity = 1 - cosine_distance.
            cur.execute(
                """SELECT id, text,
                    1 - VEC_DISTANCE_COSINE(vector, Vec_FromText(%s)) AS similarity
                FROM embeddings
                ORDER BY VEC_DISTANCE_COSINE(vector, Vec_FromText(%s))
                LIMIT %s""",
                (vec_text, vec_text, top_k),
            )
            return [
                (row[0], row[1], float(row[2]) if row[2] is not None else 0.0)
                for row in cur
            ]

    def __len__(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM embeddings")
            return cur.fetchone()[0]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
