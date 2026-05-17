"""Oracle 23ai VECTOR 기반 영구 벡터 저장소.

DatabaseVectorStore와 동일한 VectorStore 인터페이스를 구현하며,
Oracle 23ai의 네이티브 VECTOR 타입과 VECTOR_DISTANCE 함수를 사용한다.

요구사항:
    - Oracle Database 23ai 이상 (VECTOR 타입 지원)
    - oracledb Python 드라이버

Oracle 23ai Free Docker 실행:
    docker run -d --name oracle23ai -p 1521:1522 \
        -e ORACLE_PWD=password \
        container-registry.oracle.com/database/free:latest
"""

from __future__ import annotations

import numpy as np
import oracledb

from src.embedding_study._interface import VectorStore


class OracleVectorStore(VectorStore):
    """Oracle 23ai VECTOR 기반 벡터 저장소.

    Usage:
        store = OracleVectorStore(
            user="pytool", password="password",
            dsn="localhost:1522/FREEPDB1", dim=384,
        )
        store.add(["doc1"], ["hello world"], vectors)
        results = store.search(query_vector, top_k=5)
    """

    def __init__(
        self,
        user: str,
        password: str,
        dsn: str,
        dim: int = 384,
    ) -> None:
        self._conn = oracledb.connect(user=user, password=password, dsn=dsn)
        self._dim = dim
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create table and vector index if not exists."""
        with self._conn.cursor() as cur:
            cur.execute(
                f"""CREATE TABLE IF NOT EXISTS embeddings (
                    id VARCHAR2(256) PRIMARY KEY,
                    text CLOB,
                    vector VECTOR({self._dim}, FLOAT64)
                )"""
            )
        self._conn.commit()

        # CREATE VECTOR INDEX IF NOT EXISTS 는 Oracle에 없으므로
        # 이미 존재하는 경우 ORA-01408(이미 인덱스 있음)를 무시한다.
        try:
            with self._conn.cursor() as cur:
                cur.execute(
                    f"""CREATE VECTOR INDEX embeddings_vector_idx
                    ON embeddings (vector)
                    ORGANIZATION NEIGHBOR PARTITIONS
                    DISTANCE COSINE
                    WITH TARGET ACCURACY 95"""
                )
            self._conn.commit()
        except oracledb.DatabaseError as e:
            # ORA-01408: such column list already indexed
            if "ORA-01408" not in str(e):
                raise

    def _to_vector_list(self, vec: np.ndarray) -> list[float]:
        """Convert numpy array to plain float list for Oracle VECTOR binding."""
        return [float(v) for v in vec]

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
                vec_list = self._to_vector_list(vec)
                cur.execute(
                    """MERGE INTO embeddings e
                    USING (SELECT :1 AS id, :2 AS text, :3 AS vector FROM DUAL) s
                    ON (e.id = s.id)
                    WHEN MATCHED THEN UPDATE SET e.text = s.text, e.vector = s.vector
                    WHEN NOT MATCHED THEN INSERT (id, text, vector)
                    VALUES (s.id, s.text, s.vector)""",
                    [id_, text_, vec_list],
                )
        self._conn.commit()

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> list[tuple[str, str, float]]:
        vec_list = self._to_vector_list(query_vector)

        with self._conn.cursor() as cur:
            cur.execute(
                """SELECT id, text,
                    1 - VECTOR_DISTANCE(vector, :1, COSINE) AS similarity
                FROM embeddings
                ORDER BY VECTOR_DISTANCE(vector, :1, COSINE)
                FETCH FIRST :2 ROWS ONLY""",
                [vec_list, top_k],
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
