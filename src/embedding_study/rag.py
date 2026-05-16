"""Retrieval-Augmented Generation (RAG) 기본 구현.

RAG는 LLM이 답변을 생성하기 전에 외부 지식 베이스에서
관련 문서를 검색(retrieve)하여 프롬프트에 포함시키는 기법이다.

동작 흐름:
    1. 질문 → 임베딩 변환
    2. 벡터 저장소에서 유사 문서 검색
    3. 검색된 문서를 컨텍스트로 프롬프트에 추가
    4. LLM이 컨텍스트를 바탕으로 답변 생성

이 모듈은 Claude API 없이도 RAG의 검색 파이프라인을 시연하며,
LLM 호출 부분은 선택적으로 포함할 수 있다.
"""

from __future__ import annotations

from src.embedding_study._interface import Embedder, VectorStore


class RAGPipeline:
    """RAG 파이프라인 — 검색 + 생성을 조합한다.

    Usage:
        pipe = RAGPipeline(embedder, store)
        answer = pipe.ask("환불은 어떻게 하나요?")
        # → 검색된 문서를 바탕으로 답변 생성
    """

    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(
        self, query: str, top_k: int = 3
    ) -> list[tuple[str, str, float]]:
        """질문과 관련된 문서를 검색한다."""
        q_vec = self._embedder.encode([query])[0]
        return self._store.search(q_vec, top_k=top_k)

    def build_prompt(
        self, query: str, documents: list[tuple[str, str, float]]
    ) -> str:
        """검색된 문서를 포함한 프롬프트를 구성한다.

        실제 LLM 호출은 이 프롬프트를 사용한다.
        """
        context_parts = []
        for i, (doc_id, text, score) in enumerate(documents, 1):
            context_parts.append(f"[문서 {i}] (관련도: {score:.2f})\n{text}")

        context = "\n\n".join(context_parts)

        return (
            "당신은 주어진 문서를 참고하여 질문에 답변하는 AI 어시스턴트입니다.\n"
            "문서에 없는 내용은 추측하지 말고, 모른다고 답변하세요.\n"
            "\n"
            f"=== 참고 문서 ===\n"
            f"{context}\n"
            f"=== 문서 끝 ===\n"
            "\n"
            f"질문: {query}\n"
            f"답변:"
        )

    def ask(self, query: str, top_k: int = 3) -> dict:
        """질문에 대한 RAG 응답을 반환한다.

        Returns:
            {
                "query": 원본 질문,
                "documents": 검색된 문서 리스트,
                "prompt": LLM에 전달할 프롬프트,
                "answer": (LLM이 있다면) 생성된 답변, 없으면 프롬프트만 반환,
            }
        """
        docs = self.retrieve(query, top_k=top_k)
        prompt = self.build_prompt(query, docs)

        # LLM 없이도 프롬프트를 반환하므로 RAG 파이프라인 구조를 이해할 수 있다.
        return {
            "query": query,
            "documents": [
                {"id": d[0], "text": d[1], "score": round(d[2], 4)}
                for d in docs
            ],
            "prompt": prompt,
            "answer": None,  # 실제 LLM 호출 시 여기에 답변 채움
        }


def demonstrate_rag_pipeline() -> dict:
    """RAG 파이프라인 동작을 시연한다."""
    from src.embedding_study.basics import SentenceEmbedder
    from src.embedding_study.search import build_knowledge_base

    embedder = SentenceEmbedder()
    store = build_knowledge_base(embedder)
    pipe = RAGPipeline(embedder, store)

    queries = [
        "환불 기간이 어떻게 되나요?",
        "해외로 상품을 받을 수 있나요?",
        "비밀번호를 변경하고 싶어요",
    ]

    results = {}
    for query in queries:
        results[query] = pipe.ask(query, top_k=2)

    return results


def explain_rag_benefits() -> list[str]:
    """RAG의 장점을 설명하는 요약 리스트."""
    return [
        "할루시네이션 감소: LLM이 모르는 내용을 지어내지 않고, 주어진 문서만 참고한다.",
        "지식 업데이트 용이: LLM을 재학습할 필요 없이 벡터 저장소의 문서만 교체하면 된다.",
        "출처 추적 가능: 답변의 근거가 된 문서를 사용자에게 제시할 수 있다.",
        "도메인 특화: 특정 회사/분야의 내부 문서를 바탕으로 답변할 수 있다.",
        "비용 효율: 전체 문서를 매번 프롬프트에 넣지 않고, 관련 부분만 선별해 토큰을 절약한다.",
    ]
