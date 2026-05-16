"""의미 기반 검색 (Semantic Search).

전통적인 키워드 검색과 달리, 임베딩 기반 검색은 단어가 정확히
일치하지 않아도 의미가 비슷한 문서를 찾을 수 있다.

예: "기쁘다"로 검색 → "행복합니다", "즐거운" 같은 문서도 매칭.
"""

from __future__ import annotations

import numpy as np

from src.embedding_study._interface import Embedder, VectorStore


class InMemoryVectorStore(VectorStore):
    """메모리 기반 벡터 저장소.

    소규모 데모용. 실무에서는 FAISS, Chroma, Pinecone 등을 사용한다.
    """

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._vectors: np.ndarray | None = None

    def add(self, ids: list[str], texts: list[str], vectors: np.ndarray) -> None:
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2-dimensional")
        if len(ids) != len(texts) or len(ids) != len(vectors):
            raise ValueError("ids, texts, vectors must have the same length")

        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])
        self._ids.extend(ids)
        self._texts.extend(texts)

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> list[tuple[str, str, float]]:
        if self._vectors is None or len(self._vectors) == 0:
            return []

        # 쿼리 벡터 정규화
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-10)

        # 모든 저장된 벡터와 한 번에 코사인 유사도 계산
        stored_norm = self._vectors / (
            np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-10
        )
        scores = np.dot(stored_norm, query_norm)

        # 상위 top_k 인덱스 추출
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            (self._ids[i], self._texts[i], float(scores[i]))
            for i in top_indices
        ]

    def __len__(self) -> int:
        return len(self._ids)


def build_knowledge_base(embedder: Embedder) -> InMemoryVectorStore:
    """한국어 FAQ 지식 베이스를 구축한다.

    실무에서는 PDF, 웹페이지, DB에서 문서를 읽어와 벡터화한다.
    """
    documents = [
        ("faq_01", "환불은 구매 후 7일 이내에 가능합니다."),
        ("faq_02", "반품 신청은 마이페이지에서 하실 수 있습니다."),
        ("faq_03", "배송은 영업일 기준 3~5일 소요됩니다."),
        ("faq_04", "회원가입은 이메일 또는 SNS 계정으로 가능합니다."),
        ("faq_05", "비밀번호를 잊으셨다면 로그인 화면에서 찾기 버튼을 눌러주세요."),
        ("faq_06", "할인 쿠폰은 결제 단계에서 자동으로 적용됩니다."),
        ("faq_07", "고객센터 운영시간은 평일 오전 9시부터 오후 6시까지입니다."),
        ("faq_08", "해외 배송은 일부 국가에 한해 가능하며 추가 요금이 발생합니다."),
        ("faq_09", "주문 취소는 상품 준비 중 상태에서만 가능합니다."),
        ("faq_10", "적립금은 구매 금액의 1%가 자동으로 적립됩니다."),
    ]

    ids, texts = zip(*documents)
    vectors = embedder.encode(list(texts))

    store = InMemoryVectorStore()
    store.add(list(ids), list(texts), vectors)
    return store


def demonstrate_semantic_search() -> dict:
    """의미 기반 검색을 시연한다.

    키워드가 정확히 일치하지 않아도 의미가 통하는 결과를 찾는 것을 보여준다.
    """
    from src.embedding_study.basics import SentenceEmbedder

    embedder = SentenceEmbedder()
    store = build_knowledge_base(embedder)

    queries = [
        "돈을 돌려받고 싶어요",       # "환불"과 의미적 유사
        "아이디를 새로 만들고 싶어요",  # "회원가입"과 유사
        "택배가 언제 도착하나요",      # "배송"과 유사
        "비번 까먹었어요",            # "비밀번호 찾기"와 유사
    ]

    results = {}
    for query in queries:
        q_vec = embedder.encode([query])[0]
        hits = store.search(q_vec, top_k=2)
        results[query] = [
            {"doc_id": h[0], "text": h[1], "score": round(h[2], 4)}
            for h in hits
        ]

    return results


def keyword_vs_semantic_comparison() -> dict:
    """키워드 검색과 의미 검색의 차이를 보여주는 비교 실험."""
    from src.embedding_study.basics import SentenceEmbedder

    embedder = SentenceEmbedder()
    store = build_knowledge_base(embedder)

    # 키워드 검색으로는 찾을 수 없는 질문들
    test_cases = [
        ("상품을 되돌리고 싶어요", "환불"),      # "환불"이라는 단어 없음
        ("어떻게 가입하나요", "회원가입"),        # "회원가입"이라는 단어 없음
        ("물건이 언제 오나요", "배송"),           # "배송"이라는 단어 없음
    ]

    results = {}
    for query, expected_keyword in test_cases:
        q_vec = embedder.encode([query])[0]
        top = store.search(q_vec, top_k=1)[0]

        # 키워드 검색 시뮬레이션: 단어가 정확히 포함되어야 매칭
        keyword_match = any(
            expected_keyword in doc_text
            for _, doc_text, _ in store.search(q_vec, top_k=len(store))
        )
        # 실제로는 키워드 검색이면 결과가 0개일 것이다.

        results[query] = {
            "expected_keyword": expected_keyword,
            "keyword_found_in_query": expected_keyword in query,
            "semantic_top_result": top[1],
            "semantic_score": round(top[2], 4),
            "keyword_search_would_fail": not keyword_match
            and expected_keyword not in query,
        }

    return results
