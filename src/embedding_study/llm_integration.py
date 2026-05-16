"""LLM + Embedding + RAG 통합 시나리오 (DeepSeek API 사용).

4가지 시나리오를 비교한다:

시나리오 1. LLM only
    질문을 DeepSeek에 직접 전달한다.
    → 추가 컨텍스트 없음, 모델 내부 지식만 활용.

시나리오 2. LLM + RAG (키워드 검색 기반)
    질문에서 키워드를 추출 → 문서 저장소에서 키워드 매칭 →
    검색된 문서를 컨텍스트로 DeepSeek에 전달.
    → "환불"이라는 단어가 없으면 관련 문서를 놓친다.

시나리오 3. Embedding + LLM (의미 분석 기반)
    질문을 임베딩 벡터로 변환 → 사전 정의된 주제 벡터들과
    비교하여 질문의 의도/주제를 파악 → 분석 결과를 DeepSeek에 전달.
    → 문서 검색 없이 질문의 의미적 이해를 돕는다.

시나리오 4. Embedding + LLM + RAG (완전한 RAG)
    질문을 임베딩 벡터로 변환 → 벡터 유사도로 문서 검색 →
    검색된 문서를 컨텍스트로 DeepSeek에 전달.
    → "환불"이라는 단어가 없어도 의미가 통하는 문서를 찾는다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from openai import OpenAI

from src.embedding_study._interface import Embedder, VectorStore
from src.embedding_study.basics import SentenceEmbedder, cosine_similarity_manual
from src.embedding_study.search import InMemoryVectorStore
from src.utils import config


# ---------------------------------------------------------------------------
# DeepSeek LLM 클라이언트
# ---------------------------------------------------------------------------

class DeepSeekLLM:
    """DeepSeek API를 OpenAI-compatible 인터페이스로 호출한다.

    Usage:
        llm = DeepSeekLLM()
        answer = llm.ask("환불 기간이 어떻게 되나요?")
        answer = llm.ask_with_context("질문...", ["문서1...", "문서2..."])
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-chat",
    ) -> None:
        key = api_key or config.deepseek_api_key
        if not key:
            raise ValueError("DEEPSEEK_API_KEY 환경 변수가 설정되지 않았습니다.")

        self._client = OpenAI(
            api_key=key,
            base_url="https://api.deepseek.com",
        )
        self._model = model

    def ask(self, question: str) -> str:
        """LLM only — 질문을 그대로 DeepSeek에 전달한다."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": question}],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""

    def ask_with_context(self, question: str, documents: list[str]) -> str:
        """LLM + 문서 컨텍스트 — 검색된 문서를 포함하여 DeepSeek에 전달한다."""
        context_parts = [f"[문서 {i+1}]\n{doc}" for i, doc in enumerate(documents)]
        context = "\n\n".join(context_parts)

        system_prompt = (
            "당신은 주어진 문서만을 참고하여 질문에 답변하는 AI 어시스턴트입니다.\n"
            "문서에 없는 내용은 추측하지 말고, 모른다고 답변하세요."
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"참고 문서:\n{context}\n\n질문: {question}"},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# 시나리오 결과 데이터 클래스
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    """각 시나리오의 실행 결과."""

    scenario: str
    question: str
    answer: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 지식 베이스
# ---------------------------------------------------------------------------

# 회사 내부 정책 문서 (LLM이 학습하지 않은 가상의 정보)
_KNOWLEDGE_BASE = [
    {
        "id": "doc_001",
        "title": "환불 정책 v3.2 (2026년 3월 개정)",
        "content": (
            "2026년 3월 1일부로 환불 정책이 개정되었습니다. "
            "모든 제품은 구매 후 14일 이내에 환불이 가능합니다. "
            "단, 디지털 콘텐츠는 다운로드하지 않은 경우에 한해 환불됩니다. "
            "환불 절차는 마이페이지 > 주문내역 > 환불신청을 통해 진행합니다. "
            "환불 승인 후 영업일 기준 3~5일 이내에 결제 금액이 환급됩니다."
        ),
    },
    {
        "id": "doc_002",
        "title": "배송 정책 (2026년 1월 기준)",
        "content": (
            "국내 배송은 영업일 기준 2~3일 소요되며, 5만원 이상 구매 시 무료 배송입니다. "
            "5만원 미만 구매 시 배송비 3,000원이 부과됩니다. "
            "해외 배송은 미국, 일본, 중국, 독일에 한해 가능하며 배송비 15,000원이 추가됩니다. "
            "해외 배송 소요 기간은 국가별로 7~14일입니다."
        ),
    },
    {
        "id": "doc_003",
        "title": "VIP 멤버십 프로그램 (2026년 5월 신설)",
        "content": (
            "2026년 5월 1일부터 VIP 멤버십 프로그램이 신설되었습니다. "
            "연간 구매 금액 100만원 이상 고객은 자동으로 VIP 등급이 부여됩니다. "
            "VIP 혜택: 전 제품 10% 할인, 무료 배송, 전용 고객센터 운영(24시간), "
            "신제품 우선 구매 기회 제공. VIP 등급은 매년 1월 1일 재산정됩니다."
        ),
    },
    {
        "id": "doc_004",
        "title": "개인정보 처리방침 (2026년 4월 개정)",
        "content": (
            "개인정보 보유 기간은 회원 탈퇴 시 즉시 파기하는 것을 원칙으로 합니다. "
            "단, 관련 법령에 따라 보존이 필요한 정보는 아래 기간 동안 보관됩니다: "
            "계약 또는 청약 철회에 관한 기록: 5년, 대금 결제 및 재화 공급 기록: 5년, "
            "소비자 불만 또는 분쟁 처리 기록: 3년. "
            "개인정보 열람, 수정, 삭제는 마이페이지 > 개인정보 설정에서 가능합니다."
        ),
    },
    {
        "id": "doc_005",
        "title": "프로모션 및 쿠폰 정책",
        "content": (
            "쿠폰은 발급일로부터 30일 이내에 사용해야 하며, 연장이 불가능합니다. "
            "하나의 주문에는 최대 2개의 쿠폰을 중복 적용할 수 있습니다. "
            "최소 구매 금액 조건이 있는 쿠폰은 해당 조건 충족 시에만 적용됩니다. "
            "환불 시 사용된 쿠폰은 재발급되지 않습니다. "
            "신규 가입 고객에게는 첫 구매 20% 할인 쿠폰이 자동 지급됩니다."
        ),
    },
]


def get_knowledge_base_texts() -> list[str]:
    """지식 베이스의 문서 본문 리스트를 반환한다."""
    return [doc["content"] for doc in _KNOWLEDGE_BASE]


def get_knowledge_base_metadata() -> list[dict]:
    """지식 베이스 메타데이터 전체를 반환한다."""
    return _KNOWLEDGE_BASE


# ---------------------------------------------------------------------------
# 시나리오 1: LLM only
# ---------------------------------------------------------------------------

def scenario_1_llm_only(llm: DeepSeekLLM, question: str) -> ScenarioResult:
    """LLM only — DeepSeek에 직접 질문한다.

    추가 컨텍스트 없이 LLM의 내부 지식만으로 답변한다.
    → 2026년 5월 신설된 VIP 멤버십 같은 최신 정보는 알지 못한다.
    """
    answer = llm.ask(question)
    return ScenarioResult(
        scenario="1. LLM only",
        question=question,
        answer=answer,
        metadata={"method": "direct", "documents_used": []},
    )


# ---------------------------------------------------------------------------
# 시나리오 2: LLM + RAG (키워드 검색)
# ---------------------------------------------------------------------------

def _keyword_search(query: str, documents: list[str]) -> list[str]:
    """키워드 기반 검색 — 단어가 정확히 일치하는 문서만 찾는다.

    임베딩 없이 단순한 문자열 매칭으로 동작한다.
    """
    results = []
    for doc_text in documents:
        # 간단한 키워드 추출: 공백/특수문자로 분리
        import re
        keywords = set(re.findall(r"[\w가-힣]+", query.lower()))
        doc_lower = doc_text.lower()
        if any(kw in doc_lower for kw in keywords):
            results.append(doc_text)
    return results


def scenario_2_llm_rag_keyword(
    llm: DeepSeekLLM, question: str
) -> ScenarioResult:
    """LLM + RAG (키워드 검색 기반).

    질문에서 키워드를 추출 → 문서에서 해당 키워드가 있는 문서 검색 →
    검색된 문서를 컨텍스트로 DeepSeek에 전달.

    한계: "VIP 등급"이라고 물어보면 "멤버십"이라는 단어가 없어 문서를 놓친다.
    """
    all_docs = get_knowledge_base_texts()
    found = _keyword_search(question, all_docs)

    if found:
        answer = llm.ask_with_context(question, found)
    else:
        answer = llm.ask(question)  # 검색 결과 없으면 LLM only와 동일

    return ScenarioResult(
        scenario="2. LLM + RAG (키워드 검색)",
        question=question,
        answer=answer,
        metadata={
            "method": "keyword_search",
            "documents_found": len(found),
            "documents_used": [d[:50] + "..." for d in found],
        },
    )


# ---------------------------------------------------------------------------
# 시나리오 3: Embedding + LLM (의미 분석)
# ---------------------------------------------------------------------------

# 사전 정의된 주제 카테고리와 그 대표 임베딩
_TOPIC_CATEGORIES = {
    "refund": "Refund, return, money back, reimbursement, cancellation policy",
    "shipping": "Shipping, delivery, logistics, tracking, international mail",
    "membership": "Membership, VIP, loyalty program, tier, benefits, rewards",
    "privacy": "Privacy, personal data, GDPR, data protection, consent",
    "promotion": "Coupon, discount, promotion, sale, voucher, deal",
}


def _build_topic_vectors(embedder: Embedder) -> dict[str, np.ndarray]:
    """주제 카테고리 텍스트를 임베딩 벡터로 변환한다."""
    return {
        topic: embedder.encode([desc])[0]
        for topic, desc in _TOPIC_CATEGORIES.items()
    }


def _classify_intent(
    embedder: Embedder, question: str
) -> list[tuple[str, float]]:
    """질문의 의도를 임베딩 유사도로 분류한다.

    Returns:
        [(주제, 유사도), ...] 내림차순.
    """
    q_vec = embedder.encode([question])[0]
    topic_vecs = _build_topic_vectors(embedder)

    scores = [
        (topic, cosine_similarity_manual(q_vec, t_vec))
        for topic, t_vec in topic_vecs.items()
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def scenario_3_embedding_llm(
    embedder: Embedder, llm: DeepSeekLLM, question: str
) -> ScenarioResult:
    """Embedding + LLM — 임베딩으로 질문 의도를 분석하여 LLM에 전달한다.

    문서 검색은 하지 않는다. 대신:
    1. 질문을 임베딩 벡터로 변환
    2. 사전 정의된 주제 벡터와 비교하여 질문의 의도/카테고리 파악
    3. 분류 결과를 포함한 프롬프트를 DeepSeek에 전달

    이 방식은 질문의 의미적 맥락을 LLM이 더 잘 이해하도록 돕는다.
    """
    # 질문 의도 분류
    intent_scores = _classify_intent(embedder, question)
    top_intent = intent_scores[0]
    all_intents = ", ".join(
        f"{topic}({score:.2f})" for topic, score in intent_scores[:3]
    )

    # 분석 결과를 포함한 프롬프트 구성
    enhanced_prompt = (
        f"질문: {question}\n\n"
        f"[시스템 분석]\n"
        f"- 가장 관련 높은 주제: {top_intent[0]} (유사도: {top_intent[1]:.2f})\n"
        f"- 전체 주제 분포: {all_intents}\n"
        f"- 추천 접근 방식: {top_intent[0]} 주제에 초점을 맞춰 답변하세요.\n\n"
        f"위 분석을 참고하여 질문에 답변해주세요."
    )

    answer = llm.ask(enhanced_prompt)
    return ScenarioResult(
        scenario="3. Embedding + LLM (의미 분석)",
        question=question,
        answer=answer,
        metadata={
            "method": "embedding_intent_analysis",
            "top_intent": top_intent,  # (topic, score) 튜플 전체
            "all_intents": intent_scores,
            "documents_used": [],
        },
    )


# ---------------------------------------------------------------------------
# 시나리오 4: Embedding + LLM + RAG (완전한 RAG)
# ---------------------------------------------------------------------------

def scenario_4_full_rag(
    embedder: Embedder,
    store: VectorStore,
    llm: DeepSeekLLM,
    question: str,
    top_k: int = 3,
) -> ScenarioResult:
    """Embedding + LLM + RAG — 완전한 RAG 파이프라인.

    1. 질문을 임베딩 벡터로 변환
    2. 벡터 유사도로 문서 저장소에서 관련 문서 검색
    3. 검색된 문서를 컨텍스트로 DeepSeek에 전달하여 답변 생성

    이 방식이 가장 정확하다. 최신 정보를 반영하고,
    키워드가 정확히 일치하지 않아도 의미적 유사도로 문서를 찾는다.
    """
    q_vec = embedder.encode([question])[0]
    hits = store.search(q_vec, top_k=top_k)

    if hits:
        documents = [text for _, text, _ in hits]
        answer = llm.ask_with_context(question, documents)
    else:
        answer = llm.ask(question)

    return ScenarioResult(
        scenario="4. Embedding + LLM + RAG (완전한 RAG)",
        question=question,
        answer=answer,
        metadata={
            "method": "embedding_semantic_search",
            "documents_found": len(hits),
            "documents_used": [
                {"text": text[:100] + "...", "score": round(score, 4)}
                for _, text, score in hits
            ],
        },
    )


# ---------------------------------------------------------------------------
# 시나리오 비교 실행기
# ---------------------------------------------------------------------------


def build_knowledge_store(embedder: Embedder) -> InMemoryVectorStore:
    """지식 베이스를 임베딩하여 벡터 저장소를 구축한다."""
    knowledge = get_knowledge_base_metadata()
    ids = [doc["id"] for doc in knowledge]
    texts = [doc["content"] for doc in knowledge]
    vectors = embedder.encode(texts)

    store = InMemoryVectorStore()
    store.add(ids, texts, vectors)
    return store


def run_all_scenarios(
    questions: list[str] | None = None,
) -> dict[str, list[ScenarioResult]]:
    """4가지 시나리오를 모두 실행하고 결과를 비교한다.

    Args:
        questions: 테스트할 질문 리스트. None이면 기본 질문 사용.

    Returns:
        {question: [ScenarioResult, ...]} 형태.
    """
    if questions is None:
        questions = [
            "VIP 멤버십 혜택이 무엇인가요?",   # 2026년 신설 — LLM만 아는 척 불가
            "돈을 돌려받고 싶어요",           # "환불" 키워드 없음
            "개인정보는 얼마나 보관되나요?",   # 구체적 정책 정보
        ]

    llm = DeepSeekLLM()
    embedder = SentenceEmbedder()
    store = build_knowledge_store(embedder)

    all_results: dict[str, list[ScenarioResult]] = {}

    for question in questions:
        results = [
            scenario_1_llm_only(llm, question),
            scenario_2_llm_rag_keyword(llm, question),
            scenario_3_embedding_llm(embedder, llm, question),
            scenario_4_full_rag(embedder, store, llm, question),
        ]
        all_results[question] = results

    return all_results


def print_comparison(results: dict[str, list[ScenarioResult]]) -> None:
    """시나리오 비교 결과를 콘솔에 출력한다."""
    for question, scenarios in results.items():
        print(f"\n{'='*70}")
        print(f"질문: {question}")
        print(f"{'='*70}")

        for r in scenarios:
            print(f"\n─── {r.scenario} ───")
            print(f"방법: {r.metadata.get('method', 'N/A')}")
            docs = r.metadata.get("documents_used", [])
            print(f"참고 문서 수: {len(docs)}")
            if docs:
                for d in docs:
                    if isinstance(d, dict):
                        print(f"  - [{d['score']}] {d['text']}")
                    else:
                        print(f"  - {d}")
            print(f"답변: {r.answer[:200]}{'...' if len(r.answer) > 200 else ''}")
