"""Tests for LLM + Embedding + RAG integration scenarios.

DeepSeek API 호출은 모두 mock으로 대체하여 실제 API 비용이 발생하지 않도록 한다.
"""

from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest

from src.embedding_study._interface import Embedder, VectorStore
from src.embedding_study.llm_integration import (
    DeepSeekLLM,
    ScenarioResult,
    _keyword_search,
    _classify_intent,
    build_knowledge_store,
    scenario_1_llm_only,
    scenario_2_llm_rag_keyword,
    scenario_3_embedding_llm,
    scenario_4_full_rag,
    get_knowledge_base_texts,
    get_knowledge_base_metadata,
    _TOPIC_CATEGORIES,
)


# ---------------------------------------------------------------------------
# 공통 Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_openai_client() -> Mock:
    """OpenAI 클라이언트를 mock으로 대체."""
    mock_client = Mock()

    # ask() 응답
    mock_choice = Mock()
    mock_choice.message.content = "LLM only 응답입니다."

    mock_completion = Mock()
    mock_completion.choices = [mock_choice]

    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client


@pytest.fixture
def mock_llm(mock_openai_client) -> DeepSeekLLM:
    """Mock OpenAI 클라이언트를 사용하는 DeepSeekLLM."""
    with patch.object(DeepSeekLLM, "__init__", lambda self, **kw: None):
        llm = DeepSeekLLM.__new__(DeepSeekLLM)
        # DeepSeekLLM 초기화 우회
        llm._client = mock_openai_client
        llm._model = "deepseek-chat"
        return llm


@pytest.fixture
def mock_embedder() -> Mock:
    """의미 기반 벡터를 반환하는 mock Embedder."""
    embedder = Mock(spec=Embedder)
    embedder.dim = 384

    def encode_side_effect(texts: list[str]) -> np.ndarray:
        """의미 기반 벡터를 근사하기 위한 재현 가능한 mock.

        substring "ship"이 "membership"에 포함되는 충돌을 방지하기 위해
        각 키워드는 독립적으로 점수를 누적하고, 최고 점수 방향을 선택한다.
        """
        import re

        vectors = []
        for t in texts:
            vec = np.zeros(384, dtype=np.float32)
            t_lower = t.lower()

            # 각 차원에 대한 점수를 독립적으로 계산한다 (elif 대신 if)
            score_0 = any(
                re.search(rf"\b{w}\b", t_lower)
                for w in ["refund", "return", "money back", "돌려"]
            ) or any(w in t_lower for w in ["환불", "반품"])

            score_1 = any(
                re.search(rf"\b{w}\b", t_lower)
                for w in ["ship", "shipping", "delivery", "택배"]
            ) or any(w in t_lower for w in ["배송"])

            score_2 = any(
                re.search(rf"\b{w}\b", t_lower)
                for w in ["member", "membership", "vip", "loyalty", "등급"]
            ) or any(w in t_lower for w in ["멤버십"])

            score_3 = any(
                re.search(rf"\b{w}\b", t_lower)
                for w in ["privacy", "개인정보"]
            ) or any(w in t_lower for w in ["데이터 보호"])

            score_4 = any(
                re.search(rf"\b{w}\b", t_lower)
                for w in ["coupon", "discount", "promotion", "할인"]
            ) or any(w in t_lower for w in ["쿠폰"])

            # 가장 높은 점수의 차원에 1.0 할당
            if score_0:
                vec[0] = 1.0
            elif score_1:
                vec[1] = 1.0
            elif score_2:
                vec[2] = 1.0
            elif score_3:
                vec[3] = 1.0
            elif score_4:
                vec[4] = 1.0
            else:
                vec[0] = 0.5

            vectors.append(vec)
        return np.array(vectors, dtype=np.float32)

    embedder.encode.side_effect = encode_side_effect
    return embedder


@pytest.fixture
def mock_store(mock_embedder) -> Mock:
    """Mock VectorStore."""
    store = Mock(spec=VectorStore)
    store.search.return_value = [
        ("doc_001", "환불은 구매 후 14일 이내에 가능합니다.", 0.92),
        ("doc_005", "쿠폰은 발급일로부터 30일 이내에 사용해야 합니다.", 0.45),
    ]
    return store


# ---------------------------------------------------------------------------
# DeepSeekLLM
# ---------------------------------------------------------------------------

class TestDeepSeekLLM:
    def test_ask_returns_string(self, mock_llm):
        # Given
        mock_llm._client.chat.completions.create.return_value.choices[0].message.content = (
            "안녕하세요! 무엇을 도와드릴까요?"
        )

        # When
        answer = mock_llm.ask("안녕하세요!")

        # Then
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_ask_passes_correct_model(self, mock_llm):
        # When
        mock_llm.ask("질문")

        # Then
        call_kwargs = mock_llm._client.chat.completions.create.call_args
        assert call_kwargs[1]["model"] == "deepseek-chat"
        assert call_kwargs[1]["messages"][0]["role"] == "user"
        assert call_kwargs[1]["messages"][0]["content"] == "질문"

    def test_ask_with_context_includes_documents(self, mock_llm):
        # When
        mock_llm.ask_with_context("질문입니다", ["문서A 내용", "문서B 내용"])

        # Then
        call_kwargs = mock_llm._client.chat.completions.create.call_args
        messages = call_kwargs[1]["messages"]
        assert messages[0]["role"] == "system"  # system prompt
        user_content = messages[1]["content"]
        assert "문서A 내용" in user_content
        assert "문서B 내용" in user_content
        assert "질문입니다" in user_content

    def test_constructor_raises_without_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
                DeepSeekLLM()


# ---------------------------------------------------------------------------
# 키워드 검색 (시나리오 2)
# ---------------------------------------------------------------------------

class TestKeywordSearch:
    def test_finds_exact_keyword_match(self):
        docs = [
            "환불은 구매 후 14일 이내에 가능합니다.",
            "배송은 3~5일 소요됩니다.",
        ]
        results = _keyword_search("환불 기간이 어떻게 되나요?", docs)
        assert len(results) == 1
        assert "환불" in results[0]

    def test_misses_semantic_match(self):
        """키워드 검색은 "돈을 돌려받고" 같은 표현을 놓친다."""
        docs = [
            "환불은 구매 후 14일 이내에 가능합니다.",
            "배송은 3~5일 소요됩니다.",
        ]
        results = _keyword_search("돈을 돌려받고 싶어요", docs)
        # "환불"이라는 단어가 없으므로 검색 결과 0건
        assert len(results) == 0

    def test_returns_multiple_matches(self):
        docs = [
            "VIP 혜택: 10% 할인",
            "쿠폰은 30일 이내 사용",
            "배송 정책 안내",
        ]
        results = _keyword_search("할인 쿠폰 사용 방법", docs)
        assert len(results) == 2  # "할인" 포함 + "쿠폰" 포함


# ---------------------------------------------------------------------------
# 의도 분류 (시나리오 3)
# ---------------------------------------------------------------------------

class TestIntentClassification:
    def test_classifies_refund_intent(self, mock_embedder):
        intents = _classify_intent(mock_embedder, "돈을 돌려받고 싶어요")
        top_topic, top_score = intents[0]
        assert top_topic == "refund"
        assert top_score > 0.5

    def test_classifies_shipping_intent(self, mock_embedder):
        intents = _classify_intent(mock_embedder, "배송이 너무 느려요 언제 도착하나요?")
        top_topic, _ = intents[0]
        assert top_topic == "shipping"

    def test_classifies_membership_intent(self, mock_embedder):
        intents = _classify_intent(mock_embedder, "VIP 등급이 되려면 어떻게 해야 하나요?")
        top_topic, top_score = intents[0]
        assert top_topic == "membership"
        assert top_score > 0.5

    def test_topic_categories_have_five_entries(self):
        assert len(_TOPIC_CATEGORIES) == 5
        expected = {"refund", "shipping", "membership", "privacy", "promotion"}
        assert set(_TOPIC_CATEGORIES.keys()) == expected


# ---------------------------------------------------------------------------
# 시나리오 1: LLM only
# ---------------------------------------------------------------------------

class TestScenario1LLMOnly:
    def test_returns_scenario_result(self, mock_llm):
        result = scenario_1_llm_only(mock_llm, "환불 기간은?")

        assert isinstance(result, ScenarioResult)
        assert result.scenario == "1. LLM only"
        assert result.answer == "LLM only 응답입니다."

    def test_no_documents_used(self, mock_llm):
        result = scenario_1_llm_only(mock_llm, "질문")
        assert result.metadata["documents_used"] == []
        assert result.metadata["method"] == "direct"


# ---------------------------------------------------------------------------
# 시나리오 2: LLM + RAG (키워드 검색)
# ---------------------------------------------------------------------------

class TestScenario2LLMRagKeyword:
    def test_finds_documents_by_keyword(self, mock_llm):
        # Mock ask_with_context to verify it receives documents
        mock_llm.ask_with_context = Mock(return_value="키워드 RAG 응답")

        result = scenario_2_llm_rag_keyword(mock_llm, "환불 정책이 궁금합니다")

        assert result.scenario == "2. LLM + RAG (키워드 검색)"
        assert result.metadata["documents_found"] >= 1
        assert result.answer == "키워드 RAG 응답"

    def test_keyword_search_misses_semantic_queries(self, mock_llm):
        """"돈을 돌려받고 싶어요"는 "환불" 키워드가 없어 문서 검색 실패."""
        mock_llm.ask = Mock(return_value="LLM only fallback")

        result = scenario_2_llm_rag_keyword(mock_llm, "돈을 돌려받고 싶어요")

        assert result.metadata["documents_found"] == 0
        mock_llm.ask.assert_called_once()


# ---------------------------------------------------------------------------
# 시나리오 3: Embedding + LLM (의미 분석)
# ---------------------------------------------------------------------------

class TestScenario3EmbeddingLLM:
    def test_classifies_intent_and_calls_llm(self, mock_embedder, mock_llm):
        mock_llm.ask = Mock(return_value="의미 분석 기반 응답")

        result = scenario_3_embedding_llm(
            mock_embedder, mock_llm, "VIP 멤버십 혜택이 뭔가요?"
        )

        assert "3. Embedding + LLM" in result.scenario
        assert result.metadata["method"] == "embedding_intent_analysis"
        assert result.metadata["top_intent"][0] == "membership"
        assert result.answer == "의미 분석 기반 응답"

    def test_enhanced_prompt_includes_intent_analysis(self, mock_embedder, mock_llm):
        mock_llm.ask = Mock(return_value="응답")

        scenario_3_embedding_llm(mock_embedder, mock_llm, "배송은 얼마나 걸리나요?")

        call_args = mock_llm.ask.call_args[0][0]
        assert "shipping" in call_args
        assert "주제" in call_args
        assert "배송은 얼마나 걸리나요?" in call_args

    def test_prefers_semantic_over_keywords(self, mock_embedder, mock_llm):
        """의미 분석은 "돈을 돌려받고" → "refund"로 정확히 분류한다."""
        mock_llm.ask = Mock(return_value="환불 관련 응답")

        result = scenario_3_embedding_llm(
            mock_embedder, mock_llm, "돈을 돌려받고 싶어요"
        )

        # "환불"이라는 단어가 없어도 refund로 분류
        assert result.metadata["top_intent"][0] == "refund"


# ---------------------------------------------------------------------------
# 시나리오 4: Embedding + LLM + RAG (완전한 RAG)
# ---------------------------------------------------------------------------

class TestScenario4FullRAG:
    def test_searches_and_uses_documents(self, mock_embedder, mock_store, mock_llm):
        mock_llm.ask_with_context = Mock(return_value="Full RAG 응답")

        result = scenario_4_full_rag(
            mock_embedder, mock_store, mock_llm, "환불 기간은?"
        )

        assert "4. Embedding + LLM + RAG" in result.scenario
        mock_store.search.assert_called_once()
        assert result.metadata["documents_found"] == 2
        assert result.answer == "Full RAG 응답"

    def test_uses_top_k_parameter(self, mock_embedder, mock_store, mock_llm):
        mock_llm.ask_with_context = Mock(return_value="응답")

        scenario_4_full_rag(
            mock_embedder, mock_store, mock_llm, "질문", top_k=1
        )

        mock_store.search.assert_called_once()
        assert mock_store.search.call_args[1]["top_k"] == 1

    def test_fallback_when_no_documents(self, mock_embedder, mock_store, mock_llm):
        mock_store.search.return_value = []
        mock_llm.ask = Mock(return_value="fallback 응답")

        result = scenario_4_full_rag(
            mock_embedder, mock_store, mock_llm, "질문"
        )

        assert result.answer == "fallback 응답"
        assert result.metadata["documents_found"] == 0


# ---------------------------------------------------------------------------
# 지식 베이스
# ---------------------------------------------------------------------------

class TestKnowledgeBase:
    def test_has_five_documents(self):
        texts = get_knowledge_base_texts()
        assert len(texts) == 5

    def test_all_documents_have_ids(self):
        docs = get_knowledge_base_metadata()
        for doc in docs:
            assert "id" in doc
            assert "title" in doc
            assert "content" in doc

    def test_covers_all_topics(self):
        """모든 주제(refund, shipping, membership, privacy, promotion)가
        지식 베이스에 포함되어 있다."""
        all_text = " ".join(get_knowledge_base_texts())
        assert "환불" in all_text
        assert "배송" in all_text
        assert "VIP" in all_text
        assert "개인정보" in all_text
        assert "쿠폰" in all_text

    def test_build_knowledge_store_returns_store(self, mock_embedder):
        store = build_knowledge_store(mock_embedder)
        assert store is not None


# ---------------------------------------------------------------------------
# 시나리오 간 비교 검증
# ---------------------------------------------------------------------------

class TestScenarioComparison:
    """4가지 시나리오의 상대적 특성을 검증한다."""

    def test_scenario_1_vs_2_document_usage(self, mock_llm):
        """시나리오 1은 문서를 사용하지 않고, 2는 키워드 매칭을 시도한다."""
        result_1 = scenario_1_llm_only(mock_llm, "환불 정책")

        assert result_1.metadata["documents_used"] == []

    def test_scenario_2_vs_3_keyword_vs_semantic(
        self, mock_embedder, mock_llm
    ):
        """키워드 검색(시나리오2)은 "돈을 돌려받고"를 놓치지만,
        의미 분석(시나리오3)은 정확히 분류한다."""
        mock_llm.ask = Mock(return_value="응답")
        mock_llm.ask_with_context = Mock(return_value="응답")

        question = "돈을 돌려받고 싶어요"

        r2 = scenario_2_llm_rag_keyword(mock_llm, question)
        r3 = scenario_3_embedding_llm(mock_embedder, mock_llm, question)

        # 시나리오 2: 키워드 "환불"이 없어서 문서 검색 실패
        assert r2.metadata["documents_found"] == 0

        # 시나리오 3: 의미 분석으로 "refund"로 올바르게 분류
        assert r3.metadata["top_intent"][0] == "refund"

    def test_all_four_return_scenario_results(
        self, mock_embedder, mock_store, mock_llm
    ):
        """4가지 시나리오 모두 ScenarioResult를 반환한다."""
        mock_llm.ask = Mock(return_value="응답")
        mock_llm.ask_with_context = Mock(return_value="응답")

        results = [
            scenario_1_llm_only(mock_llm, "질문"),
            scenario_2_llm_rag_keyword(mock_llm, "질문"),
            scenario_3_embedding_llm(mock_embedder, mock_llm, "질문"),
            scenario_4_full_rag(mock_embedder, mock_store, mock_llm, "질문"),
        ]

        for r in results:
            assert isinstance(r, ScenarioResult)
            assert len(r.scenario) > 0
            assert len(r.question) > 0
            assert len(r.answer) > 0
            assert isinstance(r.metadata, dict)
            assert "method" in r.metadata


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_question(self, mock_llm):
        mock_llm.ask = Mock(return_value="질문을 이해하지 못했습니다.")
        result = scenario_1_llm_only(mock_llm, "")
        assert isinstance(result, ScenarioResult)

    def test_no_matching_keywords(self):
        docs = ["문서1", "문서2"]
        results = _keyword_search("abcdefg", docs)
        assert results == []

    def test_all_documents_matching(self):
        docs = ["환불 안내", "환불 정책", "환불 절차"]
        results = _keyword_search("환불", docs)
        assert len(results) == 3

    def test_knowledge_base_is_immutable_in_scope(self):
        """get_knowledge_base_texts()는 호출할 때마다 동일한 결과를 반환."""
        first = get_knowledge_base_texts()
        second = get_knowledge_base_texts()
        assert first == second
        assert len(first) == 5
