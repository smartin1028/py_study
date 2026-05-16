"""Tests for embedding_study package."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.embedding_study._interface import Embedder, VectorStore
from src.embedding_study.basics import (
    SentenceEmbedder,
    cosine_similarity_manual,
    demonstrate_embedding_basics,
)
from src.embedding_study.ollama_embedder import OllamaEmbedder
from src.embedding_study.search import (
    InMemoryVectorStore,
    build_knowledge_base,
    keyword_vs_semantic_comparison,
)
from src.embedding_study.rag import RAGPipeline, explain_rag_benefits


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_embedder() -> Mock:
    """2차원 공간에 배치하는 가상의 Embedder.

    의미가 비슷한 텍스트는 가까운 벡터를, 다른 텍스트는 먼 벡터를 반환하도록
    side_effect로 구현한다.
    """
    embedder = Mock(spec=Embedder)
    embedder.dim = 3

    def encode_side_effect(texts: list[str]) -> np.ndarray:
        vectors = []
        for t in texts:
            if "환불" in t or "돌려" in t or "반품" in t:
                vectors.append([1.0, 0.0, 0.0])  # 환불 클러스터
            elif "배송" in t or "택배" in t:
                vectors.append([0.0, 1.0, 0.0])  # 배송 클러스터
            elif "가입" in t or "회원" in t or "비밀" in t or "계정" in t:
                vectors.append([0.0, 0.0, 1.0])  # 계정 클러스터
            else:
                vectors.append([0.5, 0.5, 0.0])
        return np.array(vectors, dtype=np.float32)

    embedder.encode.side_effect = encode_side_effect
    embedder.similarity.side_effect = lambda a, b: cosine_similarity_manual(
        np.asarray(a), np.asarray(b)
    )
    return embedder


@pytest.fixture
def empty_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@pytest.fixture
def populated_store(mock_embedder: Mock) -> InMemoryVectorStore:
    """환불/배송/계정 문서 6개가 저장된 스토어."""
    docs = [
        ("d1", "환불은 구매 후 7일 이내에 가능합니다"),
        ("d2", "반품 신청은 마이페이지에서 하실 수 있습니다"),
        ("d3", "배송은 영업일 기준 3~5일 소요됩니다"),
        ("d4", "해외 배송은 추가 요금이 발생합니다"),
        ("d5", "회원가입은 이메일로 가능합니다"),
        ("d6", "비밀번호를 잊으셨다면 찾기 버튼을 눌러주세요"),
    ]
    ids, texts = zip(*docs)
    vectors = mock_embedder.encode(list(texts))
    store = InMemoryVectorStore()
    store.add(list(ids), list(texts), vectors)
    return store


# ---------------------------------------------------------------------------
# cosine_similarity_manual
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity_manual(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity_manual(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert cosine_similarity_manual(a, b) == pytest.approx(-1.0)

    def test_similar_direction(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.866, 0.5])  # 30도 회전, cos(30°) ≈ 0.866
        assert cosine_similarity_manual(a, b) == pytest.approx(0.866, abs=0.01)


# ---------------------------------------------------------------------------
# InMemoryVectorStore
# ---------------------------------------------------------------------------

class TestInMemoryVectorStore:
    def test_empty_store_search_returns_empty(self, empty_store):
        q = np.array([1.0, 0.0, 0.0])
        assert empty_store.search(q) == []

    def test_add_and_search_returns_correct_order(
        self, empty_store, mock_embedder
    ):
        # Given
        vectors = mock_embedder.encode([
            "환불 관련 문의",  # [1,0,0]
            "배송 관련 문의",  # [0,1,0]
        ])
        empty_store.add(
            ["a", "b"],
            ["환불 관련 문의", "배송 관련 문의"],
            vectors,
        )

        # When — "환불"과 가까운 쿼리
        query = mock_embedder.encode(["돈을 돌려받고 싶어요"])[0]  # [1,0,0]

        # Then
        results = empty_store.search(query, top_k=2)
        assert len(results) == 2
        assert results[0][0] == "a"  # 첫 번째는 환불 문서
        assert results[0][2] > results[1][2]  # 첫 번째 점수가 더 높다

    def test_search_respects_top_k(self, populated_store, mock_embedder):
        query = mock_embedder.encode(["환불"])[0]
        results = populated_store.search(query, top_k=1)
        assert len(results) == 1

    def test_len_reflects_added_docs(self, empty_store, mock_embedder):
        assert len(empty_store) == 0

        v = mock_embedder.encode(["테스트"])[0].reshape(1, -1)
        empty_store.add(["id1"], ["테스트"], v)
        assert len(empty_store) == 1

    def test_add_with_mismatched_lengths_raises(self, empty_store):
        vectors = np.array([[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="same length"):
            empty_store.add(["a", "b"], ["text"], vectors)

    def test_add_1d_vector_raises(self, empty_store):
        vectors = np.array([1.0, 0.0, 0.0])  # 1D
        with pytest.raises(ValueError, match="2-dimensional"):
            empty_store.add(["a"], ["text"], vectors)

    def test_multiple_adds_accumulate(self, empty_store, mock_embedder):
        v1 = mock_embedder.encode(["환불"])[0].reshape(1, -1)
        v2 = mock_embedder.encode(["배송"])[0].reshape(1, -1)

        empty_store.add(["a"], ["환불"], v1)
        empty_store.add(["b"], ["배송"], v2)

        assert len(empty_store) == 2

    def test_search_score_is_cosine_similarity(
        self, empty_store, mock_embedder
    ):
        # Given — 정확히 [1,0,0] 벡터를 저장
        v = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        empty_store.add(["d1"], ["text"], v)

        # When — 동일한 방향의 쿼리
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = empty_store.search(query)

        # Then — 동일 방향이므로 코사인 유사도 ≈ 1.0
        assert results[0][2] == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# RAGPipeline
# ---------------------------------------------------------------------------

class TestRAGPipeline:
    def test_retrieve_returns_relevant_docs(
        self, mock_embedder, populated_store
    ):
        # Given
        pipe = RAGPipeline(mock_embedder, populated_store)

        # When
        docs = pipe.retrieve("환불해주세요", top_k=2)

        # Then
        assert len(docs) == 2
        # "환불" 클러스터의 문서가 최상위에 와야 한다
        assert "환불" in docs[0][1] or "반품" in docs[0][1]

    def test_build_prompt_includes_documents(
        self, mock_embedder, populated_store
    ):
        # Given
        pipe = RAGPipeline(mock_embedder, populated_store)
        docs = pipe.retrieve("환불", top_k=1)

        # When
        prompt = pipe.build_prompt("환불 기간은?", docs)

        # Then — 둘 다 같은 벡터이므로 d1 또는 d2 어느 쪽이든 먼저 올 수 있다
        assert "환불" in prompt or "반품" in prompt
        assert "환불 기간은?" in prompt
        assert "참고 문서" in prompt

    def test_ask_returns_complete_result(self, mock_embedder, populated_store):
        # Given
        pipe = RAGPipeline(mock_embedder, populated_store)

        # When
        result = pipe.ask("배송은 얼마나 걸리나요?")

        # Then
        assert result["query"] == "배송은 얼마나 걸리나요?"
        assert len(result["documents"]) > 0
        assert len(result["prompt"]) > 0
        assert "배송" in result["documents"][0]["text"]
        assert result["answer"] is None  # LLM 없이 호출했으므로 None


# ---------------------------------------------------------------------------
# explain_rag_benefits
# ---------------------------------------------------------------------------

class TestExplainRagBenefits:
    def test_returns_five_benefits(self):
        benefits = explain_rag_benefits()
        assert len(benefits) == 5
        assert all(isinstance(b, str) for b in benefits)

    def test_includes_key_concepts(self):
        benefits = explain_rag_benefits()
        texts = " ".join(benefits)
        assert "할루시네이션" in texts or "환각" in texts
        assert "벡터" in texts or "저장소" in texts or "재학습" in texts
        assert "비용" in texts or "토큰" in texts


# ---------------------------------------------------------------------------
# keyword_vs_semantic_comparison
# ---------------------------------------------------------------------------

class TestKeywordVsSemantic:
    def test_handles_missing_keyword_query(self, mock_embedder):
        """키워드가 질문에 없어도 의미 검색이 동작함을 보인다."""
        # Given
        store = InMemoryVectorStore()
        docs = [
            ("d1", "환불은 구매 후 7일 이내에 가능합니다"),
        ]
        v = mock_embedder.encode([docs[0][1]])
        store.add(["d1"], [docs[0][1]], v)

        # When — "환불"이라는 단어가 없는 질문
        q = "돈을 돌려받고 싶어요"
        q_vec = mock_embedder.encode([q])[0]
        results = store.search(q_vec, top_k=1)

        # Then — 의미 검색으로 찾을 수 있다
        assert len(results) == 1
        assert "환불" in results[0][1]  # 원본에 "환불"이 있음
        assert results[0][2] > 0.9  # 유사도가 높다 (mock에서는 [1,0,0] 방향 일치)


# ---------------------------------------------------------------------------
# Embedder interface
# ---------------------------------------------------------------------------

class TestSentenceEmbedderIntegration:
    """실제 sentence-transformers 모델을 사용하는 통합 테스트.

    모델 다운로드가 필요하므로 CI에서는 캐싱이 권장된다.
    """

    @pytest.mark.slow
    def test_encode_returns_2d_array(self):
        # Given
        embedder = SentenceEmbedder()

        # When
        vectors = embedder.encode(["hello world", "goodbye world"])

        # Then
        assert vectors.ndim == 2
        assert vectors.shape[0] == 2
        assert vectors.shape[1] == embedder.dim

    @pytest.mark.slow
    def test_dim_is_positive_int(self):
        embedder = SentenceEmbedder()
        assert isinstance(embedder.dim, int)
        assert embedder.dim > 0

    @pytest.mark.slow
    def test_similarity_range(self):
        embedder = SentenceEmbedder()
        v = embedder.encode(["hello", "goodbye"])
        sim = embedder.similarity(v[0], v[0])  # same vector
        assert sim == pytest.approx(1.0, abs=1e-4)

    @pytest.mark.slow
    def test_semantic_similarity_difference(self):
        """의미가 비슷한 문장 vs 다른 문장 간 유사도 차이 검증."""
        embedder = SentenceEmbedder()
        texts = [
            "The weather is beautiful today",
            "It is sunny and warm outside",
            "Python is a programming language",
        ]
        vectors = embedder.encode(texts)

        similar = embedder.similarity(vectors[0], vectors[1])
        different = embedder.similarity(vectors[0], vectors[2])

        assert similar > different  # 유사한 문장이 더 높은 점수


# ---------------------------------------------------------------------------
# demonstrate functions (smoke test)
# ---------------------------------------------------------------------------

class TestDemonstrateFunctions:
    @pytest.mark.slow
    def test_demonstrate_embedding_basics(self):
        result = demonstrate_embedding_basics()
        assert "vector_shape" in result
        assert "similar_meaning" in result
        assert "different_meaning" in result
        # 의미가 비슷한 문장이 더 높은 유사도를 보여야 한다
        assert result["similar_meaning"] > result["different_meaning"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_text(self, mock_embedder):
        v = mock_embedder.encode(["하나"])
        assert v.shape == (1, 3)

    def test_empty_search_top_k_larger_than_store(
        self, mock_embedder, populated_store
    ):
        query = mock_embedder.encode(["환불"])[0]
        # top_k > 저장된 문서 수 → 전체 반환
        results = populated_store.search(query, top_k=100)
        assert len(results) == 6  # 6개 저장되어 있음

    def test_zero_vector_similarity(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        # 0으로 나누기 방지 (1e-10 epsilon)
        result = cosine_similarity_manual(a, b)
        assert not np.isnan(result)
        assert not np.isinf(result)


# ---------------------------------------------------------------------------
# OllamaEmbedder
# ---------------------------------------------------------------------------

_MOCK_OLLAMA_RESPONSE = {
    "model": "mxbai-embed-large:335m",
    "embeddings": [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
    ],
}


class TestOllamaEmbedderMock:
    def test_encode_calls_ollama_api_correctly(self):
        # Given
        import requests

        mock_resp = Mock(spec=requests.Response)
        mock_resp.json.return_value = _MOCK_OLLAMA_RESPONSE
        mock_resp.raise_for_status = Mock()

        with patch("requests.post", return_value=mock_resp) as mock_post:
            embedder = OllamaEmbedder(base_url="http://localhost:11434")

            # When
            result = embedder.encode(["hello", "world"])

            # Then
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "http://localhost:11434/api/embed"
            assert call_args[1]["json"]["model"] == "mxbai-embed-large:335m"
            assert call_args[1]["json"]["input"] == ["hello", "world"]
            assert result.shape == (2, 4)

    def test_encode_uses_custom_base_url(self):
        import requests

        mock_resp = Mock(spec=requests.Response)
        mock_resp.json.return_value = _MOCK_OLLAMA_RESPONSE
        mock_resp.raise_for_status = Mock()

        with patch("requests.post", return_value=mock_resp) as mock_post:
            embedder = OllamaEmbedder(base_url="http://192.168.1.100:8080")
            embedder.encode(["test"])

            mock_post.assert_called_once()
            assert mock_post.call_args[0][0] == "http://192.168.1.100:8080/api/embed"

    def test_dim_infers_from_encode(self):
        import requests

        mock_resp = Mock(spec=requests.Response)
        mock_resp.json.return_value = _MOCK_OLLAMA_RESPONSE
        mock_resp.raise_for_status = Mock()

        with patch("requests.post", return_value=mock_resp):
            embedder = OllamaEmbedder()
            # dim은 첫 encode 이후에 알 수 있다
            embedder.encode(["hello"])
            assert embedder.dim == 4

    def test_dim_triggers_encode_when_not_called(self):
        import requests

        mock_resp = Mock(spec=requests.Response)
        mock_resp.json.return_value = _MOCK_OLLAMA_RESPONSE
        mock_resp.raise_for_status = Mock()

        with patch("requests.post", return_value=mock_resp):
            embedder = OllamaEmbedder()
            # encode 없이 dim 접근 → 자동으로 dummy encode
            assert embedder.dim == 4

    def test_similarity_uses_cosine(self):
        import requests

        mock_resp = Mock(spec=requests.Response)
        mock_resp.json.return_value = _MOCK_OLLAMA_RESPONSE
        mock_resp.raise_for_status = Mock()

        with patch("requests.post", return_value=mock_resp):
            embedder = OllamaEmbedder()
            a = np.array([1.0, 0.0, 0.0, 0.0])
            b = np.array([1.0, 0.0, 0.0, 0.0])
            assert embedder.similarity(a, b) == pytest.approx(1.0)

    def test_custom_model_name_is_passed_to_api(self):
        import requests

        mock_resp = Mock(spec=requests.Response)
        mock_resp.json.return_value = _MOCK_OLLAMA_RESPONSE
        mock_resp.raise_for_status = Mock()

        with patch("requests.post", return_value=mock_resp) as mock_post:
            embedder = OllamaEmbedder(model="nomic-embed-text")
            embedder.encode(["test"])

            assert mock_post.call_args[1]["json"]["model"] == "nomic-embed-text"
