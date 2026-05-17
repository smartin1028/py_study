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
from src.embedding_study.menu_demo import (
    build_menu_similarity_matrix,
    build_menu_vectors,
    get_menu_data,
    plot_2d_scatter,
    plot_3d_scatter,
    plot_similarity_heatmap,
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
    def test_encode_calls_ollama_api_correctly(self, monkeypatch):
        # Given
        monkeypatch.setenv("OLLAMA_EMBED_MODEL", "bge-m3:567m")
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
            assert call_args[1]["json"]["model"] == "bge-m3:567m"
            assert call_args[1]["json"]["input"] == ["hello", "world"]
            assert result.shape == (2, 4)

    def test_encode_uses_custom_base_url(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_EMBED_MODEL", "bge-m3:567m")
        import requests

        mock_resp = Mock(spec=requests.Response)
        mock_resp.json.return_value = _MOCK_OLLAMA_RESPONSE
        mock_resp.raise_for_status = Mock()

        with patch("requests.post", return_value=mock_resp) as mock_post:
            embedder = OllamaEmbedder(base_url="http://192.168.1.100:8080")
            embedder.encode(["test"])

            mock_post.assert_called_once()
            assert mock_post.call_args[0][0] == "http://192.168.1.100:8080/api/embed"

    def test_dim_infers_from_encode(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_EMBED_MODEL", "bge-m3:567m")
        import requests

        mock_resp = Mock(spec=requests.Response)
        mock_resp.json.return_value = _MOCK_OLLAMA_RESPONSE
        mock_resp.raise_for_status = Mock()

        with patch("requests.post", return_value=mock_resp):
            embedder = OllamaEmbedder()
            # dim은 첫 encode 이후에 알 수 있다
            embedder.encode(["hello"])
            assert embedder.dim == 4

    def test_dim_triggers_encode_when_not_called(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_EMBED_MODEL", "bge-m3:567m")
        import requests

        mock_resp = Mock(spec=requests.Response)
        mock_resp.json.return_value = _MOCK_OLLAMA_RESPONSE
        mock_resp.raise_for_status = Mock()

        with patch("requests.post", return_value=mock_resp):
            embedder = OllamaEmbedder()
            # encode 없이 dim 접근 → 자동으로 dummy encode
            assert embedder.dim == 4

    def test_similarity_uses_cosine(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_EMBED_MODEL", "bge-m3:567m")
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


# ---------------------------------------------------------------------------
# 메뉴 임베딩용 mock fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_menu_embedder() -> Mock:
    """메뉴 카테고리별로 구분된 4차원 벡터를 반환하는 mock Embedder.

    메뉴 설명 텍스트에 포함된 키워드로 카테고리를 판별하여 one-hot 형태의
    벡터를 반환한다. 같은 카테고리 내 항목들은 코사인 유사도가 1.0,
    다른 카테고리 간은 0.0이 되도록 설계되어 있다.

    매핑 규칙:
        - "espresso" 또는 "coffee" 포함 → Coffee   [1, 0, 0, 0]
        - "tea" 또는 "lemon" 포함      → Tea       [0, 1, 0, 0]
        - "cake", "brownie", "chocolate" 포함 → Dessert [0, 0, 1, 0]
        - 그 외                          → Unknown   [0, 0, 0, 1]
    """
    embedder = Mock(spec=Embedder)
    embedder.dim = 4

    def encode_side_effect(texts: list[str]) -> np.ndarray:
        vectors = []
        for t in texts:
            lower = t.lower()
            if "espresso" in lower or "coffee" in lower:
                vectors.append([1.0, 0.0, 0.0, 0.0])  # Coffee 클러스터
            elif "tea" in lower or "lemon" in lower:
                vectors.append([0.0, 1.0, 0.0, 0.0])  # Tea 클러스터
            elif "cake" in lower or "brownie" in lower or "chocolate" in lower:
                vectors.append([0.0, 0.0, 1.0, 0.0])  # Dessert 클러스터
            else:
                vectors.append([0.0, 0.0, 0.0, 1.0])  # Unknown 클러스터
        return np.array(vectors, dtype=np.float32)

    embedder.encode.side_effect = encode_side_effect
    embedder.similarity.side_effect = lambda a, b: cosine_similarity_manual(
        np.asarray(a), np.asarray(b)
    )
    return embedder


# ---------------------------------------------------------------------------
# TestMenuEmbedding — 메뉴 임베딩 기능 단위 테스트
# ---------------------------------------------------------------------------


class TestMenuEmbedding:
    """메뉴 데이터, 유사도 행렬, 차트 생성 기능을 검증한다.

    모든 테스트는 mock_menu_embedder fixture를 사용하여 실제 모델 없이
    수 밀리초 내에 실행된다.
    """

    def test_menu_data_has_eight_items(self):
        """메뉴 데이터는 8개 항목이며 모든 필수 키를 포함해야 한다."""
        # When
        items = get_menu_data()

        # Then
        assert len(items) == 8
        for item in items:
            assert "category" in item, f"category 키 누락: {item}"
            assert "name" in item, f"name 키 누락: {item}"
            assert "desc" in item, f"desc 키 누락: {item}"
            assert "price" in item, f"price 키 누락: {item}"

    def test_menu_data_categories(self):
        """카테고리는 Coffee 4개, Tea 2개, Dessert 2개로 구성되어야 한다."""
        # When
        items = get_menu_data()

        # Then
        categories = {item["category"] for item in items}
        assert categories == {"Coffee", "Tea", "Dessert"}
        coffee_count = sum(1 for item in items if item["category"] == "Coffee")
        tea_count = sum(1 for item in items if item["category"] == "Tea")
        dessert_count = sum(1 for item in items if item["category"] == "Dessert")
        assert coffee_count == 4
        assert tea_count == 2
        assert dessert_count == 2

    def test_similarity_matrix_shape(self, mock_menu_embedder):
        """유사도 행렬은 8x8 크기이며 대각선은 1.0, 행렬은 대칭이어야 한다."""
        # When
        matrix, labels = build_menu_similarity_matrix(mock_menu_embedder)

        # Then
        assert matrix.shape == (8, 8)
        assert len(labels) == 8
        # 자기 자신과의 유사도는 항상 1.0
        for i in range(8):
            assert matrix[i][i] == pytest.approx(1.0, abs=1e-6)
        # 코사인 유사도 행렬은 대칭이다: sim(A,B) == sim(B,A)
        assert np.allclose(matrix, matrix.T)

    def test_coffee_items_more_similar_than_cross_category(self, mock_menu_embedder):
        """같은 카테고리(커피-커피)가 다른 카테고리(커피-디저트)보다 유사도가 높아야 한다."""
        # Given: Americano=idx0, Cappuccino=idx2, Brownie=idx7
        matrix, _ = build_menu_similarity_matrix(mock_menu_embedder)

        # When
        coffee_sim = matrix[0][2]  # Americano ↔ Cappuccino (둘 다 Coffee)
        cross_sim = matrix[0][7]   # Americano ↔ Brownie (Coffee ↔ Dessert)

        # Then: 같은 카테고리 내 유사도가 교차 카테고리보다 높다
        assert coffee_sim > cross_sim

    def test_intra_coffee_above_intra_dessert(self, mock_menu_embedder):
        """mock 기준으로 커피 내 평균 유사도와 디저트 내 평균 유사도는 모두 1.0이다."""
        matrix, labels = build_menu_similarity_matrix(mock_menu_embedder)

        # 라벨 텍스트에서 카테고리를 역추론하여 인덱스 분류
        coffee_indices = [i for i, lbl in enumerate(labels) if "Americano" in lbl
                          or "Latte" in lbl or "Cappuccino" in lbl or "Espresso" in lbl]
        dessert_indices = [i for i, lbl in enumerate(labels) if "Cheesecake" in lbl
                           or "Brownie" in lbl]

        # 같은 카테고리 내 대각선을 제외한 상삼각 평균
        intra_coffee = float(np.mean([matrix[i][j] for i in coffee_indices
                                      for j in coffee_indices if i < j]))
        intra_dessert = float(np.mean([matrix[i][j] for i in dessert_indices
                                       for j in dessert_indices if i < j]))

        # mock 벡터는 같은 카테고리 내에서 완전히 동일하므로 유사도 1.0
        assert intra_coffee == pytest.approx(1.0, abs=1e-6)
        assert intra_dessert == pytest.approx(1.0, abs=1e-6)

    def test_similarity_matrix_values_in_range(self, mock_menu_embedder):
        """유사도 행렬의 모든 값은 코사인 유사도 정의역 [-1, 1] 내에 있어야 한다."""
        matrix, _ = build_menu_similarity_matrix(mock_menu_embedder)

        assert np.all(matrix >= -1.0)
        assert np.all(matrix <= 1.0)

    def test_heatmap_file_created(self, mock_menu_embedder, tmp_path):
        """plot_similarity_heatmap이 유효한 PNG 파일을 생성해야 한다."""
        # Given
        matrix, labels = build_menu_similarity_matrix(mock_menu_embedder)
        save_path = str(tmp_path / "test_heatmap.png")

        # When
        plot_similarity_heatmap(matrix, labels, save_path=save_path)

        # Then: 파일이 존재하고 크기가 0보다 크다
        import os
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0, "PNG 파일이 비어 있음"

    def test_2d_scatter_file_created(self, mock_menu_embedder, tmp_path):
        """plot_2d_scatter가 유효한 PNG 파일을 생성해야 한다."""
        # Given
        vectors, names, categories = build_menu_vectors(mock_menu_embedder)
        save_path = str(tmp_path / "test_2d_scatter.png")

        # When
        plot_2d_scatter(vectors, names, categories, save_path=save_path)

        # Then
        import os
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_build_menu_vectors_shape(self, mock_menu_embedder):
        """build_menu_vectors는 (8, dim) 벡터와 올바른 메타데이터를 반환해야 한다."""
        # When
        vectors, names, categories = build_menu_vectors(mock_menu_embedder)

        # Then
        assert vectors.shape == (8, 4), f"예상 (8, 4), 실제 {vectors.shape}"
        assert len(names) == 8
        assert len(categories) == 8
        assert set(categories) == {"Coffee", "Tea", "Dessert"}

    def test_3d_scatter_file_created(self, mock_menu_embedder, tmp_path):
        """plot_3d_scatter가 유효한 PNG 파일을 생성해야 한다."""
        # Given
        vectors, names, categories = build_menu_vectors(mock_menu_embedder)
        save_path = str(tmp_path / "test_3d_scatter.png")

        # When
        plot_3d_scatter(vectors, names, categories, save_path=save_path)

        # Then
        import os
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0
