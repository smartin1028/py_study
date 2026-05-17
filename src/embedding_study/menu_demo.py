"""메뉴 임베딩 데모 — 커피숍 메뉴의 의미적 유사도를 임베딩 벡터로 분석하고
히트맵, 2D/3D 산점도로 시각화한다.

핵심 아이디어:
- 같은 카테고리(커피) 내 항목들은 설명 텍스트가 유사하므로 임베딩 벡터도 가깝다.
- 다른 카테고리(커피 vs 디저트)는 벡터 거리가 멀다.
- 가격은 숫자일 뿐 의미적 연관성을 배우지 못했으므로 유사도에 영향을 주지 않는다.
"""

from __future__ import annotations

import numpy as np

from src.embedding_study._interface import Embedder


def _configure_korean_font() -> None:
    """matplotlib에서 한글이 깨지지 않도록 시스템 폰트를 설정한다.

    macOS는 AppleGothic, 그 외는 기본 폰트를 사용한다.
    각 플롯 함수에서 import 후 호출한다.
    """
    import platform

    import matplotlib.pyplot as plt

    if platform.system() == "Darwin":
        plt.rcParams["font.family"] = "AppleGothic"
    # 음수 부호가 깨지지 않도록 설정
    plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------------
# 메뉴 데이터 — 커피숍 8개 항목 (카테고리, 이름, 설명, 가격)
# ---------------------------------------------------------------------------
_MENU_ITEMS: list[dict[str, str | int]] = [
    {"category": "Coffee", "name": "Americano", "desc": "espresso with hot water", "price": 3500},
    {"category": "Coffee", "name": "Caffe Latte", "desc": "espresso with steamed milk", "price": 4500},
    {"category": "Coffee", "name": "Cappuccino", "desc": "espresso with steamed milk and foam", "price": 4500},
    {"category": "Coffee", "name": "Espresso", "desc": "concentrated pure coffee shot", "price": 3000},
    {"category": "Tea", "name": "Iced Tea", "desc": "cold brewed black tea with ice", "price": 4000},
    {"category": "Tea", "name": "Lemonade", "desc": "fresh lemon juice with sparkling water", "price": 4500},
    {"category": "Dessert", "name": "Cheesecake", "desc": "new york style cream cheese cake", "price": 5500},
    {"category": "Dessert", "name": "Brownie", "desc": "warm chocolate fudge brownie with nuts", "price": 5000},
]


def get_menu_data() -> list[dict[str, str | int]]:
    """커피숍 메뉴 데이터를 복사본으로 반환한다.

    반환값은 내부 데이터의 독립적인 복사본이므로 호출부에서 자유롭게 수정 가능하다.
    각 항목은 category, name, desc, price 키를 가진 dict이다.
    """
    return [dict(item) for item in _MENU_ITEMS]


def build_menu_similarity_matrix(embedder: Embedder) -> tuple[np.ndarray, list[str]]:
    """메뉴 설명을 임베딩하여 항목 간 코사인 유사도 행렬을 계산한다.

    동작 순서:
    1. 각 메뉴 항목의 "이름: 설명 (₩가격)" 텍스트를 임베딩한다.
    2. 임베딩 벡터를 L2 정규화한다 (벡터 크기를 1로 통일).
    3. 정규화된 벡터들의 내적(dot product)으로 코사인 유사도 행렬을 만든다.
       — 정규화된 벡터의 내적은 코사인 유사도와 수학적으로 동일하다.

    Args:
        embedder: 텍스트 → 벡터 변환을 수행하는 Embedder 구현체.

    Returns:
        (matrix, labels) 튜플.
        - matrix: (8, 8) 크기의 코사인 유사도 행렬. matrix[i][j]는 항목 i와 j의 유사도.
        - labels: 히트맵 축에 표시할 8개 항목의 이름+가격 문자열.
    """
    # 메뉴 이름, 설명, 가격을 하나의 텍스트로 합쳐서 임베딩한다.
    # 가격을 포함하는 이유: "가격이 유사도에 영향을 주지 않는다"는 점을 확인하기 위함.
    texts = [f"{item['name']}: {item['desc']} (₩{item['price']:,})" for item in _MENU_ITEMS]
    vectors = embedder.encode(texts)

    # L2 정규화: 각 벡터를 자신의 크기로 나누어 단위 벡터로 만든다.
    # 1e-10은 0-벡터로 인한 division by zero 방지용 epsilon.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / (norms + 1e-10)

    # 정규화된 벡터의 내적 = 코사인 유사도
    matrix = np.dot(normalized, normalized.T)

    # 히트맵 축 라벨 — 이름 아래 가격을 작게 표시
    labels = [f"{item['name']}\n₩{item['price']:,}" for item in _MENU_ITEMS]
    return matrix, labels


def build_menu_vectors(embedder: Embedder) -> tuple[np.ndarray, list[str], list[str]]:
    """메뉴 항목을 임베딩하여 원본 벡터와 메타정보를 반환한다.

    PCA 산점도에 사용하기 위한 중간 데이터를 제공한다.
    build_menu_similarity_matrix와 달리 정규화하지 않은 원본 벡터를 반환하므로
    PCA로 차원 축소할 때 정보 손실이 적다.

    Args:
        embedder: 텍스트 → 벡터 변환을 수행하는 Embedder 구현체.

    Returns:
        (vectors, names, categories) 튜플.
        - vectors: (8, dim) 크기의 float32 배열.
        - names: 8개 메뉴 이름 목록 (산점도 라벨용).
        - categories: 8개 카테고리 목록 (산점도 색상 구분용).
    """
    texts = [f"{item['name']}: {item['desc']} (₩{item['price']:,})" for item in _MENU_ITEMS]
    vectors = embedder.encode(texts)
    names = [item["name"] for item in _MENU_ITEMS]
    categories = [item["category"] for item in _MENU_ITEMS]
    return vectors, names, categories


def plot_3d_scatter(
    vectors: np.ndarray,
    names: list[str],
    categories: list[str],
    save_path: str | None = None,
) -> None:
    """임베딩 벡터를 PCA 3차원으로 축소하여 3D 산점도를 생성한다.

    PCA(주성분 분석)로 384차원(또는 임의의 고차원) 벡터를 3차원으로 압축한 뒤
    각 메뉴 항목을 3D 공간에 점으로 배치한다. 같은 카테고리끼리는 가까이,
    다른 카테고리끼리는 멀리 위치하는지 직관적으로 확인할 수 있다.

    색상 구분:
        - Coffee: 갈색 (#8B4513)
        - Tea: 녹색 (#228B22)
        - Dessert: 분홍색 (#FF69B4)

    각 축 레이블에는 해당 주성분이 전체 분산에서 설명하는 비율이 표시된다.
    예: "PC1 (78.3%)" → 첫 번째 주성분이 데이터 분산의 78.3%를 설명.

    Args:
        vectors: (N, dim) 원본 임베딩 벡터.
        names: N개 메뉴 이름.
        categories: N개 카테고리 (Coffee / Tea / Dessert).
        save_path: 지정된 경로로 이미지를 저장한다. None이면 저장하지 않는다.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    _configure_korean_font()

    # PCA로 고차원 벡터를 3차원으로 축소
    pca = PCA(n_components=3)
    coords = pca.fit_transform(vectors)

    # 카테고리별 색상 매핑
    category_colors = {"Coffee": "#8B4513", "Tea": "#228B22", "Dessert": "#FF69B4"}

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection="3d")

    # 카테고리별로 점을 찍는다 (같은 카테고리는 같은 색)
    for cat, color in category_colors.items():
        mask = [c == cat for c in categories]
        ax.scatter(
            coords[mask, 0], coords[mask, 1], coords[mask, 2],
            c=color, label=cat, s=80, alpha=0.85,
        )

    # 각 점 옆에 메뉴 이름을 3D 좌표에 맞춰 표시
    for i, name in enumerate(names):
        ax.text(coords[i, 0], coords[i, 1], coords[i, 2], f"  {name}", fontsize=8)

    # PCA 설명 분산 비율을 축 레이블에 포함
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
    ax.set_title("메뉴 항목 임베딩 (PCA 3D)", fontsize=12)
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_2d_scatter(
    vectors: np.ndarray,
    names: list[str],
    categories: list[str],
    save_path: str | None = None,
) -> None:
    """임베딩 벡터를 PCA 2차원으로 축소하여 2D 평면 산점도를 생성한다.

    3D보다 해석이 직관적이며, PC1-PC2 평면에서 카테고리별 군집을
    한눈에 파악할 수 있다. 점이 가까울수록 의미가 유사한 항목이다.

    각 점의 위치에 메뉴 이름이 annotate로 표시되며, 격자선이 있어
    좌표값을 읽기 쉽다.

    색상 구분:
        - Coffee: 갈색 (#8B4513)
        - Tea: 녹색 (#228B22)
        - Dessert: 분홍색 (#FF69B4)

    Args:
        vectors: (N, dim) 원본 임베딩 벡터.
        names: N개 메뉴 이름.
        categories: N개 카테고리 (Coffee / Tea / Dessert).
        save_path: 지정된 경로로 이미지를 저장한다. None이면 저장하지 않는다.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # PCA로 고차원 벡터를 2차원으로 축소
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)

    category_colors = {"Coffee": "#8B4513", "Tea": "#228B22", "Dessert": "#FF69B4"}

    fig, ax = plt.subplots(figsize=(8, 6))

    # 카테고리별로 점을 찍는다
    for cat, color in category_colors.items():
        mask = [c == cat for c in categories]
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, label=cat, s=100, alpha=0.85, edgecolors="white",
        )

    # 점 옆에 메뉴 이름을 약간 띄워서 표시 (겹침 방지)
    for i, name in enumerate(names):
        ax.annotate(name, (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("메뉴 항목 임베딩 (PCA 2D)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)  # 얇은 격자선으로 좌표 가독성 향상

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_similarity_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    save_path: str | None = None,
) -> None:
    """코사인 유사도 행렬을 히트맵으로 시각화한다.

    히트맵의 각 셀은 두 메뉴 항목 간의 코사인 유사도를 나타낸다.
    대각선(자기 자신)은 항상 1.0이다.

    색상 해석:
        - 진한 빨강 (1.0): 의미가 거의 동일한 항목
        - 노랑/흰색 (0.0 근처): 의미가 전혀 다른 항목
        - 주황 (0.5~0.8): 어느 정도 연관된 항목

    각 셀 안에 소수점 둘째 자리까지 숫자로 표시되며, 우측 컬러바로
    색상-값 대응을 확인할 수 있다.

    Args:
        matrix: (N, N) 코사인 유사도 행렬. 값 범위는 [-1, 1].
        labels: N개 행/열 라벨 (메뉴 이름 + 가격).
        save_path: 지정된 경로로 이미지를 저장한다. None이면 저장하지 않는다.
    """
    import matplotlib.pyplot as plt

    _configure_korean_font()

    fig, ax = plt.subplots(figsize=(8, 6))

    # YlOrRd 컬러맵: 0(노랑) → 1(진한 빨강), 직관적인 온도계 느낌
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1)

    n = len(labels)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    # x축 라벨은 45도 기울여서 긴 이름도 겹치지 않게 표시
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("메뉴 항목 간 코사인 유사도", fontsize=12)

    # 각 셀에 유사도 숫자를 표시 (소수점 둘째 자리)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()  # 라벨이 잘리지 않도록 레이아웃 자동 조정

    if save_path:
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


def demonstrate_menu_embedding(save_chart: bool = False) -> dict:
    """메뉴 임베딩 데모를 실행하고 결과를 dict로 반환한다.

    전체 흐름:
    1. SentenceEmbedder로 메뉴 8개를 임베딩한다.
    2. 코사인 유사도 행렬을 계산한다.
    3. 카테고리 내/카테고리 간 평균 유사도를 측정한다.
       - intra_coffee: 커피 항목들끼리의 평균 유사도 (높을수록 좋음)
       - intra_dessert: 디저트 항목들끼리의 평균 유사도
       - cross_category: 커피 vs 디저트 간 평균 유사도 (낮을수록 좋음)
    4. save_chart=True이면 히트맵, 2D 산점도, 3D 산점도를 파일로 저장한다.
       저장 경로: pyTool/tmp/menu_similarity.png 등

    Args:
        save_chart: True이면 차트 이미지를 pyTool/tmp/ 디렉토리에 저장한다.

    Returns:
        결과 dict:
        - matrix_shape: 유사도 행렬 크기 (8, 8)
        - labels: 8개 축 라벨 목록
        - intra_coffee_similarity: 커피 항목 간 평균 유사도
        - intra_dessert_similarity: 디저트 항목 간 평균 유사도
        - cross_category_similarity: 커피 vs 디저트 평균 유사도
        - chart_paths (save_chart=True인 경우): 저장된 파일 경로 dict
    """
    from pathlib import Path

    from src.embedding_study.basics import SentenceEmbedder

    # Sentence-BERT 모델로 메뉴 텍스트를 임베딩 (384차원)
    embedder = SentenceEmbedder()
    matrix, labels = build_menu_similarity_matrix(embedder)

    # 커피와 디저트 인덱스를 추출하여 카테고리 내/간 유사도 계산
    coffee_indices = [i for i, item in enumerate(_MENU_ITEMS) if item["category"] == "Coffee"]
    dessert_indices = [i for i, item in enumerate(_MENU_ITEMS) if item["category"] == "Dessert"]

    # intra: 같은 카테고리 내 대각선을 제외한 상삼각 평균
    intra_coffee = float(np.mean(
        [matrix[i][j] for i in coffee_indices for j in coffee_indices if i < j]
    ))
    intra_dessert = float(np.mean(
        [matrix[i][j] for i in dessert_indices for j in dessert_indices if i < j]
    ))
    # cross: 커피-디저트 교차 쌍의 평균
    cross_category = float(np.mean(
        [matrix[i][j] for i in coffee_indices for j in dessert_indices]
    ))

    saved_paths: dict[str, str] = {}
    if save_chart:
        # 프로젝트 루트의 tmp/ 디렉토리에 저장 (.gitignore에 등록되어 있음)
        tmp_dir = Path(__file__).resolve().parents[2] / "tmp"
        tmp_dir.mkdir(exist_ok=True)

        heatmap_path = str(tmp_dir / "menu_similarity.png")
        plot_similarity_heatmap(matrix, labels, save_path=heatmap_path)
        saved_paths["heatmap"] = heatmap_path

        vectors, names, categories = build_menu_vectors(embedder)

        scatter_3d_path = str(tmp_dir / "menu_3d_scatter.png")
        plot_3d_scatter(vectors, names, categories, save_path=scatter_3d_path)
        saved_paths["scatter_3d"] = scatter_3d_path

        scatter_2d_path = str(tmp_dir / "menu_2d_scatter.png")
        plot_2d_scatter(vectors, names, categories, save_path=scatter_2d_path)
        saved_paths["scatter_2d"] = scatter_2d_path

    result = {
        "matrix_shape": matrix.shape,
        "labels": labels,
        "intra_coffee_similarity": round(intra_coffee, 4),
        "intra_dessert_similarity": round(intra_dessert, 4),
        "cross_category_similarity": round(cross_category, 4),
    }
    if saved_paths:
        result["chart_paths"] = saved_paths

    return result
