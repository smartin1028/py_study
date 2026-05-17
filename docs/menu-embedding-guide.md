# Menu Embedding Heatmap 해석 가이드

## 개요

`menu_demo.py`는 커피숍 메뉴 8개 항목을 임베딩하여 항목 간 의미적 유사도를
히트맵으로 시각화하는 데모입니다.

## 데이터 구성

| 카테고리 | 메뉴 | 설명 |
|----------|------|------|
| Coffee | Americano | espresso with hot water |
| Coffee | Caffe Latte | espresso with steamed milk |
| Coffee | Cappuccino | espresso with steamed milk and foam |
| Coffee | Espresso | concentrated pure coffee shot |
| Tea | Iced Tea | cold brewed black tea with ice |
| Tea | Lemonade | fresh lemon juice with sparkling water |
| Dessert | Cheesecake | new york style cream cheese cake |
| Dessert | Brownie | warm chocolate fudge brownie with nuts |

## 히트맵이 보여주는 것

히트맵은 각 메뉴 항목 쌍의 **코사인 유사도(Cosine Similarity)** 를 색상과 숫자로 표현합니다.

- **0.0 (흰색/노란색)**: 두 항목의 의미가 완전히 다름
- **1.0 (진한 빨간색)**: 두 항목의 의미가 완전히 같음 (대각선은 항상 1.0)
- **대각선**: 자기 자신과의 비교이므로 항상 1.0

## 해석 방법

### 1. 같은 카테고리끼리 뭉쳐 있는가?

커피 메뉴(Americano, Latte, Cappuccino, Espresso)는 모두 "espresso"라는
공통 키워드를 포함하므로 임베딩 벡터가 서로 가까워야 합니다.
→ 히트맵 좌상단 4x4 블록이 진하게 표시됩니다.

### 2. 다른 카테고리 간 차이가 있는가?

Coffee ↔ Dessert 같은 교차 카테고리는 공통된 단어가 거의 없으므로
유사도가 낮게(0.2~0.4) 나옵니다.
→ 히트맵에서 Coffee 행과 Dessert 열이 만나는 지점이 밝게 표시됩니다.

### 3. 가격은 유사도에 영향을 주는가?

가격 정보(`₩3,500`)는 임베딩 모델에게 "숫자"일 뿐 의미적 연관성을
학습하지 않았기 때문에, 같은 4,500원인 Latte와 Lemonade가 특별히
가깝게 배치되지는 않습니다. 이는 임베딩이 **의미 기반**이라는 점을
보여주는 좋은 예시입니다.

## 결과 예시 (실제 실행 시)

```
intra_coffee_similarity:  0.85~0.95  ← 커피끼리 매우 유사
intra_dessert_similarity: 0.70~0.85  ← 디저트끼리 유사
cross_category_similarity: 0.20~0.40 ← 커피 vs 디저트는 낮음
```

## 실행 방법

```bash
uv run python -c "
from src.embedding_study.menu_demo import demonstrate_menu_embedding
result = demonstrate_menu_embedding(save_chart=True)
print('히트맵:', result['chart_paths']['heatmap'])
print('2D 산점도:', result['chart_paths']['scatter_2d'])
print('3D 산점도:', result['chart_paths']['scatter_3d'])
print('커피 내 유사도:', result['intra_coffee_similarity'])
print('디저트 내 유사도:', result['intra_dessert_similarity'])
print('교차 카테고리 유사도:', result['cross_category_similarity'])
"
```

히트맵(`menu_similarity.png`), 2D 산점도(`menu_2d_scatter.png`),
3D 산점도(`menu_3d_scatter.png`)가 `pyTool/tmp/` 디렉토리에 저장됩니다.

## 3D 산점도 (PCA)

히트맵 외에 **3D 산점도**도 함께 생성됩니다. 384차원 임베딩 벡터를
PCA(주성분 분석)로 3차원으로 축소하여 각 메뉴 항목을 3D 공간에
배치합니다.

- **점 색상**: Coffee(갈색), Tea(녹색), Dessert(분홍색)
- **거리**: 가까운 점일수록 의미가 유사한 항목
- **축**: PC1/PC2/PC3 레이블에 각 성분의 설명 분산 비율이 표시됨

히트맵이 "얼마나 유사한가"를 숫자로 보여준다면, 3D 산점도는
"어떤 항목끼리 뭉쳐 있는가"를 직관적으로 보여줍니다.
