# Vector Database 가이드

## 벡터 DB란?

벡터 DB는 텍스트, 이미지, 음성 등을 **임베딩(embedding)**이라는 고차원 숫자 벡터로 변환해
저장하고, **유사도 검색**(비슷한 벡터 찾기)에 특화된 데이터베이스다.

기존 DB가 `WHERE name = '아메리카노'` 같은 정확 매칭이라면, 벡터 DB는
"아메리카노와 가장 비슷한 의미를 가진 메뉴" 같은 의미적 유사도 검색을 수행한다.

## 핵심 개념

### 벡터 임베딩

- 텍스트/이미지/음성 등을 고정 길이 실수 배열(`[0.12, -0.34, 0.56, ...]`)로 변환한 것.
- 의미가 비슷한 데이터는 벡터 공간에서도 가까운 위치에 놓인다.
- 임베딩 생성 모델: `sentence-transformers`, OpenAI Embeddings, CLIP(이미지) 등.

### 유사도 측정

| 방식 | 설명 |
|------|------|
| 코사인 유사도 | 두 벡터 간 각도. 1.0이면 완전히 같은 의미. 가장 널리 사용됨 |
| 유클리드 거리 | 두 벡터 간 직선 거리. 작을수록 유사 |
| 내적(Dot Product) | 두 벡터의 방향과 크기를 곱한 값. 클수록 유사 |

### ANN(Approximate Nearest Neighbor)

- 수백만~수억 개 벡터에서 정확한 최근접 이웃을 찾는 건 너무 느리다.
- ANN은 약간의 정확도를 희생해 검색 속도를 수천 배 높인다.
- 대표 알고리즘: HNSW(Hierarchical Navigable Small World), IVF(Inverted File), PQ(Product Quantization).

## 기존 DB와의 비교

| | 관계형 DB | 벡터 DB |
|---|---|---|
| 검색 방식 | 정확 매칭 (`WHERE`, `LIKE`) | 의미적 유사도 (코사인 유사도 등) |
| 인덱스 | B-Tree, Hash, GIN | HNSW, IVF, DiskANN |
| 질의 예 | `SELECT * FROM menu WHERE category = 'Coffee'` | "커피와 비슷한 것" → 벡터 유사도 정렬 |
| 장점 | 트랜잭션, 조인, 정합성 | 의미 검색, 추천, RAG, 멀티모달 |
| 단점 | 의미 검색 불가 | 트랜잭션, 조인에 약함 |

## 기본 사용 흐름

```python
import chromadb
from sentence_transformers import SentenceTransformer

# 1) 임베딩 모델 로드
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384차원

# 2) 벡터 DB 클라이언트 (ChromaDB 예시)
client = chromadb.Client()
collection = client.get_or_create_collection("menu")

# 3) 데이터 임베딩 → 저장
menu_items = [
    "아메리카노: 에스프레소 + 물",
    "카페라떼: 에스프레소 + 스팀 밀크",
    "레모네이드: 레몬즙 + 탄산수",
]
embeddings = model.encode(menu_items)
collection.add(
    ids=["1", "2", "3"],
    embeddings=embeddings,
    documents=menu_items,
)

# 4) 유사도 검색
query = "뜨거운 커피"
query_vec = model.encode(query)
results = collection.query(
    query_embeddings=[query_vec],
    n_results=2,
)
# → "아메리카노", "카페라떼" 순으로 반환
```

## 주요 벡터 DB 종류

### 전용 벡터 DB

| 제품 | 특징 | 적합한 상황 |
|------|------|-------------|
| **ChromaDB** | Python 네이티브, 경량, 오픈소스 | 로컬 데모, 소규모 PoC, 임베딩 학습 |
| **Milvus** | 대규모 분산 처리, GPU 가속 | 10억 건 이상, 엔터프라이즈 |
| **Weaviate** | GraphQL API, 하이브리드 검색 | 모듈형 아키텍처, 스키마 기반 |
| **Qdrant** | Rust 기반, 고성능, 필터링 | 빠른 필터 + 벡터 검색 |
| **Pinecone** | 완전 관리형 SaaS | 인프라 관리 없는 프로덕션 |
| **FAISS** (Meta) | 라이브러리 (DB 아님), GPU 최적화 | 연구, 초고속 로컬 검색 |

### 기존 DB의 벡터 확장

| DB | 확장 | 특징 |
|------|------|------|
| **PostgreSQL** | `pgvector` | SQL + 벡터 검색 동시에, 기존 테이블에 벡터 컬럼 추가 가능 |
| **MongoDB** | Atlas Vector Search | 문서 DB + 벡터 검색, 관리형 |
| **Elasticsearch** | `dense_vector` | 전문 검색 + 벡터 검색 하이브리드 |
| **Redis** | Redis Stack | 초저지연 벡터 검색, 캐시 용도 겸용 |

## 기존 DB에 벡터 저장하기 (pgvector)

PostgreSQL의 `pgvector` 확장을 사용하면 기존 관계형 DB에 벡터 데이터를 함께 저장할 수 있다.

```sql
-- 확장 설치
CREATE EXTENSION vector;

-- 기존 테이블에 벡터 컬럼 추가
ALTER TABLE menu ADD COLUMN embedding vector(384);

-- 벡터 인덱스 생성 (HNSW)
CREATE INDEX ON menu USING hnsw (embedding vector_cosine_ops);

-- 유사도 검색
SELECT
    name,
    description,
    1 - (embedding <=> query_embedding) AS similarity
FROM menu
ORDER BY embedding <=> query_embedding
LIMIT 5;
```

```python
# Python에서 pgvector 사용 예시
import psycopg2
from pgvector.psycopg2 import register_vector

conn = psycopg2.connect("postgresql://localhost/mydb")
register_vector(conn)
cur = conn.cursor()

# 저장
cur.execute(
    "INSERT INTO menu (name, embedding) VALUES (%s, %s)",
    ("아메리카노", embedding.tolist()),
)

# 검색
cur.execute(
    "SELECT name FROM menu ORDER BY embedding <=> %s LIMIT 5",
    (query_embedding.tolist(),),
)
```

## 주요 활용 사례

| 사례 | 설명 |
|------|------|
| **RAG (Retrieval-Augmented Generation)** | LLM 질의 전에 관련 문서를 벡터 검색 → 컨텍스트로 주입 |
| **의미적 검색** | 키워드가 아닌 의미로 문서/상품 검색 |
| **추천 시스템** | 사용자 행동 벡터와 유사한 아이템 추천 |
| **이미지 검색** | 이미지 → CLIP 임베딩 → 유사 이미지 검색 |
| **이상 탐지** | 정상 패턴 벡터에서 멀리 떨어진 이상치 검출 |
| **챗봇 장기 기억** | 과거 대화 벡터 저장 → 유사 맥락의 과거 대화 검색 |

## 선택 가이드

```
소규모 데모 / 학습용인가?
├── 예 → ChromaDB (Python 네이티브, 설정 최소화)
└── 아니오 → 프로덕션 규모인가?
    ├── 기존 PostgreSQL 사용 중 → pgvector
    ├── 기존 MongoDB 사용 중 → MongoDB Atlas Vector Search
    ├── 기존 Elasticsearch 사용 중 → ES dense_vector
    ├── 완전 관리형 원함 → Pinecone
    └── 대규모 자체 호스팅 → Milvus, Qdrant
```

## pyTool 프로젝트에서의 활용

이 프로젝트에는 이미 `sentence-transformers`가 의존성으로 포함되어 있으며,
`docs/menu-embedding-guide.md`에서 메뉴 임베딩 데모를 다루고 있다.

벡터 DB를 추가하려면:

```bash
uv add chromadb
```

이후 `menu_demo.py`의 인메모리 유사도 비교를 ChromaDB 기반으로 확장하거나,
RAG 데모를 `src/app/script/` 아래에 추가할 수 있다.
