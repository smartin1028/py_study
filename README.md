# pyTool

개인 학습용 프로젝트 — Pandas 예제, 임베딩/RAG 스터디, FastAPI + Claude API 도구 호출 데모.

## 프로젝트 구성

```
pyTool/
├── src/
│   ├── pandas_study/       # Pandas 100 예제 (01~100번)
│   │   ├── _interface.py   #   DataFrameExample(ABC), SupportsSave(Protocol)
│   │   ├── io_examples.py  #   01-10: CSV, Excel, Parquet, JSON, SQL I/O
│   │   ├── core.py         #   11-33: 생성, 선택, 필터링, 정렬
│   │   ├── aggregation.py  #   34-48: groupby, pivot, crosstab, melt
│   │   ├── combine.py      #   49-58: merge, join, concat, cut/qcut
│   │   ├── cleaning.py     #   59-79: 결측치, 문자열, datetime 처리
│   │   └── advanced.py     #   80-100: rolling, ewm, pipe, explode
│   │
│   ├── embedding_study/    # 임베딩/RAG/LLM 학습
│   │   ├── _interface.py   #   Embedder, VectorStore ABC
│   │   ├── basics.py       #   SentenceTransformer 기초
│   │   ├── search.py       #   의미 검색 (코사인 유사도)
│   │   ├── rag.py          #   RAG 파이프라인
│   │   └── llm_integration.py #  4가지 시나리오 비교 (LLM only → Full RAG)
│   │
│   ├── utils/               #   환경 변수 설정 중앙화 (Config)
│   │   └── env_config.py     #     .env 자동 로드 + Config 클래스
│   │
│   └── app/                # FastAPI + Claude API 도구 호출 데모
│       ├── tool_demo_main.py   # 다양한 demo 스크립트
│       ├── controller/     #   API 엔드포인트
│       ├── service/        #   비즈니스 로직
│       ├── repository/     #   데이터 접근
│       └── schema/         #   Pydantic 모델
│
├── tests/                  # 테스트 (1437줄)
│   ├── test_pandas_study.py      # Pandas 예제 79개 테스트
│   ├── test_embedding_study.py   # 임베딩 유닛 테스트
│   ├── test_llm_integration.py   # LLM 통합 테스트
│   └── mock_testing_guide.py     # Mock 테스트 가이드 (Python ↔ Java 비교)
│
├── docs/                   # 문서
│   ├── index.md
│   ├── embedding-guide.md
│   └── pytest-vs-java.md
│
├── pyproject.toml          # 프로젝트 메타데이터 및 의존성
└── uv.lock                 # 잠금 파일
```

## 환경 설정

- Python >= 3.13
- 패키지 매니저: `uv`

```bash
# 의존성 설치
uv sync

# 개발 의존성 포함 설치
uv sync --dev
```

### 환경 변수

`src/utils/env_config.py`의 `Config` 클래스로 모든 환경 변수 접근을 중앙화한다.
`.env.example`을 복사하여 `.env` 파일을 생성하면 자동으로 로드된다.

```bash
cp .env.example .env
```

```python
from src.utils import config

config.anthropic_api_key   # ANTHROPIC_API_KEY
config.anthropic_model     # ANTHROPIC_MODEL (기본값: claude-sonnet-4-6)
config.deepseek_api_key    # DEEPSEEK_API_KEY
```

| 변수 | 설명 |
|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API 키 (embedding_study LLM 통합 시나리오용) |

## 사용법

### Pandas 예제

```bash
# 특정 모듈 실행
uv run python -c "from src.pandas_study import IOExamples; print(IOExamples().run())"
uv run python -c "from src.pandas_study import CoreExamples; print(CoreExamples().run())"
uv run python -c "from src.pandas_study import AggregationExamples; print(AggregationExamples().run())"
```

### 임베딩/RAG 예제

```bash
uv run python -c "from src.embedding_study import demonstrate_embedding_basics; demonstrate_embedding_basics()"
uv run python -c "from src.embedding_study import demonstrate_semantic_search; demonstrate_semantic_search()"
uv run python -c "from src.embedding_study import demonstrate_rag_pipeline; demonstrate_rag_pipeline()"
```

### FastAPI 서버

```bash
uv run uvicorn src.app:app --reload
```

### Claude API 도구 호출 데모

```bash
uv run python src/app/script/tool_demo_claude.py
```

## 테스트

```bash
# 전체 테스트
uv run pytest

# 상세 출력
uv run pytest tests/ -v

# 커버리지 리포트
uv run pytest tests/ --cov=src --cov-report=term-missing

# 특정 모듈만
uv run pytest tests/test_pandas_study.py -v
```

## 의존성 관리

| 명령 | 설명 |
|------|------|
| `uv add <pkg>` | 프로덕션 의존성 추가 |
| `uv add --dev <pkg>` | 개발 의존성 추가 |
| `uv remove <pkg>` | 의존성 제거 |
| `uv lock` | `uv.lock` 갱신 |
| `uv sync` | 가상환경 동기화 |

## 주요 의존성

| 패키지 | 용도 |
|--------|------|
| `pandas>=3.0` | 데이터 분석 |
| `fastapi` + `uvicorn` | Web API 서버 |
| `anthropic` | Claude API |
| `sentence-transformers` | 임베딩 모델 |
| `scikit-learn` | 코사인 유사도 계산 |
| `pytest` + `pytest-mock` | 테스트 |

## 패키지 설계 원칙

- `_interface.py`에 ABC/Protocol을 정의하고 각 모듈에서 구현한다.
- 모든 Pandas 예제는 `DataFrameExample` ABC를 상속하고 `run()` 메서드를 구현한다.
- `run()`은 `dict[str, pd.DataFrame | pd.Series]`를 반환한다.
- 각 모듈의 `__init__.py`는 re-export만 담고 로직은 넣지 않는다.

## 라이선스

MIT License — 자세한 내용은 [LICENSE](./LICENSE) 파일을 참조.
