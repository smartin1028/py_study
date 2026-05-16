# LLM + Embedding + RAG 시나리오 테스트 가이드

이 문서는 4가지 시나리오(LLM only, 키워드 RAG, Embedding+LLM, Full RAG)를 테스트하는 모든 방법을 다룬다.

---

## 목차

1. [개요: 두 가지 테스트 계층](#1-개요-두-가지-테스트-계층)
2. [단위 테스트 (Mock) — 빠른 피드백](#2-단위-테스트-mock--빠른-피드백)
3. [통합 테스트 (실제 API) — 실제 동작 검증](#3-통합-테스트-실제-api--실제-동작-검증)
4. [수동 시나리오 실행](#4-수동-시나리오-실행)
5. [개별 시나리오만 실행하기](#5-개별-시나리오만-실행하기)
6. [테스트 결과 해석 방법](#6-테스트-결과-해석-방법)
7. [테스트 커스터마이징](#7-테스트-커스터마이징)
8. [트러블슈팅](#8-트러블슈팅)

---

## 1. 개요: 두 가지 테스트 계층

```
┌─────────────────────────────────────────────────┐
│               테스트 피라미드                      │
│                                                   │
│            ┌─────────────┐                        │
│            │  실제 API    │  수동 실행              │
│            │  통합 테스트  │  (느림, 비용 발생)      │
│            └──────┬──────┘                        │
│                   │                               │
│         ┌─────────┴──────────┐                    │
│         │  Mock 단위 테스트    │  CI/CD 자동화       │
│         │  32개 테스트         │  (빠름, 비용 없음)   │
│         └────────────────────┘                    │
└─────────────────────────────────────────────────┘
```

| 계층 | 실행 시간 | API 비용 | 언제 실행? |
|------|:---:|:---:|---|
| **Mock 단위 테스트** | ~3초 | $0 | 코드 변경 시마다 |
| **실제 API 통합 테스트** | ~30~60초 | $0.01~0.05 | 배포 전, 작동 확인용 |
| **수동 시나리오 실행** | ~30초 | $0.01~0.05 | 데모, 결과 분석용 |

---

## 2. 단위 테스트 (Mock) — 빠른 피드백

DeepSeek API와 임베딩 모델을 Mock으로 대체하여 실제 네트워크 호출 없이 실행한다.

### 2.1. 전체 단위 테스트 실행

```bash
# 모든 테스트 (mock + slow 제외)
uv run pytest tests/test_llm_integration.py -v -k "not slow"

# 결과: 32 passed in ~3s
```

### 2.2. 테스트 그룹별 실행

```bash
# DeepSeekLLM 클라이언트만
uv run pytest tests/test_llm_integration.py -v -k "TestDeepSeekLLM"

# 키워드 검색만
uv run pytest tests/test_llm_integration.py -v -k "TestKeywordSearch"

# 의도 분류만
uv run pytest tests/test_llm_integration.py -v -k "TestIntentClassification"

# 시나리오 1~4 전체
uv run pytest tests/test_llm_integration.py -v -k "TestScenario"

# Edge case만
uv run pytest tests/test_llm_integration.py -v -k "TestEdgeCases"
```

### 2.3. 단일 테스트 함수 실행

```bash
uv run pytest tests/test_llm_integration.py::TestKeywordSearch::test_misses_semantic_match -v
```

### 2.4. Mock 테스트의 특징

- **DeepSeek API 호출 없음**: `mock_llm` fixture가 `DeepSeekLLM`을 Mock으로 대체
- **임베딩 모델 다운로드 없음**: `mock_embedder` fixture가 384차원 벡터를 재현 가능한 로직으로 생성
- **벡터 저장소 Mock**: `mock_store` fixture가 미리 정의된 검색 결과 반환
- **의도적 한계 재현**: `"Membership"`에 `"ship"`이 포함된 substring 충돌을 `\b` 단어 경계로 해결

### 2.5. Mock 구조 이해하기

```
test_llm_integration.py
│
├── mock_openai_client   → OpenAI 클라이언트를 Mock으로 대체
│   └── .chat.completions.create() → 가짜 응답 반환
│
├── mock_llm             → DeepSeekLLM.__init__ 우회
│   ├── .ask()           → "LLM only 응답입니다." 반환
│   └── .ask_with_context() → "RAG 응답" 반환
│
├── mock_embedder        → Embedder 인터페이스 구현
│   ├── .encode()        → 키워드 기반 벡터 생성
│   ├── .dim             → 384
│   └── .similarity()    → 코사인 유사도 계산
│
└── mock_store           → VectorStore 인터페이스 구현
    └── .search()        → [(id, text, score), ...] 반환
```

---

## 3. 통합 테스트 (실제 API) — 실제 동작 검증

실제 DeepSeek API와 sentence-transformers 모델을 사용한다.

### 3.1. 사전 준비

```bash
# .env 파일에 API 키가 설정되어 있는지 확인
cat .env | grep DEEPSEEK_API_KEY
# DEEPSEEK_API_KEY=sk-xxxxxxxxxx
```

### 3.2. 느린 통합 테스트 실행

```bash
# @pytest.mark.slow 마크가 붙은 테스트만
uv run pytest tests/test_llm_integration.py -v -k "slow"

# 결과 예시:
# TestSentenceEmbedderIntegration::test_encode_returns_2d_array PASSED
# TestSentenceEmbedderIntegration::test_dim_is_positive_int PASSED
# TestSentenceEmbedderIntegration::test_similarity_range PASSED
# TestSentenceEmbedderIntegration::test_semantic_similarity_difference PASSED
# TestDemonstrateFunctions::test_demonstrate_embedding_basics PASSED
```

### 3.3. 전체 테스트 + slow 포함

```bash
uv run pytest tests/test_llm_integration.py -v
# 32 passed in ~35s (모델 다운로드 + API 호출 포함)
```

---

## 4. 수동 시나리오 실행

실제 API를 호출하여 4가지 시나리오 결과를 콘솔에서 직접 확인한다.

### 4.1. 기본 질문으로 실행

```bash
uv run python -c "
from src.embedding_study.llm_integration import run_all_scenarios, print_comparison
results = run_all_scenarios()
print_comparison(results)
"
```

### 4.2. 커스텀 질문으로 실행

```bash
uv run python -c "
from src.embedding_study.llm_integration import run_all_scenarios, print_comparison

my_questions = [
    '배송비는 얼마인가요?',
    '쿠폰은 몇 개까지 쓸 수 있나요?',
]
results = run_all_scenarios(questions=my_questions)
print_comparison(results)
"
```

### 4.3. 결과를 JSON으로 저장

```bash
uv run python -c "
import json
from src.embedding_study.llm_integration import run_all_scenarios, ScenarioResult

class ScenarioEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ScenarioResult):
            return obj.__dict__
        return super().default(obj)

results = run_all_scenarios()
print(json.dumps(
    {q: [r.__dict__ for r in scenarios] for q, scenarios in results.items()},
    indent=2, ensure_ascii=False, cls=ScenarioEncoder,
))
" > docs/scenario-results.json
```

---

## 5. 개별 시나리오만 실행하기

### 5.1. Python REPL에서 한 단계씩

```python
# 1. 임포트
from src.embedding_study.basics import SentenceEmbedder
from src.embedding_study.llm_integration import (
    DeepSeekLLM,
    build_knowledge_store,
    scenario_1_llm_only,
    scenario_2_llm_rag_keyword,
    scenario_3_embedding_llm,
    scenario_4_full_rag,
)

# 2. 의존성 초기화 (실제 네트워크 호출 발생)
llm = DeepSeekLLM()                       # API 키 로드
embedder = SentenceEmbedder()             # 모델 다운로드 (~80MB, 최초 1회)
store = build_knowledge_store(embedder)   # 문서 벡터화

# 3. 개별 시나리오 실행
question = "환불 기간이 어떻게 되나요?"

r1 = scenario_1_llm_only(llm, question)
print(f"[LLM only]\n{r1.answer}\n")

r2 = scenario_2_llm_rag_keyword(llm, question)
print(f"[키워드 RAG] 문서 {r2.metadata['documents_found']}건\n{r2.answer}\n")

r3 = scenario_3_embedding_llm(embedder, llm, question)
print(f"[Emb+LLM] 의도: {r3.metadata['top_intent']}\n{r3.answer}\n")

r4 = scenario_4_full_rag(embedder, store, llm, question)
print(f"[Full RAG] 문서 {r4.metadata['documents_found']}건\n{r4.answer}")
```

### 5.2. 시나리오별 문서 검색 결과만 보기

```python
# 시나리오 2: 어떤 문서가 키워드로 검색되었는가?
from src.embedding_study.llm_integration import _keyword_search, get_knowledge_base_texts

docs = get_knowledge_base_texts()
found = _keyword_search("환불해주세요", docs)
print(f"키워드 검색 결과: {len(found)}건")
for d in found:
    print(f"  - {d[:80]}...")

# 시나리오 4: 임베딩 유사도 점수 확인
q_vec = embedder.encode(["환불해주세요"])[0]
hits = store.search(q_vec, top_k=5)
print(f"\n임베딩 검색 결과: {len(hits)}건")
for doc_id, text, score in hits:
    print(f"  [{score:.4f}] {text[:80]}...")
```

---

## 6. 테스트 결과 해석 방법

### 6.1. 단위 테스트가 검증하는 것

| 테스트 그룹 | 검증 내용 |
|---|------|
| `TestDeepSeekLLM` | API 클라이언트가 올바른 모델명, 메시지 포맷으로 호출하는지 |
| `TestKeywordSearch` | 키워드 검색이 정확히 일치하는 단어는 찾고, 의미적 변형은 놓치는지 |
| `TestIntentClassification` | 임베딩 의도 분류가 refund/shipping/membership 등을 올바르게 구분하는지 |
| `TestScenario1~4` | 각 시나리오가 올바른 메타데이터를 반환하는지 |
| `TestKnowledgeBase` | 지식 베이스가 5개 문서를 포함하고 모든 주제를 커버하는지 |
| `TestScenarioComparison` | 시나리오 간 상대적 특성(키워드 vs 의미)이 의도대로 동작하는지 |
| `TestEdgeCases` | 빈 입력, 0개 매칭, 0 벡터 등 경계 조건에서도 오류 없이 동작하는지 |

### 6.2. 실제 API 실행 시 체크포인트

시나리오를 실행한 후 아래 기준으로 평가한다:

```
✅ Good (정답)
- 문서의 구체적 사실(날짜, 수치, 절차)을 정확히 인용
- "제공된 문서에 따르면..." 같은 출처 표시
- 모르는 내용은 "문서에 없습니다"라고 정직하게 답변

⚠️ Acceptable (허용 가능)
- 일반적인 정보를 제공하지만 구체적 수치가 없음
- 주제는 맞췄지만 상세 내용 누락

❌ Bad (실패)
- 문서에 없는 내용을 지어냄 (할루시네이션)
- 키워드 불일치로 관련 문서를 전혀 검색하지 못함
- 오분류로 엉뚱한 주제의 답변을 생성
```

### 6.3. 대표적인 성공/실패 패턴

#### 성공: VIP 멤버십 질문 + 키워드 RAG

```
질문: "VIP 멤버십 혜택이 무엇인가요?"
키워드: "VIP", "멤버십"
→ 문서에 두 키워드 모두 존재 → 검색 성공
→ 답변: "10% 할인, 무료 배송, 24시간 고객센터..." ✅
```

#### 실패: 환불 질문 + 키워드 RAG

```
질문: "돈을 돌려받고 싶어요"
키워드: "돈", "돌려받고", "싶어요"
→ 문서에 "환불"만 있고 위 키워드는 없음 → 검색 0건
→ 답변: 일반적인 상담 안내 ❌
```

#### 성공: 개인정보 질문 + Full RAG

```
질문: "개인정보는 얼마나 보관되나요?"
임베딩 유사도 검색 → "개인정보 처리방침" 문서가 1위 (0.42)
→ 답변: "계약 5년, 결제 5년, 분쟁 3년" ✅
```

---

## 7. 테스트 커스터마이징

### 7.1. 새 질문 추가하기

```python
# llm_integration.py의 run_all_scenarios()에 추가
questions = [
    "VIP 멤버십 혜택이 무엇인가요?",
    "돈을 돌려받고 싶어요",
    "개인정보는 얼마나 보관되나요?",
    "신규 가입하면 쿠폰이 발급되나요?",    # ← 추가
    "해외로 상품을 받을 수 있나요?",       # ← 추가
]
```

### 7.2. 새 지식 문서 추가하기

```python
# llm_integration.py의 _KNOWLEDGE_BASE에 추가
{
    "id": "doc_006",
    "title": "A/S 정책",
    "content": (
        "제품 보증 기간은 구매일로부터 1년입니다. "
        "보증 기간 내 무상 수리가 가능하며, "
        "A/S 접수는 고객센터 080-XXX-XXXX로 가능합니다."
    ),
},
```

### 7.3. 다른 LLM 모델로 변경하기

```python
# DeepSeek 대신 다른 모델 사용
llm = DeepSeekLLM(model="deepseek-reasoner")  # 추론 특화 모델

# 또는 Claude로 변경하려면 Anthropic 클라이언트를 추가 구현
```

### 7.4. 임베딩 모델 교체하기

```python
# basics.py에서 모델 변경
class SentenceEmbedder(Embedder):
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        self._model = SentenceTransformer(model_name)
```

---

## 8. 트러블슈팅

### 8.1. `ValueError: DEEPSEEK_API_KEY 환경 변수가 설정되지 않았습니다.`

```bash
# .env 파일 확인
cat .env | grep DEEPSEEK_API_KEY

# 환경 변수 직접 설정
export DEEPSEEK_API_KEY="sk-xxxxxxxxxx"

# config 모듈 확인
uv run python -c "from src.utils import config; print(bool(config.deepseek_api_key))"
```

### 8.2. `ImportError: 'sentence_transformers' module not found`

```bash
uv sync  # 의존성 재설치
uv run python -c "import sentence_transformers; print('OK')"
```

### 8.3. Mock 테스트에서 `AttributeError: '...' object has no attribute '...'`

Mock의 `spec=` 파라미터가 실제 클래스의 존재하지 않는 속성을 참조하려 할 때 발생한다. `mock_embedder`나 `mock_store` fixture에서 `spec=` 값을 확인한다.

### 8.4. 임베딩 검색 결과가 전혀 관련 없어 보일 때

- 영어 전용 모델(`all-MiniLM-L6-v2`)을 사용 중인지 확인
- 한국어 텍스트라면 다국어 모델로 교체 검토
- 문서가 너무 길면 청킹(chunking)하여 더 작은 단위로 저장

### 8.5. 테스트가 너무 느릴 때

```bash
# slow 마크가 붙은 통합 테스트 제외하고 실행
uv run pytest tests/test_llm_integration.py -v -k "not slow"  # ~3초

# 병렬 실행 (pytest-xdist 필요)
uv add --dev pytest-xdist
uv run pytest tests/ -n auto  # CPU 코어 수만큼 병렬
```

### 8.6. 개별 시나리오만 빠르게 확인

```bash
# 시나리오 1만 실행
uv run python -c "
from src.embedding_study.llm_integration import DeepSeekLLM, scenario_1_llm_only
llm = DeepSeekLLM()
r = scenario_1_llm_only(llm, '환불 정책이 뭔가요?')
print(r.answer)
"
```

---

## 부록: 테스트 명령어 치트시트

```bash
# ── 단위 테스트 (Mock) ──────────────────────────

# 전체 단위 테스트
uv run pytest tests/test_llm_integration.py -v -k "not slow"

# 특정 클래스
uv run pytest tests/test_llm_integration.py -v -k "TestKeywordSearch"

# 특정 함수
uv run pytest tests/test_llm_integration.py::TestDeepSeekLLM::test_ask_returns_string -v

# ── 통합 테스트 (실제 API) ──────────────────────

# slow 마크 포함 전체
uv run pytest tests/test_llm_integration.py -v

# slow 테스트만
uv run pytest tests/test_llm_integration.py -v -k "slow"

# ── 수동 실행 ───────────────────────────────────

# 4가지 시나리오 전체
uv run python -c "
from src.embedding_study.llm_integration import run_all_scenarios, print_comparison
print_comparison(run_all_scenarios())
"

# ── 전체 프로젝트 테스트 ─────────────────────────

# 모든 테스트 한 번에
uv run pytest tests/ -v

# slow 제외하고
uv run pytest tests/ -v -k "not slow"
```
