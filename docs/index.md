# pyTool 프로젝트 문서

## 개요

개인 학습용 프로젝트로, FastAPI + Claude API 도구 호출 데모와 Pandas 데이터 분석 스터디를 포함합니다.

## 기술 스택

- **Python** >= 3.13
- **FastAPI** + **Uvicorn** — Web API 서버
- **Anthropic Claude API** — AI 도구 호출 연동
- **Pandas 3.0** — 데이터 분석
- **uv** — 패키지 매니저

## 프로젝트 구조

```
src/
├── app/                   # 애플리케이션 패키지 (계층형 구조)
│   ├── tool_demo_main.py  # FastAPI 진입점
│   ├── controller/        # API 엔드포인트
│   │   └── tool_demo_controller.py
│   ├── service/           # 비즈니스 로직
│   │   └── tool_demo_service.py
│   ├── repository/        # 데이터 액세스
│   │   └── tool_demo_repository.py
│   ├── schema/            # Pydantic 모델
│   │   └── tool_demo_schema.py
│   └── script/            # 스크립트/데모
│       ├── tool_demo_runner.py
│       └── tool_demo_claude.py
│
└── pandas_study/          # Pandas 100 예제 학습 패키지
    ├── _interface.py      # ABC / Protocol 정의
    ├── io_examples.py     # 파일 I/O
    ├── core.py            # 기본 조작
    ├── aggregation.py     # 집계
    ├── combine.py         # 병합/결합
    ├── cleaning.py        # 전처리
    └── advanced.py        # 고급 기능
```

### 파일 네이밍 규칙
- `{업무}_{계층}.py` (예: `tool_demo_service.py`, `report_service.py`)

## 빠른 시작

```bash
# 의존성 설치
uv sync

# FastAPI 서버 실행
uv run uvicorn src.app.tool_demo_main:app --reload

# 테스트 실행
uv run pytest
```

## 라이선스

MIT License — Copyright (c) 2026 EumGrowth
