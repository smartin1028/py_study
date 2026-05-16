"""
FastAPI Tool Calling 데모
=========================
tool_runner.py 의 도구들을 REST API 엔드포인트로 제공합니다.

실행: uv run uvicorn src.app.tool_demo_main:app --reload
"""

from fastapi import FastAPI

from .controller.tool_demo_controller import router

app = FastAPI(title="Tool Calling Demo")
app.include_router(router)
