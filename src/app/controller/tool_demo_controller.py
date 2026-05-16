"""Controller layer: FastAPI endpoint definitions."""

from fastapi import APIRouter

from ..schema.tool_demo_schema import (
    WeatherRequest, TranslateRequest,
    CalcRequest, ChatRequest, TOOLS,
)
from ..service.tool_demo_service import exec_weather, exec_translate, exec_calculate, chat_keyword_router

router = APIRouter()


@router.get("/tools")
def list_tools():
    return {"tools": TOOLS, "count": len(TOOLS)}


@router.post("/tools/weather")
def weather(req: WeatherRequest):
    result = exec_weather(req.city)
    return {"tool": "get_weather", "input": {"city": req.city}, "result": result}


@router.post("/tools/translate")
def translate(req: TranslateRequest):
    result = exec_translate(req.text, req.target_lang)
    return {
        "tool": "translate_text",
        "input": {"text": req.text, "target_lang": req.target_lang},
        "result": result,
    }


@router.post("/tools/calculate")
def calculate(req: CalcRequest):
    result = exec_calculate(req.a, req.operator, req.b)
    return {
        "tool": "calculate",
        "input": {"a": req.a, "operator": req.operator, "b": req.b},
        "result": result,
    }


@router.post("/chat")
def chat(req: ChatRequest):
    return chat_keyword_router(req.message)
