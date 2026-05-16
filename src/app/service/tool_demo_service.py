"""Service layer: tool execution and business logic."""

import json
import re

from ..repository.tool_demo_repository import WEATHER_DB, TRANSLATION_DB


# ---- Tool Handlers ----


def exec_weather(city: str) -> dict:
    """Return mock weather data for the given city."""
    w = WEATHER_DB.get(city.strip(), {"날씨": "정보 없음", "기온": "N/A", "습도": "N/A"})
    return {"city": city, **w}


def exec_translate(text: str, target_lang: str) -> dict:
    """Return mock translation for the given text and language."""
    key = (text.strip().lower(), target_lang)
    translated = TRANSLATION_DB.get(key, f"[{text}]의 {target_lang} 번역 결과")
    return {"original": text, "target_lang": target_lang, "translated": translated}


def exec_calculate(a: float, operator: str, b: float) -> dict:
    """Perform basic arithmetic operations."""
    ops = {
        "+": a + b,
        "-": a - b,
        "*": a * b,
        "/": a / b if b != 0 else "0으로 나눌 수 없음",
    }
    result = ops.get(operator, f"지원하지 않는 연산자: {operator}")
    return {"expression": f"{a} {operator} {b}", "result": result}


# ---- Chat Logic ----


def chat_keyword_router(message: str) -> dict:
    """Route natural language messages to the appropriate tool via keyword matching."""
    msg = message

    if re.search(r"날씨|기온|weather", msg):
        for city in WEATHER_DB:
            if city in msg:
                result = exec_weather(city)
                return {
                    "tool_called": "get_weather",
                    "thought": "사용자가 날씨를 물어봤으므로 get_weather 도구를 호출합니다.",
                    "input": {"city": city},
                    "result": result,
                    "response": f"{city}의 현재 날씨는 '{result['날씨']}', 기온 {result['기온']}, 습도 {result['습도']}입니다.",
                }
        return {"error": "메시지에서 도시 이름을 찾을 수 없습니다."}

    elif re.search(r"번역|translate", msg):
        return {
            "tool_called": "translate_text",
            "thought": "사용자가 번역을 요청했으므로 translate_text 도구를 호출합니다.",
            "hint": "번역할 텍스트와 대상 언어를 /tools/translate 에 직접 보내주세요.",
        }

    elif re.search(r"계산|더하기|빼기|곱하기|나누기|\+|\-|\*|\/", msg):
        return {
            "tool_called": "calculate",
            "thought": "사용자가 계산을 요청했으므로 calculate 도구를 호출합니다.",
            "hint": "계산식을 /tools/calculate 에 직접 보내주세요.",
        }

    return {
        "tool_called": None,
        "thought": "요청에 적합한 도구가 없습니다. 일반 질문으로 답변합니다.",
        "response": f"'{msg}'에 대해 도구 호출 없이 직접 응답합니다. (도구 필요 시 날씨/번역/계산을 말씀해주세요)",
    }


def format_response(tool_name: str, result: dict) -> str:
    """Format a tool result into a human-readable sentence."""
    if tool_name == "get_weather":
        return (f"   {result['city']}의 현재 날씨는 '{result['날씨']}', "
                f"기온 {result['기온']}, 습도 {result['습도']}입니다.")
    elif tool_name == "translate_text":
        return f"   '{result['original']}' → '{result['translated']}'"
    elif tool_name == "calculate":
        return f"   계산 결과: {result['expression']} = {result['result']}"
    return json.dumps(result, ensure_ascii=False)
