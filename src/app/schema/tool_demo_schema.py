"""Schema layer: Pydantic request models and tool definitions."""

from pydantic import BaseModel


class WeatherRequest(BaseModel):
    city: str


class TranslateRequest(BaseModel):
    text: str
    target_lang: str


class CalcRequest(BaseModel):
    a: float
    operator: str
    b: float


class ChatRequest(BaseModel):
    message: str


TOOLS = [
    {
        "name": "get_weather",
        "description": "도시의 현재 날씨를 조회합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "도시 이름"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "translate_text",
        "description": "텍스트를 다른 언어로 번역합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "원본 텍스트"},
                "target_lang": {
                    "type": "string",
                    "enum": ["ko", "en", "ja", "zh", "fr", "de", "es"],
                    "description": "대상 언어 코드",
                },
            },
            "required": ["text", "target_lang"],
        },
    },
    {
        "name": "calculate",
        "description": "두 숫자의 사칙연산을 수행합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "첫 번째 숫자"},
                "operator": {
                    "type": "string",
                    "enum": ["+", "-", "*", "/"],
                    "description": "연산자",
                },
                "b": {"type": "number", "description": "두 번째 숫자"},
            },
            "required": ["a", "operator", "b"],
        },
    },
]
