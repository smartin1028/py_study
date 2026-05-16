"""
Claude API Tool Calling 데모
============================
이 스크립트는 Anthropic API의 tool calling 기능을 직접 구현하는 방법을 보여줍니다.

실행 방법:
  export ANTHROPIC_API_KEY="your-api-key"
  python src/app/script/tool_demo_claude.py
"""

import json
import sys

from anthropic import Anthropic

from ..service.tool_demo_service import exec_weather, exec_translate
from src.utils import config


TOOLS = [
    {
        "name": "get_current_weather",
        "description": (
            "지정된 도시의 현재 날씨 정보를 반환합니다. "
            "도시 이름은 한글 또는 영문 모두 지원합니다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "날씨를 조회할 도시 이름 (예: '서울', 'Tokyo', 'New York')",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "온도 단위 (celsius 또는 fahrenheit). 기본값은 celsius.",
                },
            },
            "required": ["city"],
        },
    },
    {
        "name": "text_translator",
        "description": (
            "주어진 텍스트를 지정된 언어로 번역합니다. "
            "한국어, 영어, 일본어, 중국어, 프랑스어, 독일어, 스페인어를 지원합니다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "번역할 원본 텍스트",
                },
                "target_language": {
                    "type": "string",
                    "enum": ["ko", "en", "ja", "zh", "fr", "de", "es"],
                    "description": (
                        "번역 결과 언어 코드. "
                        "ko=한국어, en=영어, ja=일본어, zh=중국어, "
                        "fr=프랑스어, de=독일어, es=스페인어"
                    ),
                },
            },
            "required": ["text", "target_language"],
        },
    },
]

CITY_MAP = {
    "seoul": "서울", "busan": "부산", "jeju": "제주",
    "tokyo": "도쿄", "new york": "뉴욕", "london": "런던",
    "paris": "파리", "sydney": "시드니",
}


def handle_get_current_weather(city: str, unit: str = "celsius") -> dict:
    """공통 exec_weather를 호출하고 fahrenheit 변환 및 반환 키를 적용."""
    mapped_city = CITY_MAP.get(city.strip().lower(), city.strip())
    canonical = exec_weather(mapped_city)

    condition = canonical["날씨"]
    temp_str = canonical["기온"]
    humidity_str = canonical["습도"]

    temp_c = int(temp_str.replace("°C", ""))
    humidity = int(humidity_str.replace("%", ""))

    temperature = round(temp_c * 9 / 5 + 32, 1) if unit == "fahrenheit" else temp_c

    return {
        "city": city,
        "unit": unit,
        "condition": condition,
        "temperature": temperature,
        "humidity": humidity,
    }


def handle_text_translator(text: str, target_language: str) -> dict:
    """공통 exec_translate를 호출하고 반환 키를 claude_tool_demo 형식으로 변환."""
    canonical = exec_translate(text, target_language)
    return {
        "original_text": canonical["original"],
        "target_language": canonical["target_lang"],
        "translated_text": canonical["translated"],
    }


TOOL_HANDLERS = {
    "get_current_weather": handle_get_current_weather,
    "text_translator": handle_text_translator,
}


def chat_with_claude(user_message: str, model: str = "claude-sonnet-4-6"):
    client = Anthropic(api_key=config.anthropic_api_key)
    messages = [{"role": "user", "content": user_message}]

    print(f"\n{'='*60}")
    print(f"🙋 사용자: {user_message}")
    print(f"{'='*60}")

    while True:
        print("\n📡 Claude API 호출 중...")
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system="당신은 도구를 활용하여 사용자를 돕는 AI 비서입니다. "
                   "필요한 경우 제공된 도구를 사용하세요.",
            messages=messages,
            tools=TOOLS,
        )

        tool_use_blocks = []
        text_parts = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_use_blocks.append(block)
                print(f"\n🔧 Claude가 도구 호출을 요청했습니다: {block.name}")
                print(f"   입력값: {json.dumps(block.input, ensure_ascii=False)}")

        if text_parts:
            combined = " ".join(text_parts)
            print(f"\n🤖 Claude: {combined}")

        if not tool_use_blocks:
            return response

        tool_results = []
        for tool_block in tool_use_blocks:
            handler = TOOL_HANDLERS.get(tool_block.name)
            if handler is None:
                result = {"error": f"알 수 없는 도구: {tool_block.name}"}
            else:
                try:
                    result = handler(**tool_block.input)
                    print(f"   ✅ 실행 결과: {json.dumps(result, ensure_ascii=False)}")
                except Exception as e:
                    result = {"error": str(e)}
                    print(f"   ❌ 실행 오류: {e}")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_block.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

        messages.append({
            "role": "assistant",
            "content": [b.model_dump() for b in response.content],
        })
        messages.append({
            "role": "user",
            "content": tool_results,
        })


def main():
    if not config.anthropic_api_key:
        print("❌ 환경변수 ANTHROPIC_API_KEY가 설정되지 않았습니다.")
        print("   export ANTHROPIC_API_KEY='your-api-key-here'")
        sys.exit(1)

    model = config.anthropic_model

    print("╔══════════════════════════════════════════════════════╗")
    print("║       Claude API Tool Calling 데모                    ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  사용 가능한 도구:                                    ║")
    print("║    🌤️  get_current_weather - 도시 날씨 조회           ║")
    print("║    🌐  text_translator    - 텍스트 번역               ║")
    print("║                                                      ║")
    print("║  사용 예시 질문:                                      ║")
    print("║    - 서울 날씨 어때?                                   ║")
    print("║    - 부산 날씨를 화씨로 알려줘                          ║")
    print("║    - Hello를 한국어로 번역해줘                          ║")
    print("║    - '안녕하세요'를 일본어, 중국어로 번역해줘             ║")
    print("║    - 종료하려면 'exit' 또는 'quit' 입력                 ║")
    print("╚══════════════════════════════════════════════════════╝")

    while True:
        try:
            user_input = input("\n📝 질문을 입력하세요 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 종료합니다.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("👋 종료합니다.")
            break

        try:
            response = chat_with_claude(user_input, model=model)
            print(f"\n{'='*60}")
            print("✅ 대화 완료")
            print(f"   토큰 사용량 - 입력: {response.usage.input_tokens}, "
                  f"출력: {response.usage.output_tokens}")
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")


if __name__ == "__main__":
    main()
