"""
로컬 Tool Calling 데모
======================
순수 Python만으로 tool calling 구조를 체험하는 스크립트입니다.
API 키 없이 바로 실행됩니다.

실행 방법:
  python src/app/script/tool_demo_runner.py
"""

import json

from ..service.tool_demo_service import exec_weather, exec_translate, exec_calculate, format_response


TOOLS = {
    "1": {
        "name": "get_weather",
        "description": "도시의 현재 날씨를 조회",
        "parameters": ["city"],
    },
    "2": {
        "name": "translate_text",
        "description": "텍스트를 다른 언어로 번역",
        "parameters": ["text", "target_lang"],
    },
    "3": {
        "name": "calculate",
        "description": "두 숫자의 사칙연산",
        "parameters": ["a", "operator", "b"],
    },
}

HANDLERS = {
    "get_weather": exec_weather,
    "translate_text": exec_translate,
    "calculate": exec_calculate,
}


def run_tool(tool_id: str):
    tool = TOOLS[tool_id]

    print(f"\n{'─'*50}")
    print(f"📌 선택된 도구: {tool['name']}")
    print(f"   설명: {tool['description']}")
    print(f"{'─'*50}")

    params = {}
    for param in tool["parameters"]:
        value = input(f"   ↳ {param}? ").strip()
        if not value:
            print("   ⚠️ 값을 입력하지 않아 기본값 사용")
            value = "기본값"
        params[param] = value

    if tool["name"] == "calculate":
        try:
            params["a"] = float(params["a"])
            params["b"] = float(params["b"])
        except ValueError:
            print(f"   ❌ 숫자가 아닙니다: a={params['a']}, b={params['b']}")
            return

    print(f"\n🔧 [{tool['name']}] 실행 중...")
    print(f"   입력: {json.dumps(params, ensure_ascii=False)}")

    handler = HANDLERS[tool["name"]]
    result = handler(**params)

    print(f"   결과: {json.dumps(result, ensure_ascii=False)}")

    print(f"\n🤖 응답:")
    if "error" in result:
        print(f"   ❌ {result['error']}")
    else:
        print(format_response(tool["name"], result))
    print(f"{'─'*50}")


def main():
    print("╔══════════════════════════════════════╗")
    print("║   🔧 Tool Calling 데모 (로컬 모드)    ║")
    print("╠══════════════════════════════════════╣")
    print("║  사용 가능한 도구:                    ║")
    for num, tool in TOOLS.items():
        print(f"║    [{num}] {tool['name']}: {tool['description']}  ║")
    print("║    [q] 종료                           ║")
    print("╚══════════════════════════════════════╝")

    while True:
        print(f"\n{'='*40}")
        choice = input("🖐  도구 번호 선택 (1/2/3, q=종료) > ").strip()

        if choice.lower() in ("q", "quit", "exit"):
            print("👋 종료합니다.")
            break
        if choice not in TOOLS:
            print(f"⚠️ 1~3 사이의 번호를 입력하거나 q로 종료하세요.")
            continue

        run_tool(choice)


if __name__ == "__main__":
    main()
