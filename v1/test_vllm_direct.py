#!/usr/bin/env python3
"""
直接调用 vLLM OpenAI 兼容接口测试
配置:
  - API Base: https://llm.ethanchenyansong.com/llm/v1
  - Model: Qwen3.5
  - 支持 thinking 模式开关

使用方法:
  python test_vllm_direct.py [--query "问题"] [--batch] [--think]
"""
import sys
import json
import time
import argparse
import requests
from datetime import datetime
from typing import List, Optional

# ============ 配置 ============
VLLM_API_BASE = "https://llm.ethanchenyansong.com/llm/v1"
VLLM_MODEL_NAME = "Qwen3.5"  # 可通过 --model 覆盖
TIMEOUT = 120  # 秒

# 测试用例（用户指定）
TEST_CASES = [
    ("OpenClaw 是什麼？", 0.1),
    ("如何安裝 OpenClaw？", 0.1),
    ("OpenClaw 裡頭的 Skill 有什麼用？", 0.2),
    ("OpenClaw 會游泳嗎？", 0.1),
]


def call_vllm(
    query: str,
    temperature: float = 0.1,
    enable_thinking: bool = False,
    model: str = VLLM_MODEL_NAME,
    api_base: str = VLLM_API_BASE,
    max_tokens: int = 2048
) -> tuple[bool, Optional[dict], Optional[str]]:
    """
    直接调用 vLLM /chat/completions 接口
    :return: (success, response_json, error_message)
    """
    url = f"{api_base}/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        # 如需 API Key 可添加:
        # "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": query}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "stream": False
    }
    
    try:
        start = time.time()
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=TIMEOUT
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            return True, result, None
        else:
            error_msg = f"HTTP {response.status_code}"
            try:
                error = response.json()
                error_msg += f": {error.get('error', {})}"
            except:
                error_msg += f": {response.text[:200]}"
            return False, None, error_msg
            
    except requests.exceptions.Timeout:
        return False, None, f"Timeout after {TIMEOUT}s"
    except requests.exceptions.ConnectionError:
        return False, None, f"Connection refused to {api_base}"
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def print_response(result: dict, elapsed: float, query: str):
    """格式化打印响应"""
    print(f"\n⏱️  Response time: {elapsed:.2f}s")
    print(f"🤖 Model: {result.get('model')}")
    
    # 打印答案
    answer = result["choices"][0]["message"]["content"]
    print(f"\n❓ Query: {query}")
    print(f"\n💬 Answer:")
    print("=" * 70)
    print(answer)
    print("=" * 70)
    
    # 打印 token 使用
    usage = result.get("usage", {})
    if usage:
        print(f"\n📊 Tokens:")
        print(f"   Prompt:     {usage.get('prompt_tokens')}")
        print(f"   Completion: {usage.get('completion_tokens')}")
        print(f"   Total:      {usage.get('total_tokens')}")


def run_single_test(query: str, temperature: float, enable_thinking: bool, model: str):
    """运行单次测试"""
    print(f"\n{'='*70}")
    print(f"🔍 Testing: {query[:50]}{'...' if len(query)>50 else ''}")
    print(f"⚙️  Params: temp={temperature}, thinking={enable_thinking}, model={model}")
    
    success, result, error = call_vllm(
        query=query,
        temperature=temperature,
        enable_thinking=enable_thinking,
        model=model
    )
    
    if success and result:
        elapsed = result.get("elapsed", 0)  # 如果需要可手动计算
        print_response(result, elapsed, query)
        return True
    else:
        print(f"❌ Error: {error}")
        return False


def run_batch_tests(enable_thinking: bool, model: str):
    """批量运行预设测试用例"""
    print(f"🚀 Batch testing vLLM API: {VLLM_API_BASE}")
    print(f"📦 Model: {model}")
    print(f"🧠 Thinking mode: {enable_thinking}")
    print("=" * 70)
    
    results = []
    for i, (query, temp) in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}]", end=" ")
        success = run_single_test(query, temp, enable_thinking, model)
        results.append((query, success))
        
        if i < len(TEST_CASES):
            input("\n👉 Press Enter for next test...")
    
    # 汇总
    print(f"\n{'='*70}")
    print("📊 BATCH SUMMARY")
    print("=" * 70)
    for query, success in results:
        status = "✅" if success else "❌"
        preview = query[:45] + "..." if len(query) > 45 else query
        print(f"{status} {preview}")
    print("=" * 70)


def interactive_mode(model: str):
    """交互模式：持续输入问题测试"""
    print(f"🎯 Interactive mode - vLLM @ {VLLM_API_BASE}")
    print(f"📦 Model: {model}")
    print("Commands: 'quit' to exit, 'think on/off' to toggle thinking\n")
    
    enable_thinking = False
    
    while True:
        user_input = input("\n❓ Query: ").strip()
        
        if user_input.lower() in ('quit', 'exit', 'q', ''):
            break
        
        # 命令处理
        if user_input.lower().startswith('think '):
            cmd = user_input.split()[1]
            enable_thinking = cmd.lower() in ('on', 'true', '1')
            print(f"🧠 Thinking mode: {'ON' if enable_thinking else 'OFF'}")
            continue
        
        if user_input.lower() == 'batch':
            run_batch_tests(enable_thinking, model)
            continue
        
        # 执行测试
        run_single_test(
            query=user_input,
            temperature=0.1,
            enable_thinking=enable_thinking,
            model=model
        )


def main():
    parser = argparse.ArgumentParser(description="Direct vLLM API tester")
    parser.add_argument("--query", "-q", type=str, help="Single query to test")
    parser.add_argument("--batch", "-b", action="store_true", help="Run batch test cases")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode (default)")
    parser.add_argument("--think", action="store_true", help="Enable thinking mode (Qwen3.5)")
    parser.add_argument("--no-think", action="store_true", help="Disable thinking mode")
    parser.add_argument("--temp", type=float, default=0.1, help="Temperature (default: 0.1)")
    parser.add_argument("--model", type=str, default=VLLM_MODEL_NAME, help=f"Model name (default: {VLLM_MODEL_NAME})")
    parser.add_argument("--api", type=str, default=VLLM_API_BASE, help=f"API base URL")
    parser.add_argument("--save", type=str, help="Save response to JSON file")
    
    args = parser.parse_args()
    
    # 确定 thinking 模式
    if args.no_think:
        enable_thinking = False
    elif args.think:
        enable_thinking = True
    else:
        enable_thinking = False  # 默认关闭
    
    # 执行模式
    if args.batch:
        run_batch_tests(enable_thinking, args.model)
    elif args.query:
        success, result, error = call_vllm(
            query=args.query,
            temperature=args.temp,
            enable_thinking=enable_thinking,
            model=args.model,
            api_base=args.api
        )
        if success and result:
            print_response(result, 0, args.query)
            if args.save:
                with open(args.save, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"💾 Saved to {args.save}")
        else:
            print(f"❌ Error: {error}")
            sys.exit(1)
    else:
        # 默认交互模式
        interactive_mode(args.model)


if __name__ == "__main__":
    main()