#!/usr/bin/env python3
"""
测试 /api/generate 接口
使用方法: python test_generate.py [--host http://192.168.208.2:5000]
"""
import sys
import json
import argparse
import requests
from datetime import datetime

DEFAULT_API_BASE = "http://192.168.208.2:5000"
TIMEOUT = 120  # 大模型生成可能较慢


def test_generate(api_base: str, query: str, context_k: int = 3, temperature: float = 0.1):
    """调用 /api/generate 接口"""
    url = f"{api_base}/api/generate"
    payload = {
        "query": query,
        "context_k": context_k,
        "temperature": temperature
    }
    
    print(f"🔍 Query: {query}")
    print(f"⚙️  Context: {context_k} records, temperature={temperature}")
    print(f"🌐 Endpoint: {url}")
    print("-" * 60)
    
    start_time = datetime.now()
    
    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success ({elapsed:.2f}s)\n")
            
            # 打印答案
            print("🤖 Generated Answer:")
            print("=" * 60)
            print(result.get("answer", "No answer"))
            print("=" * 60)
            
            # 打印使用的上下文
            context_used = result.get("context_used", [])
            if context_used:
                print(f"\n📚 Context Used ({len(context_used)} records):")
                for i, ctx in enumerate(context_used, 1):
                    print(f"\  [{i}] (sim={ctx['similarity']:.4f}) {ctx['question'][:60]}...")
            
            # 打印元信息
            print(f"\n📦 Model: {result.get('model')}")
            usage = result.get("usage", {})
            if usage:
                print(f"📊 Tokens: prompt={usage.get('prompt_tokens')}, "
                      f"completion={usage.get('completion_tokens')}, "
                      f"total={usage.get('total_tokens')}")
            
            return True, result
        else:
            print(f"❌ Error {response.status_code}")
            try:
                error = response.json()
                print(f"   {error}")
            except:
                print(f"   {response.text[:200]}")
            return False, None
            
    except requests.exceptions.Timeout:
        print(f"❌ Timeout after {TIMEOUT}s")
        return False, None
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection refused - Is the server running at {api_base}?")
        return False, None
    except Exception as e:
        print(f"❌ Unexpected error: {type(e).__name__}: {e}")
        return False, None


def run_batch_tests(api_base: str):
    """运行一组预设测试用例"""
    test_cases = [
        ("OpenClaw 是什麼？", 3, 0.1),
        ("如何安裝 OpenClaw？", 3, 0.1),
        ("OpenClaw裡頭的Skill有什麼用？", 5, 0.2),
        ("OpenClaw會游泳嗎？", 3, 0.1),
    ]
    
    results = []
    for query, k, temp in test_cases:
        print(f"\n{'='*60}")
        success, result = test_generate(api_base, query, k, temp)
        results.append((query, success))
        input("\n👉 Press Enter for next test...")
    
    # 打印汇总
    print(f"\n{'='*60}")
    print("📊 BATCH TEST SUMMARY")
    print("=" * 60)
    for query, success in results:
        status = "✅" if success else "❌"
        preview = query[:40] + "..." if len(query) > 40 else query
        print(f"{status} {preview}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test /api/generate endpoint")
    parser.add_argument("--host", type=str, default=DEFAULT_API_BASE,
                       help=f"API base URL (default: {DEFAULT_API_BASE})")
    parser.add_argument("--query", type=str, 
                       help="Single query to test (optional)")
    parser.add_argument("--batch", action="store_true",
                       help="Run batch of preset test queries")
    parser.add_argument("--context-k", type=int, default=3,
                       help="Number of context records (default: 3)")
    parser.add_argument("--temp", type=float, default=0.1,
                       help="Temperature for generation (default: 0.1)")
    parser.add_argument("--save", type=str,
                       help="Save response to JSON file")
    
    args = parser.parse_args()
    
    if args.batch:
        run_batch_tests(args.host)
    elif args.query:
        success, result = test_generate(
            args.host, 
            args.query, 
            args.context_k, 
            args.temp
        )
        if success and args.save and result:
            with open(args.save, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"💾 Response saved to {args.save}")
    else:
        # 交互模式
        print(f"🎯 Interactive mode - testing {args.host}/api/generate")
        print("Type 'quit' to exit, 'batch' to run preset tests\n")
        
        while True:
            query = input("\n❓ Query: ").strip()
            if query.lower() in ('quit', 'exit', 'q'):
                break
            if query.lower() == 'batch':
                run_batch_tests(args.host)
                continue
            if not query:
                continue
                
            test_generate(args.host, query, args.context_k, args.temp)
            print()


if __name__ == "__main__":
    main()