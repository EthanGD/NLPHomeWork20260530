#!/usr/bin/env python3
"""
批量导入问答对到 RAG 系统
使用方法: python bulk_import.py [--host http://localhost:5001] [--skip-existing]
"""
import os
import sys
import time
import argparse
import requests
from typing import List, Dict

# 添加当前目录到路径，确保能导入 queryPos
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 默认配置
DEFAULT_API_BASE = "http://localhost:5001"
STORE_ENDPOINT = "/api/store"
TIMEOUT = 30  # 秒
RATE_LIMIT_DELAY = 0.1  # 请求间隔，避免过快


def load_qa_data() -> List[Dict[str, str]]:
    """从 queryPos.py 加载数据"""
    try:
        from queryPos import qa_data
        print(f"📦 Loaded {len(qa_data)} QA pairs from queryPos.py")
        return qa_data
    except ImportError as e:
        print(f"❌ Failed to import queryPos.py: {e}")
        print("💡 Make sure queryPos.py is in the same directory and defines 'qa_data' list")
        sys.exit(1)
    except AttributeError:
        print("❌ queryPos.py does not define 'qa_data' variable")
        sys.exit(1)


def store_qa_pair(api_base: str, question: str, answer: str, skip_existing: bool) -> tuple[bool, str]:
    """
    调用 /api/store 存储单个问答对
    :return: (success, message)
    """
    url = f"{api_base}{STORE_ENDPOINT}"
    payload = {
        "question": question.strip(),
        "answer": answer.strip()
    }
    
    try:
        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=TIMEOUT
        )
        
        if response.status_code == 201:
            return True, "Created"
        elif response.status_code == 200:
            result = response.json()
            if skip_existing and not result.get("updated"):
                return False, "Skipped (exists)"
            return True, "Updated" if result.get("updated") else "Exists"
        else:
            error = response.json().get("error", "Unknown error")
            return False, f"Error {response.status_code}: {error}"
            
    except requests.exceptions.RequestException as e:
        return False, f"Request failed: {str(e)}"


def bulk_import(api_base: str, skip_existing: bool = True, delay: float = 0.1) -> Dict[str, int]:
    """批量导入所有问答对"""
    qa_data = load_qa_data()
    
    stats = {
        "total": len(qa_data),
        "success": 0,
        "skipped": 0,
        "failed": 0
    }
    
    print(f"🚀 Starting bulk import to {api_base}")
    print(f"📊 Total items: {stats['total']}")
    print(f"⚙️  Skip existing: {skip_existing}")
    print("-" * 60)
    
    for i, item in enumerate(qa_data, 1):
        # 提取字段（兼容不同键名）
        question = item.get("query") or item.get("question") or item.get("q")
        answer = item.get("pos") or item.get("answer") or item.get("a")
        
        if not question or not answer:
            print(f"❌ [{i}/{stats['total']}] Skip: missing question/answer")
            stats["failed"] += 1
            continue
        
        # 截断显示
        q_preview = question[:40] + "..." if len(question) > 40 else question
        
        success, msg = store_qa_pair(api_base, question, answer, skip_existing)
        
        if success:
            if "Skipped" in msg:
                stats["skipped"] += 1
                print(f"⏭️  [{i}/{stats['total']}] {q_preview} → {msg}")
            else:
                stats["success"] += 1
                print(f"✅ [{i}/{stats['total']}] {q_preview} → {msg}")
        else:
            stats["failed"] += 1
            print(f"❌ [{i}/{stats['total']}] {q_preview} → {msg}")
        
        # 速率限制
        time.sleep(delay)
    
    return stats


def print_summary(stats: Dict[str, int], elapsed: float):
    """打印导入总结"""
    print("\n" + "=" * 60)
    print("📊 IMPORT SUMMARY")
    print("=" * 60)
    print(f"⏱️  Elapsed time: {elapsed:.2f} seconds")
    print(f"📦 Total items:   {stats['total']}")
    print(f"✅ Successfully:  {stats['success']} ({stats['success']/stats['total']*100:.1f}%)")
    print(f"⏭️  Skipped:      {stats['skipped']} ({stats['skipped']/stats['total']*100:.1f}%)")
    print(f"❌ Failed:        {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Bulk import QA pairs to RAG API")
    parser.add_argument("--host", type=str, default=DEFAULT_API_BASE,
                       help=f"API base URL (default: {DEFAULT_API_BASE})")
    parser.add_argument("--no-skip", action="store_true",
                       help="Don't skip existing questions (allow update)")
    parser.add_argument("--delay", type=float, default=RATE_LIMIT_DELAY,
                       help=f"Request delay in seconds (default: {RATE_LIMIT_DELAY})")
    
    args = parser.parse_args()
    
    skip_existing = not args.no_skip
    request_delay = args.delay
    
    start_time = time.time()
    stats = bulk_import(args.host, skip_existing, request_delay)
    elapsed = time.time() - start_time
    
    print_summary(stats, elapsed)
    
    # 返回非零退出码如果有失败
    if stats["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()