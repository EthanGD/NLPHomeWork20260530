#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 Qwen3.5 API 生成中医针灸问答对
数据源：Acupuncture/*.md
输出：qa_dataset.jsonl
"""

import os
import sys
import json
import glob
import re
import requests
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Acupuncture")
OUTPUT_FILE = os.path.join(BASE_DIR, "qa_dataset.jsonl")

API_URL = "https://llm.ethanchenyansong.com/llm/v1/chat/completions"
MODEL_NAME = "Qwen3.5"

# API 配置
API_CONFIG = {
    "model": MODEL_NAME,
    "temperature": 0.7,
    "max_tokens": 2048,
}

def load_md_files(data_dir):
    """加载所有 md 文件内容"""
    texts = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        fname = os.path.basename(fp)
        with open(fp, "r", encoding="utf-8") as f:
            content = f.read()
        texts.append({"file": fname, "content": content})
        print(f"Loaded: {fname} ({len(content)} chars)")
    return texts

def split_into_chunks(text, chunk_size=3000, overlap=500):
    """将文本分割成小块"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # 尝试在句号或换行处分割
        if end < len(text):
            last_break = max(chunk.rfind('\n'), chunk.rfind('。'), chunk.rfind('？'))
            if last_break > chunk_size * 0.7:
                chunk = chunk[:last_break]
                end = start + last_break
        chunks.append(chunk.strip())
        start = end - overlap
    return chunks

def generate_qa_from_chunk(chunk, book_name, chunk_index):
    """使用 API 从文本块生成问答对"""
    
    prompt = f"""请根据以下文本内容，生成 3-5 个高质量的中医针灸问答对。
只输出 JSON 数组格式，不要有其他文字。

文本内容：
{chunk[:3000]}

输出格式示例：
[{{"question": "问题 1", "answer": "答案 1"}}, {{"question": "问题 2", "answer": "答案 2"}}]"""

    try:
        response = requests.post(
            API_URL,
            json={
                "model": API_CONFIG["model"],
                "messages": [
                    {"role": "system", "content": "你是一名中医针灸专家，擅长生成高质量的中医知识问答对。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": API_CONFIG["temperature"],
                "max_tokens": API_CONFIG["max_tokens"],
            },
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        print(f"  Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"  Response body: {response.text[:500]}")
            return []
        
        result = response.json()
        print(f"  API result keys: {result.keys()}")
        
        if "choices" not in result or len(result["choices"]) == 0:
            print(f"  No choices in response: {result}")
            return []
        
        content = result["choices"][0]["message"]["content"]
        print(f"  Content length: {len(content)}")
        
        # 解析 JSON - 多种策略
        qa_list = []
        
        # 策略 1: 查找 ```json 代码块
        code_block_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if code_block_match:
            try:
                qa_list = json.loads(code_block_match.group(1))
                if isinstance(qa_list, list):
                    print(f"  策略 1 成功：找到 {len(qa_list)} 个问答对")
            except:
                qa_list = []
        
        # 策略 2: 查找第一个 [ 和最后一个 ]
        if not qa_list:
            json_start = content.find('[')
            json_end = content.rfind(']')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = content[json_start:json_end+1]
                try:
                    qa_list = json.loads(json_str)
                    if isinstance(qa_list, list):
                        print(f"  策略 2 成功：找到 {len(qa_list)} 个问答对")
                except:
                    qa_list = []
        
        # 策略 3: 逐行解析 JSON 对象
        if not qa_list:
            qa_list = []
            # 查找 {"question": ...} 模式
            qa_pattern = r'\{\s*"question"\s*:\s*"([^"]*)"\s*,\s*"answer"\s*:\s*"([^"]*)"\s*\}'
            matches = re.findall(qa_pattern, content, re.DOTALL)
            if matches:
                for q, a in matches:
                    qa_list.append({"question": q, "answer": a})
                print(f"  策略 3 成功：找到 {len(qa_list)} 个问答对")
        
        if qa_list and isinstance(qa_list, list):
            # 添加元数据
            for qa in qa_list:
                if "question" in qa and "answer" in qa:
                    qa["source"] = book_name
                    qa["chunk_id"] = chunk_index
            return qa_list
        
        print(f"  所有解析策略失败")
        return []
            
    except requests.exceptions.RequestException as e:
        print(f"  API request error: {e}")
        return []
    except Exception as e:
        print(f"  API error: {type(e).__name__}: {e}")
        return []

def main():
    # Set output encoding
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    print("="*60)
    print("中医针灸问答对生成")
    print("="*60)
    
    # 加载数据
    print("\n[1/3] 加载文本文件...")
    texts = load_md_files(DATA_DIR)
    total_chars = sum(len(t["content"]) for t in texts)
    print(f"总计：{len(texts)} 个文件，{total_chars:,} 字符")
    
    # 生成问答对
    print("\n[2/3] 生成问答对...")
    all_qa_pairs = []
    
    for book in texts:
        print(f"\n处理：{book['file']}")
        chunks = split_into_chunks(book["content"], chunk_size=3000, overlap=500)
        print(f"  分割成 {len(chunks)} 个文本块")
        
        for i, chunk in enumerate(chunks):
            print(f"  处理块 {i+1}/{len(chunks)}...", end=" ", flush=True)
            qa_list = generate_qa_from_chunk(chunk, book["file"], i)
            all_qa_pairs.extend(qa_list)
            print(f"生成 {len(qa_list)} 个问答对")
            
            # 每 5 个块保存一次
            if (i + 1) % 5 == 0:
                with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                    for qa in all_qa_pairs:
                        f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                print(f"  已保存 {len(all_qa_pairs)} 个问答对到 {OUTPUT_FILE}")
    
    # 保存最终结果
    print("\n[3/3] 保存结果...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for qa in all_qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    
    print(f"\n{'='*60}")
    print(f"完成！共生成 {len(all_qa_pairs)} 个问答对")
    print(f"已保存到：{OUTPUT_FILE}")
    print(f"{'='*60}")
    
    # 显示一些样本
    if all_qa_pairs:
        print("\n样本问答对:")
        for i, qa in enumerate(all_qa_pairs[:3]):
            print(f"\n{i+1}. Q: {qa['question'][:60]}...")
            print(f"   A: {qa['answer'][:100]}...")
            print(f"   来源：{qa['source']}")

if __name__ == "__main__":
    main()
