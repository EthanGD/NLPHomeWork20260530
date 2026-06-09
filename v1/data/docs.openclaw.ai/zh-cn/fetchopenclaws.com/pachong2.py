# crawl_fetchopenclaws.py
import requests
from bs4 import BeautifulSoup
import re
import json
import time
from urllib.parse import urljoin
from typing import List, Dict, Optional

# ========== 配置 ==========
BASE_URL = "https://fetchopenclaws.com"
INDEX_URL = "https://fetchopenclaws.com/answers"
OUTPUT_JSON = "fetchopenclaws_faq.json"
OUTPUT_PY = "fetchopenclaws_faq.py"
REQUEST_DELAY = 1.0  # 请求间隔（秒）
# =======================

def clean_text(text: str) -> str:
    """清理文本：移除多余空白、注释、装饰字符"""
    if not text:
        return ""
    # 移除 Vue/React 注释 <!-- -->
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # 合并连续空白（但保留段落换行）
    text = re.sub(r'[ \t]+', ' ', text)
    # 清理首尾空白
    return text.strip()

def extract_answer_part1(soup: BeautifulSoup) -> str:
    """
    提取第一部分答案：[data-ai-summary="true"]
    """
    summary_div = soup.find('div', attrs={'data-ai-summary': 'true'})
    if not summary_div:
        return ""
    
    parts = []
    # 提取所有 p 标签内容
    for p in summary_div.find_all('p', recursive=False):
        text = clean_text(p.get_text(strip=True))
        if text:
            parts.append(text)
    
    return '\n\n'.join(parts)

def extract_answer_part2(soup: BeautifulSoup) -> str:
    """
    提取第二部分答案：<div class="space-y-6"> 中的结构化内容
    包括：分步指南、示例提示词、常见陷阱、常见问题、用户反馈
    """
    content_div = soup.find('div', class_='space-y-6')
    if not content_div:
        return ""
    
    sections = []
    
    # 遍历每个 section
    for section in content_div.find_all('section', recursive=False):
        section_text = extract_section_content(section)
        if section_text:
            sections.append(section_text)
    
    return '\n\n'.join(sections)

def extract_section_content(section: BeautifulSoup) -> str:
    """
    提取单个 section 的内容，保留结构化格式
    """
    parts = []
    
    # 1. 提取标题（h2）
    title = section.find('h2')
    if title:
        # 移除标题中的 SVG 图标
        for svg in title.find_all('svg'):
            svg.decompose()
        title_text = clean_text(title.get_text(strip=True))
        if title_text:
            parts.append(f"【{title_text}】")
    
    # 2. 提取有序列表（ol > li）
    ol = section.find('ol')
    if ol:
        list_items = []
        for idx, li in enumerate(ol.find_all('li', recursive=False), 1):
            # 移除序号装饰 span
            for span in li.find_all('span', class_=lambda x: x and 'rounded-full' in x if isinstance(x, str) else False):
                if 'size-5' in span.get('class', []):
                    span.decompose()
            item_text = clean_text(li.get_text(strip=True))
            if item_text:
                list_items.append(f"{idx}. {item_text}")
        if list_items:
            parts.append('\n'.join(list_items))
    
    # 3. 提取无序列表（ul > li）
    ul = section.find('ul')
    if ul:
        list_items = []
        for li in ul.find_all('li', recursive=False):
            # 移除装饰性 SVG
            for svg in li.find_all('svg'):
                svg.decompose()
            item_text = clean_text(li.get_text(strip=True))
            if item_text:
                list_items.append(f"• {item_text}")
        if list_items:
            parts.append('\n'.join(list_items))
    
    # 4. 提取普通段落/代码块
    for elem in section.find_all(['p', 'div'], recursive=False):
        if elem.get('class') and any('rounded-xl' in c for c in elem.get('class', [])):
            text = clean_text(elem.get_text(strip=True))
            if text:
                parts.append(text)
    
    # 5. 提取用户反馈卡片
    for article in section.find_all('article'):
        feedback = []
        # 角色
        role = article.find('p', class_=lambda x: x and 'text-xs' in x and 'text-muted-foreground' in x)
        if role:
            feedback.append(f"[{clean_text(role.get_text(strip=True))}]")
        # 内容
        content = article.find('p', class_=lambda x: x and 'text-sm' in x and 'text-foreground' in x)
        if content:
            feedback.append(clean_text(content.get_text(strip=True)))
        if feedback:
            parts.append(' '.join(feedback))
    
    return '\n\n'.join(p for p in parts if p)

def extract_query_from_detail(soup: BeautifulSoup) -> Optional[str]:
    """从详情页提取问题（h1 标签）"""
    h1 = soup.find('h1')
    if h1:
        return clean_text(h1.get_text(strip=True))
    return None

def get_question_links(index_url: str) -> List[Dict[str, str]]:
    """
    从索引页提取所有问题链接和标题
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept-Language': 'zh-CN,zh;q=0.9',
    }
    
    questions = []
    
    try:
        print(f"📥 请求索引页: {index_url}")
        response = requests.get(index_url, headers=headers, timeout=30)
        response.raise_for_status()
        response.encoding = 'utf-8'
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 查找所有包含 h2 问题的 <a> 标签
        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href', '')
            # 筛选 /answers/xxx 格式的链接
            if not href.startswith('/answers/') or href == '/answers':
                continue
            
            h2 = a_tag.find('h2')
            if h2:
                query = clean_text(h2.get_text(strip=True))
                if query:
                    full_url = urljoin(BASE_URL, href)
                    questions.append({
                        "query": query,
                        "url": full_url,
                        "slug": href
                    })
                    print(f"✅ 发现: {query[:40]}...")
        
        print(f"\n🔍 共找到 {len(questions)} 个问题链接")
        return questions
        
    except Exception as e:
        print(f"❌ 获取索引页失败: {e}")
        return []

def crawl_detail_page(url: str) -> Optional[Dict[str, str]]:
    """
    爬取单个详情页，返回问答对
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept-Language': 'zh-CN,zh;q=0.9',
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        response.encoding = 'utf-8'
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 提取问题（用于验证）
        query = extract_query_from_detail(soup)
        if not query:
            return None
        
        # 提取两部分答案
        part1 = extract_answer_part1(soup)
        part2 = extract_answer_part2(soup)
        
        # 合并答案
        pos_parts = []
        if part1:
            pos_parts.append(part1)
        if part2:
            pos_parts.append(part2)
        
        pos = '\n\n'.join(pos_parts)
        
        if not pos:
            print(f"⚠️  无答案内容: {query[:40]}...")
            return None
        
        return {
            "query": query,
            "pos": pos
        }
        
    except Exception as e:
        print(f"❌ 爬取 {url} 失败: {e}")
        return None

def crawl_all_faq(index_url: str) -> List[Dict[str, str]]:
    """
    爬取所有问答对
    """
    questions = get_question_links(index_url)
    if not questions:
        return []
    
    qa_pairs = []
    
    for idx, q in enumerate(questions, 1):
        print(f"\n[{idx}/{len(questions)}] 爬取: {q['query'][:30]}...")
        
        result = crawl_detail_page(q['url'])
        if result:
            qa_pairs.append(result)
            print(f"✅ 成功: {result['query'][:30]}...")
        else:
            print(f"❌ 失败: {q['query'][:30]}...")
        
        # 礼貌延迟
        if idx < len(questions):
            time.sleep(REQUEST_DELAY)
    
    print(f"\n🎉 采集完成！有效问答对: {len(qa_pairs)}/{len(questions)}")
    return qa_pairs

def save_json(data: List[Dict], filepath: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 JSON: {filepath}")

def save_python(data: List[Dict], filepath: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# -*- coding: utf-8 -*-\nqa_data = [\n")
        for item in data:
            # 转义三重引号
            q = item['query'].replace('"""', '"\'"')
            p = item['pos'].replace('"""', '"\'"')
            f.write("    {\n")
            f.write(f'        "query": """{q}""",\n')
            f.write(f'        "pos": """{p}""",\n')
            f.write("    },\n")
        f.write("]\n")
    print(f"💾 Python: {filepath}")

def save_txt(data: List[Dict], filepath: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(data, 1):
            f.write(f"【{idx}】❓ {item['query']}\n")
            f.write(f"💡 {item['pos']}\n")
            f.write("─" * 70 + "\n\n")
    print(f"💾 TXT: {filepath}")

# ========== 主入口 ==========
if __name__ == "__main__":
    print("🚀 开始爬取 FetchOpenClaws FAQ...")
    
    qa_data = crawl_all_faq(INDEX_URL)
    
    if qa_data:
        save_json(qa_data, OUTPUT_JSON)
        save_python(qa_data, OUTPUT_PY)
        save_txt(qa_data, "fetchopenclaws_faq.txt")
        
        # 预览
        print("\n📋 数据预览（前2条）:")
        for i, item in enumerate(qa_data[:2], 1):
            print(f"\n[{i}] ❓ {item['query']}")
            print(f"    💡 {item['pos'][:200]}...")
    else:
        print("⚠️  未采集到数据，请检查网络连接或网页结构")