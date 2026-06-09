# crawl_openclaw_faq_h3.py
import requests
from bs4 import BeautifulSoup, Tag
import re
import json
from urllib.parse import urljoin
from typing import List, Dict, Optional

# 基础配置
BASE_URL = "https://docs.openclaw.ai"
TARGET_URL = "https://docs.openclaw.ai/zh-CN/help/faq"
OUTPUT_JSON = "openclaw_faq_zh.json"
OUTPUT_PY = "openclaw_faq_zh.py"

def extract_answer_content(start_elem: Tag, end_elem: Optional[Tag]) -> str:
    """
    提取两个 h3 标签之间的答案内容
    支持: span[data-as], ul/li, p, a, code, strong 等
    """
    if not start_elem:
        return ""
    
    parts = []
    current = start_elem.next_sibling
    
    while current and current != end_elem:
        if isinstance(current, Tag):
            # 跳过非内容标签
            if current.name in ['script', 'style', 'nav', 'header', 'footer']:
                current = current.next_sibling
                continue
            
            # 处理不同标签类型
            if current.name == 'span' and current.get('data-as') == 'p':
                text = current.get_text(strip=True)
                if text:
                    parts.append(text)
            
            elif current.name == 'ul':
                # 处理列表
                list_items = []
                for li in current.find_all('li', recursive=False):
                    item_text = clean_element_text(li)
                    if item_text:
                        list_items.append(f"• {item_text}")
                if list_items:
                    parts.append('\n'.join(list_items))
            
            elif current.name == 'p':
                text = clean_element_text(current)
                if text:
                    parts.append(text)
            
            elif current.name == 'a':
                # 链接转为 [文本](完整URL) 格式
                link_text = current.get_text(strip=True)
                href = current.get('href', '')
                if href:
                    full_url = urljoin(BASE_URL, href) if not href.startswith('http') else href
                    parts.append(f"{link_text}({full_url})")
                elif link_text:
                    parts.append(link_text)
            
            elif current.name == 'code':
                # 代码块加反引号
                code_text = current.get_text(strip=True)
                if code_text:
                    parts.append(f"`{code_text}`")
            
            elif current.name == 'strong':
                # 加粗用 ** 标记
                bold_text = current.get_text(strip=True)
                if bold_text:
                    parts.append(f"**{bold_text}**")
            
            else:
                # 其他标签递归提取文本
                text = current.get_text(separator=' ', strip=True)
                if text:
                    parts.append(text)
        
        elif isinstance(current, str) and current.strip():
            # 纯文本节点
            parts.append(current.strip())
        
        current = current.next_sibling
    
    # 合并段落，保留换行
    result = '\n\n'.join(p for p in parts if p.strip())
    # 清理多余空白但保留必要换行
    result = re.sub(r'[ \t]+', ' ', result)
    return result.strip()

def clean_element_text(elem: Tag) -> str:
    """
    清理元素内的文本，处理嵌套的 a/code/strong 等标签
    """
    texts = []
    for child in elem.children:
        if isinstance(child, str):
            text = child.strip()
            if text:
                texts.append(text)
        elif child.name == 'a':
            link_text = child.get_text(strip=True)
            href = child.get('href', '')
            if href:
                full_url = urljoin(BASE_URL, href) if not href.startswith('http') else href
                texts.append(f"{link_text}({full_url})")
            elif link_text:
                texts.append(link_text)
        elif child.name == 'code':
            code_text = child.get_text(strip=True)
            if code_text:
                texts.append(f"`{code_text}`")
        elif child.name == 'strong':
            bold_text = child.get_text(strip=True)
            if bold_text:
                texts.append(f"**{bold_text}**")
        else:
            sub_text = child.get_text(strip=True)
            if sub_text:
                texts.append(sub_text)
    return ' '.join(texts)

def extract_query_from_h3(h3: Tag) -> Optional[str]:
    """
    从 h3 标签中提取问题文本
    结构: <h3>...<span class="cursor-pointer">问题文本</span>...</h3>
    """
    span = h3.find('span', class_='cursor-pointer')
    if span:
        return span.get_text(strip=True)
    # 备用：直接取 h3 文本（去除装饰内容）
    return h3.get_text(strip=True)

def crawl_faq_h3_structure(url: str) -> List[Dict[str, str]]:
    """
    爬取采用 h3 结构的 FAQ 页面
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }
    
    try:
        print(f"📥 请求: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        response.encoding = 'utf-8'
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 查找所有作为问题的 h3 标签
        h3_list = soup.find_all('h3', class_=lambda x: x and 'font-semibold' in x if isinstance(x, str) else False)
        # 更精准的查找：通过 id 属性或内部 span.cursor-pointer
        h3_list = [h3 for h3 in h3_list if h3.find('span', class_='cursor-pointer') or h3.get('id')]
        
        if not h3_list:
            # 备用选择器
            h3_list = soup.find_all('h3', id=True)
        
        print(f"🔍 找到 {len(h3_list)} 个 h3 问题标签")
        
        qa_pairs = []
        
        for i, h3 in enumerate(h3_list):
            query = extract_query_from_h3(h3)
            if not query:
                continue
            
            # 答案范围：当前 h3 的下一个兄弟元素 到 下一个 h3 之前
            next_h3 = h3_list[i + 1] if i + 1 < len(h3_list) else None
            pos = extract_answer_content(h3, next_h3)
            
            if pos:
                qa_pair = {
                    "query": query,
                    "pos": pos
                }
                qa_pairs.append(qa_pair)
                preview = pos[:80].replace('\n', ' ')
                print(f"✅ [{i+1}] {query} → {preview}...")
            else:
                print(f"⚠️  [{i+1}] 无答案内容: {query}")
        
        print(f"\n🎉 采集完成！有效问答对: {len(qa_pairs)}/{len(h3_list)}")
        return qa_pairs
        
    except requests.RequestException as e:
        print(f"❌ 网络错误: {e}")
        return []
    except Exception as e:
        print(f"❌ 解析错误: {e}")
        import traceback
        traceback.print_exc()
        return []

def save_json(data: List[Dict], filepath: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 JSON: {filepath}")

def save_python(data: List[Dict], filepath: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# -*- coding: utf-8 -*-\nqa_data = [\n")
        for item in data:
            # 转义三重引号避免语法错误
            q = item['query'].replace('"""', '"\'"')
            p = item['pos'].replace('"""', '"\'"')
            f.write(f'    {{"query": """{q}""", "pos": """{p}"""}},\n')
        f.write("]\n")
    print(f"💾 Python: {filepath}")

def save_txt(data: List[Dict], filepath: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(data, 1):
            f.write(f"【{idx}】问：{item['query']}\n")
            f.write(f"答：{item['pos']}\n")
            f.write("─" * 60 + "\n\n")
    print(f"💾 TXT: {filepath}")

# ========== 主入口 ==========
if __name__ == "__main__":
    qa_data = crawl_faq_h3_structure(TARGET_URL)
    
    if qa_data:
        save_json(qa_data, OUTPUT_JSON)
        save_python(qa_data, OUTPUT_PY)
        save_txt(qa_data, "openclaw_faq_zh.txt")
        
        # 预览
        print("\n📋 预览前 2 条:")
        for item in qa_data[:2]:
            print(f"\n❓ {item['query']}")
            print(f"💡 {item['pos'][:150]}...")
    else:
        print("⚠️  未采集到数据，请检查网页结构")