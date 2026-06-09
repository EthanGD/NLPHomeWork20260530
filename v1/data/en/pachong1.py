# crawl_openclaw_faq.py
import requests
from bs4 import BeautifulSoup
import re
import json
from typing import List, Dict

def extract_text(element) -> str:
    """
    提取元素文本，保留段落结构，移除多余空白
    """
    if not element:
        return ""
    
    # 获取所有文本节点
    texts = []
    for child in element.children:
        if child.name is None:  # 文本节点
            text = child.strip()
            if text:
                texts.append(text)
        elif child.name == 'a':  # 链接：只保留链接文本
            link_text = child.get_text(strip=True)
            if link_text:
                texts.append(link_text)
        else:  # 其他标签：递归提取
            sub_text = child.get_text(separator=' ', strip=True)
            if sub_text:
                texts.append(sub_text)
    
    # 用换行符连接不同段落，空格连接同段落
    result = '\n'.join(texts)
    # 清理多余空白
    result = re.sub(r'\s+', ' ', result).strip()
    return result

def crawl_openclaw_faq(url: str) -> List[Dict[str, str]]:
    """
    爬取 OpenClaw FAQ 页面的问答对
    
    Args:
        url: 目标页面链接
        
    Returns:
        List[Dict]: 包含 query 和 pos 的字典列表
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        # 1. 获取网页内容
        print(f"📥 请求页面: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        response.encoding = response.apparent_encoding  # 自动检测编码
        
        # 2. 解析 HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 3. 查找所有问答对
        qa_pairs = []
        
        # 查找所有问题标题
        questions = soup.find_all('p', attrs={'data-component-part': 'accordion-title'})
        
        for question in questions:
            query = question.get_text(strip=True)
            if not query:
                continue
            
            # 查找对应的答案内容
            # 答案通常在问题元素的下一个兄弟元素中
            answer_div = question.find_next('div', attrs={'data-component-part': 'accordion-content'})
            
            if answer_div:
                pos = extract_text(answer_div)
            else:
                # 备用方案：尝试通过 aria-labelledby 关联
                question_id = question.get('id') or query.lower().replace(' ', '-').replace('?', '')
                answer_div = soup.find('div', attrs={'aria-labelledby': question_id})
                pos = extract_text(answer_div) if answer_div else ""
            
            # 只添加有答案的问答对
            if pos:
                qa_pair = {
                    "query": query,
                    "pos": pos
                }
                qa_pairs.append(qa_pair)
                print(f"✅ 采集: {query[:50]}...")
        
        print(f"\n🎉 采集完成！共获取 {len(qa_pairs)} 个问答对")
        return qa_pairs
        
    except requests.RequestException as e:
        print(f"❌ 网络请求错误: {e}")
        return []
    except Exception as e:
        print(f"❌ 解析错误: {e}")
        return []

def save_to_json(data: List[Dict], filepath: str):
    """保存为 JSON 文件"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 已保存至: {filepath}")

def save_to_python(data: List[Dict], filepath: str):
    """保存为 Python 列表格式（方便直接导入）"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("qa_data = [\n")
        for item in data:
            f.write("    {\n")
            f.write(f'        "query": """{item["query"]}""",\n')
            f.write(f'        "pos": """{item["pos"]}""",\n')
            f.write("    },\n")
        f.write("]\n")
    print(f"💾 已保存为 Python 格式: {filepath}")

# ========== 主程序 ==========
if __name__ == "__main__":
    URL = "https://docs.openclaw.ai/help/faq#how-do-i-install-openclaw-on-a-vps"
    
    # 爬取数据
    qa_data = crawl_openclaw_faq(URL)
    
    if qa_data:
        # 保存为 JSON
        save_to_json(qa_data, "openclaw_faq.json")
        
        # 保存为 Python 格式
        save_to_python(qa_data, "openclaw_faq.py")
        
        # 打印预览
        print("\n📋 数据预览:")
        for i, item in enumerate(qa_data[:2], 1):
            print(f"\n[{i}] Query: {item['query']}")
            print(f"    Pos: {item['pos'][:100]}...")
    else:
        print("⚠️  未采集到数据，请检查网页结构或网络")