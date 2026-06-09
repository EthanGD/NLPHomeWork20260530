import os
import sys
import pickle
import numpy as np
from flask import Flask, request, jsonify, g
from sqlalchemy import and_

from config import Config, get_model_path, SEARCH_TOP_K, RAG_CONTEXT_K, VLLM_API_BASE, VLLM_MODEL_NAME
from models import QARecord, get_db, engine, Base
from utils import EmbeddingService, batch_cosine_similarity

import requests

# ============ 初始化 ============
app = Flask(__name__)
app.config.from_object(Config)

# 从启动参数获取模型路径 (python app.py 2)
MODEL_ARG = sys.argv[1] if len(sys.argv) > 1 else "1"
MODEL_PATH = get_model_path(MODEL_ARG)
print(f"🎯 Using embedding model: {MODEL_PATH}")

# 初始化向量服务（单例）
embedding_service = EmbeddingService(MODEL_PATH)


# ============ 数据库上下文 ============
@app.teardown_appcontext
def teardown_db(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def get_db_session():
    if 'db' not in g:
        g.db = next(get_db())
    return g.db


# ============ API: 功能1 - 存储问题答案对 ============
@app.route('/api/store', methods=['POST'])
def store_qa():
    """
    存储问题-答案对
    Request JSON:
    {
        "question": "澳门供水的水源来自哪里？",
        "answer": "澳门供水主要来自珠江流域，通过珠海对澳门供水..."
    }
    """
    db = get_db_session()
    data = request.get_json()
    
    if not data or 'question' not in data or 'answer' not in data:
        return jsonify({"error": "Missing 'question' or 'answer' field"}), 400
    
    question = data['question'].strip()
    answer = data['answer'].strip()
    
    if not question or not answer:
        return jsonify({"error": "Question and answer cannot be empty"}), 400
    
    # 检查是否已存在
    existing = db.query(QARecord).filter(QARecord.question == question).first()
    if existing:
        return jsonify({
            "message": "Question already exists",
            "id": existing.id,
            "updated": False
        }), 200  # 或改为409冲突
    
    # 向量化问题
    try:
        embedding = embedding_service.encode([question])[0]  # (1024,)
        embedding_bytes = embedding_service.serialize(embedding)
    except Exception as e:
        return jsonify({"error": f"Embedding failed: {str(e)}"}), 500
    
    # 存入数据库
    new_record = QARecord(
        question=question,
        answer=answer,
        embedding=embedding_bytes
    )
    db.add(new_record)
    db.commit()
    db.refresh(new_record)
    
    return jsonify({
        "message": "QA pair stored successfully",
        "id": new_record.id,
        "question": new_record.question
    }), 201


# ============ API: 功能2 - 相似问题搜索 ============
@app.route('/api/search', methods=['POST'])
def search_similar():
    """
    根据问题搜索相似记录
    Request JSON:
    {
        "query": "澳门的水从哪里来",
        "top_k": 10  # optional, default 10
    }
    Response:
    {
        "results": [
            {"id": 1, "question": "...", "answer": "...", "similarity": 0.92},
            ...
        ]
    }
    """
    db = get_db_session()
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' field"}), 400
    
    query = data['query'].strip()
    top_k = data.get('top_k', SEARCH_TOP_K)
    
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400
    
    # 1. 编码query
    try:
        query_vec = embedding_service.encode([query])[0]
    except Exception as e:
        return jsonify({"error": f"Embedding failed: {str(e)}"}), 500
    
    # 2. 读取所有记录（生产环境建议用向量数据库如FAISS/Chroma）
    records = db.query(QARecord).all()
    if not records:
        return jsonify({"results": [], "message": "No records found"}), 200
    
    # 3. 批量计算相似度
    results = []
    for record in records:
        stored_vec = embedding_service.deserialize(record.embedding)
        sim = float(np.dot(query_vec, stored_vec))  # 已归一化，点积=余弦相似度
        results.append({
            "record": record,
            "similarity": sim
        })
    
    # 4. 排序取Top-K
    results.sort(key=lambda x: x['similarity'], reverse=True)
    top_results = results[:top_k]
    
    return jsonify({
        "query": query,
        "total_records": len(records),
        "returned": len(top_results),
        "results": [
            {
                "id": r['record'].id,
                "question": r['record'].question,
                "answer": r['record'].answer,
                "similarity": round(r['similarity'], 4),
                "created_at": r['record'].created_at.isoformat() if r['record'].created_at else None
            }
            for r in top_results
        ]
    }), 200


# ============ API: 功能3 - RAG增强生成 ============
@app.route('/api/generate', methods=['POST'])
def rag_generate():
    """
    RAG增强生成：检索+大模型生成
    Request JSON:
    {
        "query": "澳门供水的水质标准是什么？",
        "context_k": 3,  # optional, default 3
        "temperature": 0.1  # optional
    }
    """
    db = get_db_session()
    data = request.get_json()
    
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' field"}), 400
    
    query = data['query'].strip()
    context_k = data.get('context_k', RAG_CONTEXT_K)
    temperature = data.get('temperature', 0.1)
    
    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400
    
    # 1. 编码并检索Top-K上下文
    try:
        query_vec = embedding_service.encode([query])[0]
    except Exception as e:
        return jsonify({"error": f"Embedding failed: {str(e)}"}), 500
    
    records = db.query(QARecord).all()
    if not records:
        # 无知识库时直接调用大模型
        return _call_vllm(query, [], temperature)
    
    # 计算相似度并排序
    scored = []
    for record in records:
        stored_vec = embedding_service.deserialize(record.embedding)
        sim = float(np.dot(query_vec, stored_vec))
        scored.append((record, sim))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    top_records = scored[:context_k]
    
    # 2. 构建RAG Prompt
    context_text = "\n\n".join([
        f"参考知识 {i+1}:\n问: {rec.question}\n答: {rec.answer}"
        for i, (rec, _) in enumerate(top_records)
    ])
    
    rag_prompt = f"""你是一个专业的澳门供水行业智能助手。请根据以下参考知识回答问题，如果参考知识与问题无关，请基于你的通用知识回答。

【参考知识】
{context_text}

【用户问题】
{query}

【回答要求】
1. 优先依据参考知识回答，标注来源
2. 保持专业、准确、简洁
3. 如知识不足，请诚实说明

【回答】
"""
    
    # 3. 调用vLLM API
    return _call_vllm(rag_prompt, top_records, temperature, original_query=query)


def _call_vllm(prompt: str, context_records: list, temperature: float, original_query: str = None):
    """调用vLLM OpenAI兼容接口"""
    url = f"{VLLM_API_BASE}/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        # 如有API Key可添加: "Authorization": f"Bearer {API_KEY}"
    }
    
    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 2048,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        generated_answer = result["choices"][0]["message"]["content"]
        
        # 构建响应
        return jsonify({
            "query": original_query or prompt,
            "answer": generated_answer,
            "context_used": [
                {
                    "id": rec.id,
                    "question": rec.question,
                    "similarity": round(sim, 4)
                }
                for rec, sim in context_records
            ] if context_records else [],
            "model": VLLM_MODEL_NAME,
            "usage": result.get("usage", {})
        }), 200
        
    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": f"vLLM API call failed: {str(e)}",
            "fallback_answer": "抱歉，暂时无法连接到大模型服务，请稍后重试。"
        }), 502


# ============ 健康检查 ============
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_path": MODEL_PATH,
        "database": Config.DATABASE_PATH,
        "vllm_endpoint": VLLM_API_BASE
    }), 200


# ============ 初始化数据库（可选） ============
# @app.before_first_request
# def init_database():
#     Base.metadata.create_all(bind=engine)
def init_database():
    """初始化数据库表"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables initialized")


# ============ 启动入口 ============
if __name__ == '__main__':
    # 初始化数据库 # 替換@app.before_first_request
    init_database()
    print(f"🚀 Starting RAG Flask Server on {Config.HOST}:{Config.PORT}")
    print(f"📦 Model: {MODEL_PATH}")
    print(f"🗄️  Database: {Config.DATABASE_PATH}")
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)