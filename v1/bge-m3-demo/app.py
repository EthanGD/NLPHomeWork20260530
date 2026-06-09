"""
Flask 应用：使用 bge-m3 向量模型存储和搜索问题-答案对
优化版本：支持余弦相似度、BLOB存储、批量操作、日志记录
"""

import os
import sys
import sqlite3
import pickle
import logging
from functools import wraps
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from FlagEmbedding import FlagModel

# ==================== 配置 ====================
# 使用绝对路径，避免相对路径解析问题
BASE_DIR = Path(__file__).parent.resolve()
MODEL_LOCAL_PATH = BASE_DIR / 'models' / 'bge-m3'  # ./models/bge-m3
DATABASE = "qa_vectors.db"
MODEL_NAME = "BAAI/bge-m3"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
USE_FP16 = False  # WSL/CPU 环境建议关闭，避免兼容问题
VECTOR_DIM = 1024  # bge-m3 的向量维度
TOP_K = 10  # 默认返回结果数量

# 确保目录存在
MODEL_LOCAL_PATH.mkdir(parents=True, exist_ok=True)

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log", encoding="utf-8", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

# ==================== Flask 应用初始化 ====================
app = Flask(__name__)
CORS(app)  # 启用跨域支持

# ==================== 全局模型单例 ====================
_model = None


def _is_valid_model_folder(path: Path) -> bool:
    """
    检查路径是否为有效的模型文件夹
    需要包含关键文件：config.json + 至少一个模型权重文件
    """
    if not path.exists() or not path.is_dir():
        return False
    required_files = ['config.json']
    weight_files = ['pytorch_model.bin', 'model.safetensors', 'pytorch_model.bin.index.json']
    
    has_config = any((path / f).exists() for f in required_files)
    has_weights = any((path / f).exists() for f in weight_files)
    
    return has_config and has_weights


def get_model():
    """
    延迟加载模型单例，带 fallback 逻辑：
    1. 优先尝试从本地路径加载
    2. 如果本地不存在，则从 HF Hub 下载并缓存到本地
    """
    global _model
    if _model is not None:
        return _model
    
    logger.info(f"正在初始化模型 (本地路径: {MODEL_LOCAL_PATH})...")
    
    try:
        # 🔹 策略 1: 尝试从本地路径加载
        if _is_valid_model_folder(MODEL_LOCAL_PATH):
            logger.info(f"检测到本地模型，从 {MODEL_LOCAL_PATH} 加载...")
            _model = FlagModel(
                str(MODEL_LOCAL_PATH),  # 转字符串，避免 Path 对象兼容性问题
                query_instruction_for_retrieval=QUERY_INSTRUCTION,
                use_fp16=USE_FP16,
                # 本地加载时不需要 cache_dir
            )
            logger.info("✅ 本地模型加载成功！")
            return _model
        else:
            logger.info(f"本地路径无效或模型不完整，准备从 Hub 下载...")
            
        # 🔹 策略 2: 从 Hub 下载，自动缓存到指定目录
        logger.info(f"正在从 HuggingFace Hub 下载 {MODEL_NAME} 到 {MODEL_LOCAL_PATH}...")
        _model = FlagModel(
            MODEL_NAME,  # 使用 repo_id，让 HF 自动处理下载
            query_instruction_for_retrieval=QUERY_INSTRUCTION,
            use_fp16=USE_FP16,
            cache_dir=str(MODEL_LOCAL_PATH.parent),  # 缓存到 ./models/ 目录
        )
        logger.info("✅ 模型下载并加载成功！")
        return _model
        
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}", exc_info=True)
        # 清理可能的半成品
        _model = None
        raise


# ==================== 数据库工具 ====================
def get_db_connection():
    """获取数据库连接"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库表"""
    conn = get_db_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS qa_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT UNIQUE NOT NULL,
            answer TEXT NOT NULL,
            vector BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_question ON qa_pairs(question)")
    conn.commit()
    conn.close()
    logger.info("数据库初始化完成")


def vector_to_bytes(vector: np.ndarray) -> bytes:
    """向量 → 二进制 (pickle)"""
    return pickle.dumps(vector)


def bytes_to_vector(data: bytes) -> np.ndarray:
    """二进制 → 向量"""
    return pickle.loads(data)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    计算两个向量的余弦相似度
    返回: 0.0 ~ 1.0，越大表示越相似
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


# ==================== 请求验证装饰器 ====================
def validate_json(*required_fields):
    """
    装饰器：验证请求是否为 JSON 且包含必需字段
    用法: @validate_json('question', 'answer')
    """

    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            if not request.is_json:
                return jsonify({"error": "Content-Type must be application/json"}), 400
            data = request.get_json(silent=True)
            if data is None:
                return jsonify({"error": "Invalid JSON body"}), 400
            missing = [
                field
                for field in required_fields
                if field not in data or not data[field]
            ]
            if missing:
                return jsonify({"error": f"Missing required fields: {missing}"}), 400
            return f(*args, **kwargs)

        return wrapped

    return decorator


# ==================== API 路由 ====================


@app.route("/api/health", methods=["GET"])
def health_check():
    """健康检查接口"""
    return (
        jsonify(
            {
                "status": "healthy",
                "model": MODEL_NAME,
                "model_loaded": _model is not None,
            }
        ),
        200,
    )


@app.route("/api/add", methods=["POST"])
@validate_json("question", "answer")
def add_qa_pair():
    """
    添加单个问题-答案对
    请求体: {"question": "问题文本", "answer": "答案文本"}
    """
    try:
        data = request.get_json()
        question = data["question"].strip()
        answer = data["answer"].strip()

        # 向量化（使用 query_instruction）
        embeddings = get_model().encode([question], convert_to_numpy=True)
        vector = embeddings[0]

        conn = get_db_connection()
        try:
            conn.execute(
                "INSERT INTO qa_pairs (question, answer, vector) VALUES (?, ?, ?)",
                (question, answer, vector_to_bytes(vector)),
            )
            conn.commit()
            logger.info(f"成功添加 QA: {question[:50]}...")
            return (
                jsonify(
                    {
                        "message": "添加成功",
                        "question": question,
                        "vector_shape": vector.shape,
                    }
                ),
                201,
            )
        except sqlite3.IntegrityError:
            logger.warning(f"问题已存在: {question[:50]}...")
            return jsonify({"error": "问题已存在，无法重复添加"}), 409
        finally:
            conn.close()

    except Exception as e:
        logger.error(f"添加 QA 失败: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/batch_add", methods=["POST"])
@validate_json("items")
def batch_add_qa_pairs():
    """
    批量添加问题-答案对
    请求体: {"items": [{"question": "...", "answer": "..."}, ...]}
    """
    try:
        data = request.get_json()
        items = data["items"]

        if not isinstance(items, list) or len(items) == 0:
            return jsonify({"error": "items 必须是非空数组"}), 400

        # 提取问题列表，批量向量化（比循环快 3-5 倍）
        questions = [item["question"].strip() for item in items]
        answers = [item["answer"].strip() for item in items]

        # 批量编码
        embeddings = get_model().encode(questions, convert_to_numpy=True)

        conn = get_db_connection()
        success_count = 0
        errors = []

        for i, (question, answer, vector) in enumerate(
            zip(questions, answers, embeddings)
        ):
            try:
                conn.execute(
                    "INSERT INTO qa_pairs (question, answer, vector) VALUES (?, ?, ?)",
                    (question, answer, vector_to_bytes(vector)),
                )
                success_count += 1
            except sqlite3.IntegrityError:
                errors.append(
                    {"index": i, "question": question[:30], "error": "问题已存在"}
                )
            except Exception as e:
                errors.append({"index": i, "question": question[:30], "error": str(e)})

        conn.commit()
        conn.close()

        logger.info(f"批量添加完成: {success_count}/{len(items)} 成功")

        return jsonify(
            {
                "message": f"成功添加 {success_count}/{len(items)} 条",
                "success_count": success_count,
                "total": len(items),
                "errors": errors if errors else None,
            }
        ), (201 if success_count > 0 else 400)

    except Exception as e:
        logger.error(f"批量添加失败: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/search", methods=["POST"])
@validate_json("question")
def search_qa_pair():
    """
    搜索相似问题
    请求体: {"question": "查询文本", "top_k": 10}
    返回: 余弦相似度最高的 top_k 条记录
    """
    try:
        data = request.get_json()
        question = data["question"].strip()
        top_k = data.get("top_k", TOP_K)

        # 查询向量化
        query_embedding = get_model().encode([question], convert_to_numpy=True)[0]

        conn = get_db_connection()
        rows = conn.execute(
            "SELECT id, question, answer, vector FROM qa_pairs"
        ).fetchall()
        conn.close()

        if not rows:
            return (
                jsonify({"message": "数据库中暂无记录", "results": [], "count": 0}),
                200,
            )

        # 计算余弦相似度
        results = []
        for row in rows:
            stored_vector = bytes_to_vector(row["vector"])
            similarity = cosine_similarity(query_embedding, stored_vector)
            results.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "similarity": round(similarity, 4),  # 保留4位小数
                    "distance": round(1 - similarity, 4),  # 距离 = 1 - 相似度
                }
            )

        # 按相似度降序排序，取 top_k
        results.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = results[:top_k]

        logger.info(f"搜索 '{question[:30]}...' 返回 {len(top_results)} 条结果")

        return (
            jsonify(
                {
                    "query": question,
                    "results": top_results,
                    "count": len(top_results),
                    "total_in_db": len(results),
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"搜索失败: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/delete", methods=["DELETE"])
@validate_json("question")
def delete_qa_pair():
    """
    删除问题-答案对
    请求体: {"question": "问题文本"}
    """
    try:
        data = request.get_json()
        question = data["question"].strip()

        conn = get_db_connection()
        cursor = conn.execute("DELETE FROM qa_pairs WHERE question = ?", (question,))
        conn.commit()
        deleted = cursor.rowcount
        conn.close()

        if deleted == 0:
            return jsonify({"error": "问题不存在"}), 404

        logger.info(f"成功删除: {question[:50]}...")
        return (
            jsonify(
                {"message": "删除成功", "deleted_count": deleted, "question": question}
            ),
            200,
        )

    except Exception as e:
        logger.error(f"删除失败: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/list", methods=["GET"])
def list_qa_pairs():
    """
    列出所有问题-答案对（支持分页）
    查询参数: ?page=1&limit=20
    """
    try:
        page = request.args.get("page", 1, type=int)
        limit = request.args.get("limit", 20, type=int)
        limit = min(limit, 100)  # 限制最大返回数量

        offset = (page - 1) * limit

        conn = get_db_connection()
        # 获取总数
        total = conn.execute("SELECT COUNT(*) FROM qa_pairs").fetchone()[0]
        # 获取分页数据
        rows = conn.execute(
            "SELECT id, question, answer, created_at FROM qa_pairs ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        conn.close()

        items = [
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

        return (
            jsonify(
                {
                    "items": items,
                    "page": page,
                    "limit": limit,
                    "total": total,
                    "pages": (total + limit - 1) // limit,
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"列表查询失败: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ==================== 错误处理 ====================


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "接口不存在"}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"服务器内部错误: {error}")
    return jsonify({"error": "服务器内部错误"}), 500


# ==================== 应用入口 ====================

if __name__ == "__main__":
    # 初始化数据库
    init_db()

    # 预加载模型（可选：如果希望启动时就加载）
    # get_model()

    logger.info("=" * 50)
    logger.info(f"API 服务启动中...")
    logger.info(f"模型: {MODEL_NAME}")
    logger.info(f"使用 FP16: {USE_FP16}")
    logger.info(f"数据库: {DATABASE}")
    logger.info("=" * 50)

    # 开发环境直接运行，生产环境建议用 gunicorn
    is_debug = os.getenv("FLASK_ENV") == "development"

    if is_debug:
        app.run(host="0.0.0.0", port=5000, debug=True)
    else:
        logger.info("生产环境建议使用: gunicorn -w 2 -b 0.0.0.0:5000 app:app")
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
