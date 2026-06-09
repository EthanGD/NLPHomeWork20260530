import os
import sys
from datetime import datetime

# 基础路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, ".." )  # 回退到 testAICodingLoRA

# 模型路径配置
MODEL_PATHS = {
    "1": os.path.join(PROJECT_ROOT, "models", "BAAI", "bge-m3"),
    "2": os.path.join(PROJECT_ROOT, "models", "bge-m3-water-industry"),
}

# 默认向量模型路径
DEFAULT_MODEL_PATH = MODEL_PATHS["1"]

# 数据库配置
DATABASE_PATH = os.path.join(BASE_DIR, "rag.db")
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# vLLM API配置
VLLM_API_BASE = "https://llm.ethanchenyansong.com/llm/v1"
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "Qwen3.5")  # 可根据实际部署模型名修改

# 向量维度 (BGE-M3 输出1024维，稠密向量)
VECTOR_DIM = 1024

# 相似度搜索配置
SEARCH_TOP_K = 10
RAG_CONTEXT_K = 3

# Flask配置
SECRET_KEY = os.getenv("SECRET_KEY", "rag-secret-key-change-in-prod")
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 5001))


def get_model_path(model_arg: str) -> str:
    """
    根据启动参数解析模型路径
    :param model_arg: "1" | "2" | "/custom/path"
    :return: 模型绝对路径
    """
    if model_arg in MODEL_PATHS:
        return MODEL_PATHS[model_arg]
    # 如果是自定义路径，验证是否存在
    if os.path.exists(model_arg) and os.path.isdir(model_arg):
        return os.path.abspath(model_arg)
    raise ValueError(f"Invalid model path: {model_arg}")


class Config:
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = SECRET_KEY
    DEBUG = DEBUG
    HOST = HOST
    PORT = PORT
    DATABASE_PATH = DATABASE_PATH