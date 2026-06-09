import pickle
import numpy as np
from typing import List, Optional
from FlagEmbedding import BGEM3FlagModel
from config import get_model_path, VECTOR_DIM


class EmbeddingService:
    """BGE-M3 向量编码服务（单例）"""
    _instance = None
    _model: Optional[BGEM3FlagModel] = None
    
    def __new__(cls, model_path: str):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_path: str):
        if self._model is None:
            print(f"🔄 Loading BGE-M3 model from: {model_path}")
            self._model = BGEM3FlagModel(
                model_path,
                use_fp16=True,  # 节省显存，如需更高精度可改为False
                # devices=["cuda:0"] if self._check_cuda() else ["cpu"]
                device="cuda:0" if self._check_cuda() else "cpu"
            )
            print("✅ Model loaded successfully")
    
    @staticmethod
    def _check_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        编码文本为稠密向量
        :param texts: 文本列表
        :param normalize: 是否L2归一化（计算余弦相似度必需）
        :return: (n, 1024) numpy array
        """
        result = self._model.encode(
            texts,
            batch_size=8,
            max_length=8192,  # BGE-M3支持长文本
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        embeddings = result["dense_vecs"]
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings
    
    def serialize(self, embedding: np.ndarray) -> bytes:
        """向量化为bytes存入数据库"""
        return pickle.dumps(embedding.astype(np.float32))
    
    def deserialize(self, data: bytes) -> np.ndarray:
        """从数据库读取并还原向量"""
        return pickle.loads(data)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算两个归一化向量的余弦相似度"""
    return float(np.dot(vec1, vec2))


def batch_cosine_similarity(query_vec: np.ndarray, candidate_vecs: List[np.ndarray]) -> List[float]:
    """批量计算余弦相似度"""
    if len(candidate_vecs) == 0:
        return []
    # 假设向量已归一化，点积=余弦相似度
    similarities = [cosine_similarity(query_vec, vec) for vec in candidate_vecs]
    return similarities