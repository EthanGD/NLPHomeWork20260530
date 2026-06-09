#!/usr/bin/env python3
"""
初始化数据库并插入示例数据
使用方法: python init_db.py
"""
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import engine, Base, QARecord, SessionLocal
from utils import EmbeddingService
from config import get_model_path

# 加载模型
MODEL_ARG = sys.argv[1] if len(sys.argv) > 1 else "1"
MODEL_PATH = get_model_path(MODEL_ARG)
embedding_service = EmbeddingService(MODEL_PATH)

def init_db():
    # 创建表
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")
    
    # 示例数据
    samples = [
        ("澳门的供水水源来自哪里？", "澳门供水主要来自珠江流域，由珠海通过竹仙洞水库、洪湾泵站等设施向澳门供水，同时澳门建有石排湾水库、九澳水库等本地蓄水设施作为调节。"),
        ("澳门自来水的水质标准是什么？", "澳门自来水水质严格遵循国家《生活饮用水卫生标准》(GB5749-2022)，并参考世界卫生组织指南，定期检测106项指标，确保水质安全。"),
        ("遇到停水怎么办？", "如遇计划性停水，澳门水务局会提前通过官网、微信公众号、短信等渠道通知。突发停水可拨打24小时服务热线+853 2833 3000查询。")
    ]
    
    db = SessionLocal()
    for question, answer in samples:
        # 检查是否已存在
        if db.query(QARecord).filter(QARecord.question == question).first():
            print(f"⏭️  Skip existing: {question[:20]}...")
            continue
        
        # 向量化并存储
        embedding = embedding_service.encode([question])[0]
        record = QARecord(
            question=question,
            answer=answer,
            embedding=embedding_service.serialize(embedding)
        )
        db.add(record)
        print(f"➕ Added: {question}")
    
    db.commit()
    db.close()
    print("✅ Sample data inserted")


if __name__ == '__main__':
    init_db()