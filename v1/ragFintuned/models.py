from sqlalchemy import create_engine, Column, Integer, String, Text, LargeBinary, DateTime, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config import DATABASE_URL, VECTOR_DIM

Base = declarative_base()


class QARecord(Base):
    """问题-答案对存储模型"""
    __tablename__ = "qa_records"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(String(500), unique=True, nullable=False, index=True, comment="问题文本（唯一）")
    answer = Column(Text, nullable=False, comment="答案文本")
    embedding = Column(LargeBinary, nullable=False, comment="问题向量化结果（binary pickle）")
    created_at = Column(DateTime, default=func.now(), comment="创建时间")
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), comment="更新时间")
    
    def to_dict(self):
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


# 数据库会话管理
engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

Base.metadata.create_all(bind=engine)


def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()