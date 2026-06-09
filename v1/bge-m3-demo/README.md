# Flask + bge-m3 向量搜索 Demo

## 功能说明

### 功能 1：添加问题 - 答案对
- 将问题使用 bge-m3 模型向量化
- 问题作为搜索字段，同时存储原始问题和答案
- 问题字段有唯一性限制

### 功能 2：搜索问题
- 输入问题，从 SQLite 中读取记录
- 计算查询向量与存储向量的欧式距离
- 返回距离最近的 10 条记录

## API 接口

### 1. 添加问题 - 答案对
```
POST /api/add
Content-Type: application/json

请求体:
{
    "question": "问题文本",
    "answer": "答案文本"
}

响应:
{
    "message": "添加成功",
    "question": "问题文本"
}
```

### 2. 搜索问题
```
POST /api/search
Content-Type: application/json

请求体:
{
    "question": "问题文本"
}

响应:
{
    "results": [
        {
            "question": "存储的问题",
            "answer": "存储的答案",
            "distance": 0.123
        }
    ],
    "count": 10
}
```

### 3. 健康检查
```
GET /api/health

响应:
{
    "status": "healthy",
    "model": "bge-m3"
}
```

## 安装和运行

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行服务
```bash
python app.py
```

### 3. 测试示例

#### 添加数据
```bash
curl -X POST http://localhost:5000/api/add \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是机器学习？", "answer": "机器学习是人工智能的一个分支..."}'
```

#### 搜索数据
```bash
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"question": "机器学习的定义是什么？"}'
```

## 数据库结构

SQLite 数据库文件：`qa_vectors.db`

表结构：
```sql
CREATE TABLE qa_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT UNIQUE NOT NULL,
    answer TEXT NOT NULL,
    vector TEXT NOT NULL
)
```
