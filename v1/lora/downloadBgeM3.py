#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下載 BAAI/bge-m3 模型到本地目錄
"""

import os
from sentence_transformers import SentenceTransformer

# 模型名稱
model_name = "BAAI/bge-m3"

# 本地保存路徑
local_path = "models/BAAI/bge-m3"

# 創建目錄
os.makedirs(local_path, exist_ok=True)

print(f"正在下載模型：{model_name}")
print(f"保存路徑：{local_path}")

# 下載並保存模型
model = SentenceTransformer(model_name, trust_remote_code=True)
model.save(local_path)

print(f"\n✅ 模型下載完成！")
print(f"模型路徑：{local_path}")
print(f"\n目錄結構：")
for root, dirs, files in os.walk(local_path):
    level = root.replace(local_path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 2 * (level + 1)
    for file in files[:10]:  # 只顯示前 10 個文件
        print(f'{subindent}{file}')