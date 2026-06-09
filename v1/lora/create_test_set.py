#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
從原始數據中創建測試集
"""

import json
import random
from queryPos import qa_data

# 設置隨機種子，確保可重複
random.seed(42)

# 打亂數據
shuffled_data = qa_data.copy()
random.shuffle(shuffled_data)

# 留出 30 條作為測試集 (約 15%)
test_size = 30
test_data = shuffled_data[:test_size]
train_data = shuffled_data[test_size:]

# 保存測試集
with open('/wt/code/test_set.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

# 保存訓練集 (用於重新訓練)
with open('/wt/code/train_set.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

print(f"✅ 測試集：{len(test_data)} 條 -> /wt/code/test_set.json")
print(f"✅ 訓練集：{len(train_data)} 條 -> /wt/code/train_set.json")