#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
對比原始模型和微調模型的效果
"""

import os
import json
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. 配置
# ==========================================
ORIGINAL_MODEL_PATH = "/wt/code/models/BAAI/bge-m3"
FINETUNED_MODEL_PATH = "/wt/code/models/finetuned-lora-bge-m3/checkpoint-260"
TEST_SET_PATH = "/wt/code/test_set.json"

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. 加載測試集
# ==========================================
print("正在加載測試集...")
with open(TEST_SET_PATH, 'r', encoding='utf-8') as f:
    test_data = json.load(f)
print(f"測試集共 {len(test_data)} 條樣本\n")

# ==========================================
# 3. 加載模型
# ==========================================
print("正在加載原始模型...")
original_model = SentenceTransformer(
    ORIGINAL_MODEL_PATH, 
    device=device, 
    trust_remote_code=True
)

print("正在加載微調模型...")
finetuned_model = SentenceTransformer(
    FINETUNED_MODEL_PATH, 
    device=device, 
    trust_remote_code=True
)
print()

# ==========================================
# 4. 評估函數
# ==========================================
def evaluate_model(model, model_name, test_data):
    """
    評估模型在測試集上的表現
    """
    results = []
    
    print(f"正在評估 {model_name} ...")
    
    for i, item in enumerate(tqdm(test_data, desc=model_name)):
        query = item["query"]
        positive = item["pos"]
        
        # 生成一個無關的負樣本（從其他數據中隨機選擇）
        neg_candidates = [d["pos"] for d in test_data if d["query"] != query]
        if len(neg_candidates) > 0:
            negative = np.random.choice(neg_candidates)
        else:
            negative = "這是一個無關的測試文本。"
        
        # 編碼
        embeddings = model.encode([query, positive, negative], convert_to_tensor=True)
        
        # 計算相似度
        sim_positive = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0).item()
        sim_negative = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[2], dim=0).item()
        
        results.append({
            "query": query,
            "sim_positive": sim_positive,
            "sim_negative": sim_negative,
            "correct": sim_positive > sim_negative
        })
    
    # 計算統計指標
    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    avg_sim_positive = np.mean([r["sim_positive"] for r in results])
    avg_sim_negative = np.mean([r["sim_negative"] for r in results])
    avg_margin = np.mean([r["sim_positive"] - r["sim_negative"] for r in results])
    
    print(f"\n{model_name} 評估結果:")
    print(f"  準確率：{accuracy:.2%} ({sum(1 for r in results if r['correct'])}/{len(results)})")
    print(f"  正樣本平均相似度：{avg_sim_positive:.4f}")
    print(f"  負樣本平均相似度：{avg_sim_negative:.4f}")
    print(f"  平均邊際 (Margin)：{avg_margin:.4f}")
    print()
    
    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "avg_sim_positive": avg_sim_positive,
        "avg_sim_negative": avg_sim_negative,
        "avg_margin": avg_margin,
        "results": results
    }

# ==========================================
# 5. 執行評估
# ==========================================
print("="*60)
print("開始模型對比評估")
print("="*60 + "\n")

original_results = evaluate_model(original_model, "原始模型 (BGE-M3)", test_data)
finetuned_results = evaluate_model(finetuned_model, "微調模型 (BGE-M3-Water)", test_data)

# ==========================================
# 6. 對比總結
# ==========================================
print("="*60)
print("對比總結")
print("="*60)

print(f"\n{'指標':<20} {'原始模型':<15} {'微調模型':<15} {'提升':<10}")
print("-"*60)
print(f"{'準確率':<20} {original_results['accuracy']:<15.2%} {finetuned_results['accuracy']:<15.2%} {finetuned_results['accuracy'] - original_results['accuracy']:+.2%}")
print(f"{'正樣本相似度':<20} {original_results['avg_sim_positive']:<15.4f} {finetuned_results['avg_sim_positive']:<15.4f} {finetuned_results['avg_sim_positive'] - original_results['avg_sim_positive']:+.4f}")
print(f"{'負樣本相似度':<20} {original_results['avg_sim_negative']:<15.4f} {finetuned_results['avg_sim_negative']:<15.4f} {finetuned_results['avg_sim_negative'] - original_results['avg_sim_negative']:+.4f}")
print(f"{'平均邊際':<20} {original_results['avg_margin']:<15.4f} {finetuned_results['avg_margin']:<15.4f} {finetuned_results['avg_margin'] - original_results['avg_margin']:+.4f}")

# 判斷微調是否成功
if finetuned_results['accuracy'] > original_results['accuracy']:
    print("\n✅ 微調成功！微調模型表現優於原始模型")
else:
    print("\n⚠️ 微調效果不明顯，建議：")
    print("   1. 增加訓練數據量")
    print("   2. 調整學習率或 Epoch")
    print("   3. 添加 Hard Negative 樣本")

# ==========================================
# 7. 保存評估報告
# ==========================================
report = {
    "test_set_size": len(test_data),
    "original_model": original_results,
    "finetuned_model": finetuned_results,
    "improvement": {
        "accuracy": finetuned_results['accuracy'] - original_results['accuracy'],
        "avg_margin": finetuned_results['avg_margin'] - original_results['avg_margin']
    }
}

report_path = "/wt/code/evaluation_report.json"
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n✅ 評估報告已保存至：{report_path}")

# ==========================================
# 8. 詳細案例對比 (前 5 條)
# ==========================================
print("\n" + "="*60)
print("詳細案例對比 (前 5 條)")
print("="*60)

for i in range(min(5, len(test_data))):
    print(f"\n【案例 {i+1}】")
    print(f"Query: {test_data[i]['query']}")
    print(f"原始模型 - 正樣本：{original_results['results'][i]['sim_positive']:.4f} | 負樣本：{original_results['results'][i]['sim_negative']:.4f}")
    print(f"微調模型 - 正樣本：{finetuned_results['results'][i]['sim_positive']:.4f} | 負樣本：{finetuned_results['results'][i]['sim_negative']:.4f}")