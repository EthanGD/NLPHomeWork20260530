#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
對比原始模型和微調模型的效果 - 支援命令列參數
"""

import os
import json
import torch
import argparse
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. 命令列參數解析
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="BGE-M3 模型評估對比工具")
    
    parser.add_argument("--original", type=str, default="/wt/code/models/BAAI/bge-m3",
                        help="原始模型路徑 (預設: BGE-M3)")
    
    parser.add_argument("--finetuned", type=str, default="/wt/code/bge-m3-water-industry",
                        help="微調後模型路徑")
    
    parser.add_argument("--testset", type=str, default="/wt/code/test_set.json",
                        help="測試集 JSON 檔案路徑")
    
    parser.add_argument("--gpu", type=str, default="6",
                        help="使用的 GPU 編號 (預設: 6)")
    
    parser.add_argument("--output", type=str, default="/wt/code/evaluation_report.json",
                        help="評估報告輸出路徑")
    
    return parser.parse_args()


# ==========================================
# 主程式
# ==========================================
def main():
    args = parse_args()

    # 設定 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用 GPU: {args.gpu}")
    print(f"原始模型路徑 : {args.original}")
    print(f"微調模型路徑 : {args.finetuned}")
    print(f"測試集路徑   : {args.testset}")
    print("-" * 70)

    # ==========================================
    # 2. 加載測試集
    # ==========================================
    print("正在加載測試集...")
    with open(args.testset, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    print(f"測試集共 {len(test_data)} 條樣本\n")

    # ==========================================
    # 3. 加載模型
    # ==========================================
    print("正在加載原始模型...")
    original_model = SentenceTransformer(
        args.original, 
        device=device, 
        trust_remote_code=True
    )

    print("正在加載微調模型...")
    finetuned_model = SentenceTransformer(
        args.finetuned, 
        device=device, 
        trust_remote_code=True
    )
    print()

    # ==========================================
    # 4. 評估函數（保持不變）
    # ==========================================
    def evaluate_model(model, model_name, test_data):
        results = []
        print(f"正在評估 {model_name} ...")
        
        for item in tqdm(test_data, desc=model_name):
            query = item["query"]
            positive = item["pos"]
            
            # 生成負樣本（從其他 query 的 positive 中隨機選）
            neg_candidates = [d["pos"] for d in test_data if d["query"] != query]
            negative = np.random.choice(neg_candidates) if neg_candidates else "這是一個無關的測試文本。"
            
            # 編碼並計算相似度
            embeddings = model.encode([query, positive, negative], convert_to_tensor=True)
            
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
        print(f"  平均邊際 (Margin)：{avg_margin:.4f}\n")
        
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
    print("="*70)
    print("開始模型對比評估")
    print("="*70 + "\n")

    original_results = evaluate_model(original_model, "原始模型 (BGE-M3)", test_data)
    finetuned_results = evaluate_model(finetuned_model, "微調模型 (BGE-M3-Water)", test_data)

    # ==========================================
    # 6. 對比總結
    # ==========================================
    print("="*70)
    print("對比總結")
    print("="*70)

    improvement_acc = finetuned_results['accuracy'] - original_results['accuracy']
    improvement_margin = finetuned_results['avg_margin'] - original_results['avg_margin']

    print(f"\n{'指標':<20} {'原始模型':<15} {'微調模型':<15} {'提升':<10}")
    print("-"*70)
    print(f"{'準確率':<20} {original_results['accuracy']:<15.2%} {finetuned_results['accuracy']:<15.2%} {improvement_acc:+.2%}")
    print(f"{'正樣本相似度':<20} {original_results['avg_sim_positive']:<15.4f} {finetuned_results['avg_sim_positive']:<15.4f} {finetuned_results['avg_sim_positive'] - original_results['avg_sim_positive']:+.4f}")
    print(f"{'負樣本相似度':<20} {original_results['avg_sim_negative']:<15.4f} {finetuned_results['avg_sim_negative']:<15.4f} {finetuned_results['avg_sim_negative'] - original_results['avg_sim_negative']:+.4f}")
    print(f"{'平均邊際':<20} {original_results['avg_margin']:<15.4f} {finetuned_results['avg_margin']:<15.4f} {improvement_margin:+.4f}")

    if finetuned_results['accuracy'] > original_results['accuracy'] + 0.01:   # 提升超過1%才算明顯成功
        print("\n✅ 微調成功！微調模型表現明顯優於原始模型")
    else:
        print("\n⚠️ 微調效果不明顯或提升有限，建議檢查訓練數據品質與數量")

    # ==========================================
    # 7. 保存評估報告
    # ==========================================
    report = {
        "test_set_size": len(test_data),
        "original_model": original_results,
        "finetuned_model": finetuned_results,
        "improvement": {
            "accuracy": improvement_acc,
            "avg_margin": improvement_margin
        }
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 評估報告已保存至：{args.output}")

    # ==========================================
    # 8. 詳細案例對比 (前 5 條)
    # ==========================================
    print("\n" + "="*70)
    print("詳細案例對比 (前 5 條)")
    print("="*70)

    for i in range(min(5, len(test_data))):
        print(f"\n【案例 {i+1}】 Query: {test_data[i]['query']}")
        print(f"原始模型 → 正: {original_results['results'][i]['sim_positive']:.4f} | 負: {original_results['results'][i]['sim_negative']:.4f}")
        print(f"微調模型 → 正: {finetuned_results['results'][i]['sim_positive']:.4f} | 負: {finetuned_results['results'][i]['sim_negative']:.4f}")


if __name__ == "__main__":
    main()