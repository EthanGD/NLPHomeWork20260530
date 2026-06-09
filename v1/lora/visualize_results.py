#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可視化模型對比結果 - 支援微軟雅黑中文
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# ==================== 設定中文字型 ====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False   # 解決負號顯示為方塊的問題

# 加載評估報告
with open('evaluation_report.json', 'r', encoding='utf-8') as f:
    report = json.load(f)

original = report['original_model']
finetuned = report['finetuned_model']

# 提取數據
orig_sim_pos = [r['sim_positive'] for r in original['results']]
fine_sim_pos = [r['sim_positive'] for r in finetuned['results']]

margins_orig = [r['sim_positive'] - r['sim_negative'] for r in original['results']]
margins_fine = [r['sim_positive'] - r['sim_negative'] for r in finetuned['results']]

# 創建圖表
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1. 準確率對比
acc_orig = original['accuracy']
acc_fine = finetuned['accuracy']

bars = axes[0].bar(['原始模型', '微調模型'], 
                   [acc_orig, acc_fine], 
                   color=['#FF6B6B', '#4ECDC4'])
axes[0].set_ylabel('準確率')
axes[0].set_title('準確率對比', fontsize=14, pad=15)
for bar, v in zip(bars, [acc_orig, acc_fine]):
    axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.01, 
                 f'{v:.2%}', ha='center', va='bottom', fontsize=12)

# 2. 正樣本相似度分布
axes[1].hist(orig_sim_pos, bins=20, alpha=0.75, label='原始模型', 
             color='#FF6B6B', edgecolor='black')
axes[1].hist(fine_sim_pos, bins=20, alpha=0.75, label='微調模型', 
             color='#4ECDC4', edgecolor='black')
axes[1].set_xlabel('相似度 (Positive Similarity)')
axes[1].set_ylabel('頻數')
axes[1].set_title('正樣本相似度分布', fontsize=14, pad=15)
axes[1].legend(fontsize=11)

# 3. 相似度邊際分布
bp = axes[2].boxplot([margins_orig, margins_fine], 
                     labels=['原始模型', '微調模型'],
                     patch_artist=True,
                     widths=0.6)

# 美化 boxplot
colors = ['#FF6B6B', '#4ECDC4']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor('#2C3E50')

axes[2].set_ylabel('邊際 (sim_positive - sim_negative)')
axes[2].set_title('相似度邊際分布', fontsize=14, pad=15)
axes[2].grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('model_comparison.png', 
            dpi=300, bbox_inches='tight')

print("✅ 可視化圖表已成功保存至：model_comparison.png")
print("   已使用微軟雅黑字型支援中文顯示")
plt.show()