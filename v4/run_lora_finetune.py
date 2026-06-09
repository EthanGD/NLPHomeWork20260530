#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键运行脚本：生成问答对 + LoRA 微调
"""

import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(script_name, description):
    """Run a script and check for errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(
        [sys.executable, script_name],
        cwd=BASE_DIR,
        capture_output=False,
        encoding="utf-8"
    )
    
    if result.returncode != 0:
        print(f"\nERROR: {description} failed!")
        return False
    return True

def main():
    print("="*60)
    print("中医针灸 LoRA 微调流程")
    print("="*60)
    
    # Step 1: Generate QA pairs
    if not run_script("generate_qa_pairs.py", "生成问答对"):
        sys.exit(1)
    
    # Step 2: LoRA fine-tuning
    if not run_script("train_lora.py", "LoRA 微调"):
        sys.exit(1)
    
    print("\n" + "="*60)
    print("  全部完成！")
    print("="*60)
    print("\n输出文件:")
    print("  - qa_dataset.jsonl: 问答对数据集")
    print("  - results/lora_model/: LoRA 微调模型")
    print("  - results/lora_training_metrics.csv: 训练指标")
    print("  - results/lora_training_metrics.png: 训练曲线图")
    print("  - results/lora_loss_curve.png: 损失曲线")
    print("  - results/lora_final_metrics.json: 最终评估报告")
    print()

if __name__ == "__main__":
    main()
