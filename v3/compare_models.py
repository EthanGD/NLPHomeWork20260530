#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比微调前后模型的生成效果
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================
#  CONFIGURATION
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_MODEL = os.path.join(BASE_DIR, "models")
FINETUNED_MODEL = os.path.join(BASE_DIR, "results", "checkpoints", "checkpoint-800")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# 测试问题列表
TEST_PROMPTS = [
    "腹胀可以刺什么穴位",
    "头痛可以刺什么穴位",
    "十二经脉包括哪些",
    "针灸的禁忌有哪些",
    "足三里的功效是",
]

# ============================
#  LOAD MODEL
# ============================
def load_model(model_path):
    print(f"\nLoading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  ✓ Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    return model, tokenizer

# ============================
#  GENERATE
# ============================
def generate(model, tokenizer, prompt, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

# ============================
#  MAIN
# ============================
def main():
    print("="*70)
    print("  Model Comparison: Original vs Finetuned")
    print("="*70)
    
    # Load models
    orig_model, orig_tokenizer = load_model(ORIGINAL_MODEL)
    ft_model, ft_tokenizer = load_model(FINETUNED_MODEL)
    
    # Test
    print("\n" + "="*70)
    print("  Testing Prompts")
    print("="*70 + "\n")
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n{'─'*70}")
        print(f"  Test {i}: {prompt}")
        print(f"{'─'*70}")
        
        # Original model
        print(f"\n【原模型】Qwen3.5-0.8B-Base")
        orig_response = generate(orig_model, orig_tokenizer, prompt)
        print(f"{orig_response}")
        
        # Finetuned model
        print(f"\n【微调模型】checkpoint-800")
        ft_response = generate(ft_model, ft_tokenizer, prompt)
        print(f"{ft_response}")
        
        # Compare length
        print(f"\n  对比：原模型生成 {len(orig_response)} 字 | 微调模型生成 {len(ft_response)} 字")
    
    print("\n" + "="*70)
    print("  Comparison Complete")
    print("="*70)

if __name__ == "__main__":
    main()
