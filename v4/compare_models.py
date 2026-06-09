#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比微调前后模型的生成效果
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ============================
#  CONFIGURATION
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL = os.path.join(BASE_DIR, "models")
LORA_MODEL = os.path.join(BASE_DIR, "results", "lora_model")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# 测试问题列表
TEST_PROMPTS = [
    "腹胀可以刺什么穴位",
    "头痛可以刺什么穴位",
    "十二经脉包括哪些",
    "针灸的禁忌有哪些",
    "足三里的功效是",
    "什么是经络",
    "腧穴的作用是什么",
]

# ============================
#  LOAD MODEL
# ============================
def load_base_model():
    """加载基础模型"""
    print(f"\nLoading base model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  ✓ Base model loaded ({sum(p.numel() for p in model.parameters()):,} params)")
    return model, tokenizer

def load_lora_model(base_model, tokenizer):
    """加载 LoRA 微调模型"""
    print(f"\nLoading LoRA adapter: {LORA_MODEL}")
    model = PeftModel.from_pretrained(base_model, LORA_MODEL)
    print(f"  ✓ LoRA model loaded")
    return model

# ============================
#  GENERATE
# ============================
def generate(model, tokenizer, prompt, max_new_tokens=512):
    # Qwen3.5 格式
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stop_strings=["<|eot_id|>"],
            tokenizer=tokenizer,
        )
    
    generated = outputs[0][input_len:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    # 清理特殊标记
    text = text.replace("<|eot_id|>", "").replace("<|start_header_id|>", "").replace("<|end_header_id|>", "")
    return text.strip()

# ============================
#  MAIN
# ============================
def main():
    print("="*70)
    print("  Model Comparison: Original vs LoRA Finetuned")
    print("="*70)
    
    # Load base model once
    base_model, tokenizer = load_base_model()
    
    # Load LoRA adapter
    lora_model = load_lora_model(base_model, tokenizer)
    
    # Test
    print("\n" + "="*70)
    print("  Testing Prompts")
    print("="*70 + "\n")
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n{'='*70}")
        print(f"Q{i}: {prompt}")
        print('='*70)
        
        # Original model
        print(f"\n【原模型】Qwen3.5-0.8B-Base")
        print('-'*70)
        orig_response = generate(base_model, tokenizer, prompt)
        print(orig_response)
        
        # LoRA model
        print(f"\n【LoRA 微调】中医针灸领域")
        print('-'*70)
        lora_response = generate(lora_model, tokenizer, prompt)
        print(lora_response)
        
        # Compare
        print(f"\n📊 字数：原模型 {len(orig_response)}字 | LoRA 模型 {len(lora_response)}字")
    
    print("\n" + "="*70)
    print("  Comparison Complete")
    print("="*70)

if __name__ == "__main__":
    main()
