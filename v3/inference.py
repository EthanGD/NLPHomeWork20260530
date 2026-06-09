#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA 微调模型推理脚本
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================
#  CONFIGURATION
# ============================
# 使用最佳 checkpoint (根据日志，checkpoint-800 时 eval_accuracy 最高 0.9165)
# MODEL_PATH = r"\NLP\v3\results\checkpoints\checkpoint-800"

MODEL_PATH = r"/wt/v3/results/checkpoints/checkpoint-800"
# 或者使用 final_model
# MODEL_PATH = r"\NLP\v3\results\final_model"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print(f"Loading model from: {MODEL_PATH}")
print(f"Device: {DEVICE}, Dtype: {DTYPE}")

# ============================
#  LOAD MODEL & TOKENIZER
# ============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

print(f"Model loaded successfully!")
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

# ============================
#  INFERENCE
# ============================
def generate_text(prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
    """Generate text from a prompt"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)

print("\n" + "="*60)
print("  Model Ready - Enter prompts (type 'quit' to exit)")
print("="*60 + "\n")

while True:
    prompt = input("Enter prompt: ").strip()
    if prompt.lower() in ['quit', 'exit', 'q']:
        break
    if not prompt:
        continue
    
    print("\nGenerating...")
    response = generate_text(prompt)
    print(f"\nResponse:\n{response}\n")
    print("-"*60 + "\n")
