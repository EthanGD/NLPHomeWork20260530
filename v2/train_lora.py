# -*- coding: utf-8 -*-
import sys, os, json, math, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# 1. Load data
sys.path.insert(0, os.path.dirname(__file__))
import importlib.util
spec = importlib.util.spec_from_file_location("queryPos", os.path.join(os.path.dirname(__file__), "queryPos.py"))
qp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qp)

qa_pairs = [d for d in qp.qa_data if d["query"].strip() and d["pos"].strip()]
print(f"Total QA pairs: {len(qa_pairs)}")

# 2. Config
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models")
output_dir = os.path.join(BASE_DIR, "lora_output")
os.makedirs(output_dir, exist_ok=True)
set_seed(42)

# 3. Format & split
def format_example(query, pos):
    return f"问题：{query}\n\n回答：{pos}"

random.shuffle(qa_pairs)
split_idx = int(len(qa_pairs) * 0.85)
train_pairs = qa_pairs[:split_idx]
val_pairs = qa_pairs[split_idx:]
print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

# 4. Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    model_path,
    device_map=0,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model.config.use_cache = False
model = model.train()

# Freeze vision encoder
for name, param in model.named_parameters():
    if "visual" in name:
        param.requires_grad = False

print(f"Model device: {next(model.parameters()).device}")

# 5. LoRA - only text layers
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    modules_to_save=None,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 6. Dataset
class QADataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length=512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        text = format_example(pair["query"], pair["pos"])
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                              padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

bs = 2 if device == "cpu" else 8
train_dataset = QADataset(train_pairs, tokenizer)
val_dataset = QADataset(val_pairs, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

# 7. Training
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
num_epochs = 30
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

train_losses, val_losses = [], []
best_val_loss = float("inf")

for epoch in range(num_epochs):
    model.train()
    total_train_loss, train_steps = 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_train_loss += loss.item()
        train_steps += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_train_loss = total_train_loss / train_steps
    train_losses.append(avg_train_loss)

    model.eval()
    total_val_loss, val_steps = 0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_val_loss += outputs.loss.item()
            val_steps += 1

    avg_val_loss = total_val_loss / val_steps
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, LR={scheduler.get_last_lr()[0]:.2e}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model.save_pretrained(os.path.join(output_dir, "best"))
        tokenizer.save_pretrained(os.path.join(output_dir, "best"))
        print(f"  -> Saved best model (val_loss={avg_val_loss:.4f})")

model.save_pretrained(os.path.join(output_dir, "final"))
tokenizer.save_pretrained(os.path.join(output_dir, "final"))
print("Training complete!")

# 8. Loss curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, "b-o", label="Train Loss")
plt.plot(range(1, num_epochs+1), val_losses, "r-s", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LoRA Fine-tuning Loss Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150, bbox_inches="tight")
print(f"Loss curve saved")

with open(os.path.join(output_dir, "loss_data.json"), "w", encoding="utf-8") as f:
    json.dump({"train_losses": train_losses, "val_losses": val_losses}, f, ensure_ascii=False, indent=2)

# 9. Evaluation generation
print("\n--- Evaluation ---")
model.eval()
eval_results = []
gen_kwargs = {"max_new_tokens": 256, "do_sample": False, "temperature": 0.1, "top_p": 0.9}

best_model_path = os.path.join(output_dir, "best")
if os.path.exists(best_model_path):
    eval_model = PeftModel.from_pretrained(model, best_model_path)
    eval_model = eval_model.merge_and_unload()
else:
    eval_model = model

for i, pair in enumerate(val_pairs):
    prompt = f"问题：{pair['query']}\n\n回答："
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = eval_model.generate(**inputs, **gen_kwargs,
                                   pad_token_id=tokenizer.pad_token_id,
                                   eos_token_id=tokenizer.eos_token_id)
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    eval_results.append({"query": pair["query"], "reference": pair["pos"][:200], "generated": generated[:200]})
    if i < 5:
        print(f"\nQ: {pair['query'][:60]}...")
        print(f"Ref: {pair['pos'][:80]}...")
        print(f"Gen: {generated[:80]}...")

with open(os.path.join(output_dir, "eval_results.json"), "w", encoding="utf-8") as f:
    json.dump(eval_results, f, ensure_ascii=False, indent=2)

# 10. Evaluation graphs
ref_lens = [len(r["reference"]) for r in eval_results]
gen_lens = [len(r["generated"]) for r in eval_results]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(ref_lens, bins=15, alpha=0.6, label="Reference", color="blue")
axes[0].hist(gen_lens, bins=15, alpha=0.6, label="Generated", color="orange")
axes[0].set_xlabel("Response Length (chars)")
axes[0].set_ylabel("Count")
axes[0].set_title("Response Length Distribution")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].scatter(ref_lens, gen_lens, alpha=0.6)
max_len = max(ref_lens + gen_lens) if (ref_lens + gen_lens) else 1
axes[1].plot([0, max_len], [0, max_len], "r--", label="Ideal")
axes[1].set_xlabel("Reference Length")
axes[1].set_ylabel("Generated Length")
axes[1].set_title("Generated vs Reference Length")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eval_graph.png"), dpi=150, bbox_inches="tight")
print(f"Eval graph saved")
plt.close('all')
print(f"\nAll done! Results in: {output_dir}")
