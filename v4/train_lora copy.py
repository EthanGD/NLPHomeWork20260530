#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3.5-0.8B-Base LoRA 微调脚本
数据：qa_dataset.jsonl (中医针灸问答对)
评估指标：Train Loss, Val Loss, Perplexity, Token Accuracy
输出：损失曲线图、评估报告
"""

import os
import sys
import json
import math
import csv
import re
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ============================
#  CONFIGURATION
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models")
DATA_FILE = os.path.join(BASE_DIR, "qa_dataset.jsonl")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
LORA_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "lora_model")

# CUDA device
CUDA_DEVICE = "cuda:6"

# Training hyperparameters
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
WARMUP_STEPS = 50
LOGGING_STEPS = 10
EVAL_STEPS = 50
SAVE_STEPS = 100
TEST_SPLIT = 0.1
SEED = 42

# LoRA parameters
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LORA_OUTPUT_DIR, exist_ok=True)

# Set CUDA device
if CUDA_DEVICE and CUDA_DEVICE != "auto":
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE.split(":")[-1]
    print(f"Using CUDA device: {CUDA_DEVICE}")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

print(f"Model:    {MODEL_PATH}")
print(f"Data:     {DATA_FILE}")
print(f"Output:   {LORA_OUTPUT_DIR}")

# ============================
#  1. LOAD TOKENIZER
# ============================
print("\n[1/6] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"  Vocab size: {tokenizer.vocab_size}")
print(f"  Pad token: {tokenizer.pad_token}")

# ============================
#  2. LOAD & PREPARE DATA
# ============================
print("\n[2/6] Loading data...")

def format_prompt(example):
    """Format QA pair into prompt-completion format"""
    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{example['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{example['answer']}<|eot_id|>"""
    return prompt

def tokenize_function(examples):
    """Tokenize and pad sequences"""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding="max_length",
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Load dataset
if os.path.exists(DATA_FILE):
    dataset = load_dataset("json", data_files=DATA_FILE)
    print(f"  Loaded {len(dataset['train'])} QA pairs")
    
    # Format into prompt-completion
    formatted_data = []
    for item in dataset["train"]:
        formatted_data.append({
            "text": format_prompt(item)
        })
    
    dataset = Dataset.from_list(formatted_data)
else:
    print(f"ERROR: Data file not found: {DATA_FILE}")
    print("Please run generate_qa_pairs.py first")
    sys.exit(1)

# Split into train/val
split = dataset.train_test_split(test_size=TEST_SPLIT, seed=SEED)
print(f"  Train: {len(split['train'])} | Val: {len(split['test'])}")

# Tokenize
print("  Tokenizing...")
tokenized_dataset = split.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
)

# ============================
#  3. LOAD MODEL
# ============================
print("\n[3/6] Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Configure LoRA
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print(f"  Model loaded with LoRA")
print(f"  Device: {model.device}")

# ============================
#  4. TRAINING
# ============================
print("\n[4/6] Starting training...")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir=LORA_OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    warmup_steps=WARMUP_STEPS,
    logging_steps=LOGGING_STEPS,
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    eval_strategy="steps",
    logging_strategy="steps",
    save_strategy="steps",
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    fp16=True,
    dataloader_drop_last=False,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
    seed=SEED,
    logging_first_step=True,
)

class LoRATrainer(Trainer):
    """Custom Trainer with metrics logging"""
    
    def __init__(self, *args, **kwargs):
        kwargs.pop("compute_metrics", None)
        super().__init__(*args, **kwargs)
        self.metrics = {
            "step": [],
            "train_loss": [],
            "eval_loss": [],
            "perplexity": [],
            "learning_rate": [],
        }
    
    def log(self, logs, *args, **kwargs):
        super().log(logs, *args, **kwargs)
        step = self.state.global_step
        self.metrics["step"].append(step)
        
        if "loss" in logs:
            self.metrics["train_loss"].append(logs["loss"])
            self.metrics["learning_rate"].append(logs.get("learning_rate", 0))
        
        if "eval_loss" in logs:
            idx = len(self.metrics["step"]) - 1
            self.metrics["eval_loss"].append(logs["eval_loss"])
            self.metrics["perplexity"].append(math.exp(logs["eval_loss"]))
        else:
            self.metrics["eval_loss"].append(None)
            self.metrics["perplexity"].append(None)
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None,
                        ignore_keys=None, metric_key_prefix="eval"):
        output = super().evaluation_loop(
            dataloader, description, prediction_loss_only=True,
            ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix,
        )
        
        # Compute token accuracy
        model.eval()
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()
                preds = shift_logits.argmax(dim=-1)
                mask = shift_labels != -100
                correct = (preds == shift_labels) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
        
        accuracy = total_correct / max(total_tokens, 1)
        output.metrics[f"{metric_key_prefix}_accuracy"] = accuracy
        
        # Update metrics
        step = self.state.global_step
        if step in self.metrics["step"]:
            idx = self.metrics["step"].index(step)
            if idx < len(self.metrics["eval_loss"]):
                self.metrics["eval_loss"][idx] = output.metrics.get(f"{metric_key_prefix}_loss", self.metrics["eval_loss"][idx])
                self.metrics["perplexity"][idx] = math.exp(self.metrics["eval_loss"][idx])
        
        return output

trainer = LoRATrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

print("  Training...")
trainer.train()

# ============================
#  5. SAVE MODEL
# ============================
print("\n[5/6] Saving model...")

# Save LoRA weights
trainer.model.save_pretrained(LORA_OUTPUT_DIR)
tokenizer.save_pretrained(LORA_OUTPUT_DIR)
print(f"  LoRA model saved to: {LORA_OUTPUT_DIR}")

# ============================
#  6. GENERATE EVALUATION REPORT
# ============================
print("\n[6/6] Generating evaluation report...")

# Save metrics CSV
history = trainer.metrics
csv_path = os.path.join(OUTPUT_DIR, "lora_training_metrics.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Step", "Train_Loss", "Val_Loss", "Perplexity", "Learning_Rate"])
    for i, step in enumerate(history["step"]):
        tl = f"{history['train_loss'][i]:.6f}" if i < len(history["train_loss"]) and history["train_loss"][i] is not None else ""
        vl = f"{history['eval_loss'][i]:.6f}" if i < len(history["eval_loss"]) and history["eval_loss"][i] is not None else ""
        ppl = f"{history['perplexity'][i]:.4f}" if i < len(history["perplexity"]) and history["perplexity"][i] is not None else ""
        lr = f"{history['learning_rate'][i]:.2e}" if i < len(history["learning_rate"]) and history["learning_rate"][i] is not None else ""
        w.writerow([step, tl, vl, ppl, lr])
print(f"  Metrics saved to: {csv_path}")

# Build cleaned arrays for plotting
def clean_plot_data(steps, values):
    pairs = [(s, v) for s, v in zip(steps, values) if v is not None]
    return ([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else ([], [])

t_steps, t_loss = clean_plot_data(history["step"], history["train_loss"])
e_steps, e_loss = clean_plot_data(history["step"], history["eval_loss"])
p_steps, ppl = clean_plot_data(history["step"], history["perplexity"])
lr_steps, lr = clean_plot_data(history["step"], history["learning_rate"])

# Generate charts
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Qwen3.5-0.8B-Base LoRA Fine-tuning - Training Metrics", 
             fontsize=16, fontweight="bold")

# Loss curve
ax = axes[0, 0]
if t_steps:
    ax.plot(t_steps, t_loss, "b-", linewidth=2, label="Train Loss", alpha=0.8)
if e_steps:
    ax.plot(e_steps, e_loss, "r-o", linewidth=2, markersize=5, label="Val Loss")
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("Loss Curve (Train & Validation)", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
if e_loss:
    ax.axhline(y=e_loss[-1], color="r", linestyle="--", alpha=0.3, label=f"Final Val Loss: {e_loss[-1]:.4f}")

# Perplexity
ax = axes[0, 1]
if p_steps:
    ax.plot(p_steps, ppl, "g-s", linewidth=2, markersize=5, label="Perplexity")
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Perplexity", fontsize=12)
ax.set_title("Validation Perplexity", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
if ppl:
    ax.axhline(y=ppl[-1], color="g", linestyle="--", alpha=0.3)

# Learning Rate
ax = axes[1, 0]
if lr_steps:
    ax.plot(lr_steps, lr, "m-^", linewidth=2, markersize=5, label="Learning Rate")
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Learning Rate", fontsize=12)
ax.set_title("Learning Rate Schedule", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_yscale("log")

# Summary
ax = axes[1, 1]
ax.axis("off")

initial_loss = t_loss[0] if t_loss else "N/A"
final_train_loss = t_loss[-1] if t_loss else "N/A"
final_val_loss = e_loss[-1] if e_loss else "N/A"
final_ppl = ppl[-1] if ppl else "N/A"
loss_reduction = (initial_loss - final_train_loss) if len(t_loss) > 1 else 0
reduction_pct = (loss_reduction / initial_loss * 100) if initial_loss != "N/A" and len(t_loss) > 1 else 0

summary = (
    "Training Summary\n"
    "================\n\n"
    f"Model:         Qwen3.5-0.8B-Base + LoRA\n"
    f"LoRA Rank:     {LORA_R}\n"
    f"LoRA Alpha:    {LORA_ALPHA}\n"
    f"Data:          {len(split['train'])} train / {len(split['test'])} val\n"
    f"Epochs:        {NUM_EPOCHS}\n"
    f"Total Steps:   {len(t_steps)}\n\n"
    f"Initial Loss:      {initial_loss:.4f}" if isinstance(initial_loss, float) else f"Initial Loss:      {initial_loss}"
)
summary += (
    f"\nFinal Train Loss:  {final_train_loss:.4f}" if isinstance(final_train_loss, float) else f"\nFinal Train Loss:  {final_train_loss}"
)
summary += (
    f"\nFinal Val Loss:    {final_val_loss:.4f}" if isinstance(final_val_loss, float) else f"\nFinal Val Loss:    {final_val_loss}"
)
summary += (
    f"\nFinal Perplexity:  {final_ppl:.2f}" if isinstance(final_ppl, float) else f"\nFinal Perplexity:  {final_ppl}"
)
summary += f"\n\nLoss Reduction:    {loss_reduction:.4f} ({reduction_pct:.1f}%)"

ax.text(0.05, 0.5, summary, fontsize=11, va="center", family="monospace",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.95])
chart_path = os.path.join(OUTPUT_DIR, "lora_training_metrics.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Chart saved to: {chart_path}")

# Generate standalone loss curve
fig2, ax2 = plt.subplots(figsize=(12, 6))
if t_steps:
    ax2.plot(t_steps, t_loss, "b-", linewidth=2.5, label="Train Loss", alpha=0.8)
if e_steps:
    ax2.plot(e_steps, e_loss, "r-o", linewidth=2.5, markersize=6, label="Validation Loss", alpha=0.8)
ax2.set_xlabel("Step", fontsize=13)
ax2.set_ylabel("Loss", fontsize=13)
ax2.set_title("Loss Curve - Qwen3.5-0.8B-Base LoRA Fine-tuning", fontsize=14, fontweight="bold")
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

# Add annotations
if t_steps and t_loss:
    ax2.annotate(f'Start: {t_loss[0]:.4f}', 
                 xy=(t_steps[0], t_loss[0]), 
                 xytext=(t_steps[0]+20, t_loss[0]-0.1),
                 fontsize=10, alpha=0.7)
if e_steps and e_loss:
    ax2.annotate(f'Final: {e_loss[-1]:.4f}', 
                 xy=(e_steps[-1], e_loss[-1]), 
                 xytext=(e_steps[-1]-50, e_loss[-1]+0.1),
                 fontsize=10, alpha=0.7)

fig2.tight_layout()
loss_curve_path = os.path.join(OUTPUT_DIR, "lora_loss_curve.png")
fig2.savefig(loss_curve_path, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Loss curve saved to: {loss_curve_path}")

# Save final metrics as JSON
final_metrics = {
    "model": "Qwen3.5-0.8B-Base + LoRA",
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "train_samples": len(split["train"]),
    "val_samples": len(split["test"]),
    "total_steps": len(t_steps),
    "initial_train_loss": float(t_loss[0]) if t_loss else None,
    "final_train_loss": float(t_loss[-1]) if t_loss else None,
    "final_val_loss": float(e_loss[-1]) if e_loss else None,
    "final_perplexity": float(ppl[-1]) if ppl else None,
    "loss_reduction": float(loss_reduction) if t_loss else None,
    "loss_reduction_percent": float(reduction_pct) if t_loss else None,
    "timestamp": datetime.now().isoformat(),
}
json_path = os.path.join(OUTPUT_DIR, "lora_final_metrics.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(final_metrics, f, indent=2, ensure_ascii=False)
print(f"  Final metrics saved to: {json_path}")

print(f"\n{'='*60}")
print("  TRAINING COMPLETE")
print(f"{'='*60}")
print(f"  Output:     {LORA_OUTPUT_DIR}")
print(f"  Metrics:    lora_training_metrics.csv")
print(f"  Charts:     lora_training_metrics.png, lora_loss_curve.png")
print(f"  Report:     lora_final_metrics.json")

if final_val_loss != "N/A":
    print(f"\n  Final Val Loss:   {final_val_loss:.4f}")
    print(f"  Final Perplexity: {final_ppl:.2f}")
    print(f"  Loss Reduction:   {reduction_pct:.1f}%")
print()
