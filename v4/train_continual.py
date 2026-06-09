#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3.5-0.8B-Base 领域继续预训练 (Continual Pre-training)
数据: 中医经典文本 (.md 格式)
评估指标: Loss, Perplexity (困惑度), Token Accuracy (Token预测准确率)

注意: 因数据为原始纯文本（非问答对），此为继续预训练而非SFT。
     準確率/召回率/精確率/F1 为分类指标，不适用生成式LM预训练。
     替代使用: Loss, Perplexity, Token Accuracy。
"""

import os
import sys
import glob
import math
import csv
import re
import json
import warnings
import traceback
from datetime import datetime

warnings.filterwarnings("ignore")

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)

# ============================
#  CONFIGURATION
#  (all paths relative to script location)
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "Acupuncture")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# CUDA device selection (set to "auto" or specific device like "cuda:6")
CUDA_DEVICE = "cuda:6"   # Use GPU 6 (L20, 48GB)

# L20 48GB - full fine-tuning 0.75B with full attention layers needs conservative settings
MAX_SEQ_LENGTH = 512     # full attention layers are O(n^2), 512 is safe for L20
STRIDE = 256
BATCH_SIZE = 1           # keep small to avoid OOM
GRAD_ACCUM = 8           # compensate batch via gradient accumulation (effective batch = 8)
LEARNING_RATE = 5e-5
NUM_EPOCHS = 10
WARMUP_STEPS = 50
TEST_SPLIT = 0.1
LOGGING_STEPS = 10
EVAL_STEPS = 50
SAVE_STEPS = 200
SEED = 42

set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "final_model"), exist_ok=True)

# Set CUDA device and memory optimization
if CUDA_DEVICE and CUDA_DEVICE != "auto":
    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE.split(":")[-1]
    print(f"Using CUDA device: {CUDA_DEVICE}")
else:
    print(f"Using default CUDA device (auto)")

# Memory optimization for L20
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

print(f"Base dir: {BASE_DIR}")
print(f"Model:    {MODEL_PATH}")
print(f"Data:     {DATA_DIR}")
print(f"Output:   {OUTPUT_DIR}")

# ============================
#  1. LOAD TOKENIZER
# ============================
print("[1/6] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"  Vocab size: {tokenizer.vocab_size}")
print(f"  Pad token: {tokenizer.pad_token}")
print(f"  EOS token: {tokenizer.eos_token}")

# ============================
#  2. LOAD & PROCESS DATA
# ============================
print("[2/6] Loading data...")

def load_md_files(data_dir):
    texts = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        fname = os.path.basename(fp)
        with open(fp, "r", encoding="utf-8") as f:
            content = f.read()
        content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
        content = re.sub(r'\*{1,3}', '', content)
        content = content.strip()
        if content:
            texts.append(content)
            print(f"  Loaded: {fname} ({len(content)} chars)")
    return texts

raw_texts = load_md_files(DATA_DIR)
total_chars = sum(len(t) for t in raw_texts)
print(f"  Total: {len(raw_texts)} files, {total_chars:,} chars")

# ============================
#  3. TOKENIZE & CHUNK
# ============================
print("[3/6] Tokenizing & chunking...")

def chunk_text(text, tokenizer, max_length, stride):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i + max_length]
        if len(chunk) >= max_length // 2:
            if len(chunk) < max_length:
                chunk = chunk + [tokenizer.pad_token_id] * (max_length - len(chunk))
            chunks.append(chunk[:max_length])
    return chunks

all_chunks = []
for text in raw_texts:
    all_chunks.extend(chunk_text(text, tokenizer, MAX_SEQ_LENGTH, STRIDE))

print(f"  Total chunks: {len(all_chunks)}")

dataset = Dataset.from_dict({
    "input_ids": all_chunks,
    "labels": all_chunks.copy(),
})
split = dataset.train_test_split(test_size=TEST_SPLIT, seed=SEED)
print(f"  Train: {len(split['train'])} | Val: {len(split['test'])}")

# Log a sample
sample_ids = split["train"][0]["input_ids"]
sample_text = tokenizer.decode(sample_ids, skip_special_tokens=True)
print(f"  Sample chunk: {sample_text[:80]}...")

# ============================
#  4. LOAD MODEL
# ============================
print("[4/6] Loading model...")

model = None
load_errors = []

# Strategy 1: Try AutoModelForCausalLM
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=False,
    )
    print("  Loaded via AutoModelForCausalLM")
except Exception as e:
    load_errors.append(f"AutoModelForCausalLM: {e}")

# Strategy 2: Load full multimodal model, use text backbone
if model is None:
    try:
        full = AutoModel.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        model = full.model
        print("  Loaded via AutoModel.model (text backbone)")
    except Exception as e:
        load_errors.append(f"AutoModel.model: {e}")

# Strategy 3: Try with trust_remote_code
if model is None:
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        print("  Loaded via AutoModelForCausalLM (trust_remote_code=True)")
    except Exception as e:
        load_errors.append(f"AutoModelForCausalLM (trust): {e}")

if model is None:
    print("FATAL: Could not load model!")
    for err in load_errors:
        print(f"  - {err}")
    sys.exit(1)

model.config.use_cache = False
n_params = sum(p.numel() for p in model.parameters())
print(f"  Parameters: {n_params / 1e9:.2f}B")
print(f"  Device: {model.device}")
print(f"  Dtype: {model.dtype}")

# ============================
#  5. TRAINING
# ============================
print("[5/6] Starting training...")

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
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
    bf16=torch.cuda.is_available(),
    fp16=False,
    dataloader_drop_last=False,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    seed=SEED,
    dataloader_num_workers=0,
    logging_first_step=True,
    # save_safetensors=True,  # removed in transformers 5.x
)


class CLMMetricsTrainer(Trainer):
    """Custom Trainer with per-batch accuracy (avoids storing full logits, which is OOM for vocab_size=248k)."""

    def __init__(self, *args, **kwargs):
        # do NOT pass compute_metrics to parent — we compute accuracy in evaluation_loop
        kwargs.pop("compute_metrics", None)
        super().__init__(*args, **kwargs)
        self.metrics = {
            "step": [],
            "train_loss": [],
            "eval_loss": [],
            "perplexity": [],
            "token_accuracy": [],
        }

    def log(self, logs, *args, **kwargs):
        super().log(logs, *args, **kwargs)
        step = self.state.global_step
        if "loss" in logs:
            self.metrics["step"].append(step)
            self.metrics["train_loss"].append(logs["loss"])
        if "eval_loss" in logs:
            if step not in self.metrics["step"]:
                self.metrics["step"].append(step)
                self.metrics["train_loss"].append(None)
            idx = self.metrics["step"].index(step)
            while len(self.metrics["eval_loss"]) <= idx:
                self.metrics["eval_loss"].append(None)
            self.metrics["eval_loss"][idx] = logs["eval_loss"]
            while len(self.metrics["perplexity"]) <= idx:
                self.metrics["perplexity"].append(None)
            self.metrics["perplexity"][idx] = math.exp(logs["eval_loss"])

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None,
                        ignore_keys=None, metric_key_prefix="eval"):
        """Override to compute token accuracy per batch without storing full logits."""

        # First pass: compute loss only (memory efficient)
        prediction_loss_only = True
        output = super().evaluation_loop(
            dataloader, description, prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix,
        )

        # Second pass: compute token accuracy batch-by-batch (without storing logits)
        model = self.model
        model.eval()
        total_correct = 0
        total_tokens = 0
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                logits = model(**batch).logits  # [batch, seq, vocab]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()
                preds = shift_logits.argmax(dim=-1)
                mask = shift_labels != -100
                correct = (preds == shift_labels) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
                del logits, shift_logits, preds, correct, mask
                torch.cuda.empty_cache()

        accuracy = total_correct / max(total_tokens, 1)
        output.metrics[f"{metric_key_prefix}_accuracy"] = accuracy

        # Store in metrics history
        step = self.state.global_step
        if step in self.metrics["step"]:
            idx = self.metrics["step"].index(step)
            while len(self.metrics["token_accuracy"]) <= idx:
                self.metrics["token_accuracy"].append(None)
            self.metrics["token_accuracy"][idx] = accuracy

        return output


trainer = CLMMetricsTrainer(
    model=model,
    args=training_args,
    data_collator=collator,
    train_dataset=split["train"],
    eval_dataset=split["test"],
)

print("  Training...")
trainer.train()

# Parse eval accuracy from logs
for log in trainer.state.log_history:
    if "eval_accuracy" in log:
        step = log.get("step", log.get("epoch", 0))
        acc = log["eval_accuracy"]
        if step in trainer.metrics["step"]:
            idx = trainer.metrics["step"].index(step)
            while len(trainer.metrics["token_accuracy"]) <= idx:
                trainer.metrics["token_accuracy"].append(None)
            trainer.metrics["token_accuracy"][idx] = acc
        elif isinstance(step, float):
            approx_step = int(step * len(trainer.metrics["step"]) / NUM_EPOCHS) if NUM_EPOCHS else 0
            if approx_step < len(trainer.metrics["step"]) and len(trainer.metrics["token_accuracy"]) <= approx_step:
                while len(trainer.metrics["token_accuracy"]) <= approx_step:
                    trainer.metrics["token_accuracy"].append(None)
                trainer.metrics["token_accuracy"][approx_step] = acc

# ============================
#  6. SAVE & EVALUATE
# ============================
print("[6/6] Saving results & generating charts...")

model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
print("  Model saved to final_model/")

# Save metrics CSV
history = trainer.metrics
csv_path = os.path.join(OUTPUT_DIR, "training_metrics.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Step", "Train_Loss", "Val_Loss", "Perplexity", "Token_Accuracy"])
    for i, step in enumerate(history["step"]):
        tl = f"{history['train_loss'][i]:.6f}" if i < len(history["train_loss"]) and history["train_loss"][i] is not None else ""
        vl = f"{history['eval_loss'][i]:.6f}" if i < len(history["eval_loss"]) and history["eval_loss"][i] is not None else ""
        ppl = f"{history['perplexity'][i]:.4f}" if i < len(history["perplexity"]) and history["perplexity"][i] is not None else ""
        acc = f"{history['token_accuracy'][i]:.4f}" if i < len(history["token_accuracy"]) and history["token_accuracy"][i] is not None else ""
        w.writerow([step, tl, vl, ppl, acc])
print(f"  Metrics saved to {csv_path}")

# Build cleaned arrays for plotting
def clean_plot_data(steps, values):
    pairs = [(s, v) for s, v in zip(steps, values) if v is not None]
    return ([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else ([], [])

t_steps, t_loss = clean_plot_data(history["step"], history["train_loss"])
e_steps, e_loss = clean_plot_data(history["step"], history["eval_loss"])
p_steps, ppl = clean_plot_data(history["step"], history["perplexity"])
a_steps, acc = clean_plot_data(history["step"], history["token_accuracy"])

# Generate chart
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Qwen3.5-0.8B-Base Continual Pre-training — Training Metrics", fontsize=15, fontweight="bold")

# Loss curve
ax = axes[0, 0]
if t_steps:
    ax.plot(t_steps, t_loss, "b-", linewidth=1.5, label="Train Loss", alpha=0.8)
if e_steps:
    ax.plot(e_steps, e_loss, "r-o", linewidth=1.5, markersize=4, label="Val Loss")
ax.set_xlabel("Step", fontsize=11)
ax.set_ylabel("Loss", fontsize=11)
ax.set_title("Loss Curve", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
if e_loss:
    ax.axhline(y=e_loss[-1], color="r", linestyle="--", alpha=0.3)

# Perplexity
ax = axes[0, 1]
if p_steps:
    ax.plot(p_steps, ppl, "g-s", linewidth=1.5, markersize=4, label="Perplexity")
ax.set_xlabel("Step", fontsize=11)
ax.set_ylabel("Perplexity", fontsize=11)
ax.set_title("Validation Perplexity (PPL)", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
if ppl:
    ax.axhline(y=ppl[-1], color="g", linestyle="--", alpha=0.3)

# Token Accuracy
ax = axes[1, 0]
if a_steps and any(v is not None for v in acc):
    ax.plot(a_steps, acc, "m-^", linewidth=1.5, markersize=4, label="Token Accuracy")
ax.set_xlabel("Step", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("Token Prediction Accuracy", fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Summary text
ax = axes[1, 1]
ax.axis("off")
final_train_loss = t_loss[-1] if t_loss else "N/A"
final_val_loss = e_loss[-1] if e_loss else "N/A"
final_ppl = ppl[-1] if ppl else "N/A"
final_acc = acc[-1] if acc else "N/A"
improvement = (t_loss[0] - t_loss[-1]) if len(t_loss) > 1 else 0

summary = (
    "Training Summary\n"
    "================\n\n"
    f"Model:       Qwen3.5-0.8B-Base\n"
    f"Data files:  {len(raw_texts)}\n"
    f"Chunks:      {len(all_chunks)}\n"
    f"Train/Val:   {len(split['train'])} / {len(split['test'])}\n"
    f"Epochs:      {NUM_EPOCHS}\n"
    f"Steps:       {len(t_steps)}\n\n"
    f"Initial Loss:   {t_loss[0]:.4f}" if t_loss else ""
)
summary += (
    f"\nFinal Train Loss: {final_train_loss:.4f}\n"
    f"Final Val Loss:   {final_val_loss:.4f}\n"
    f"Final Perplexity: {final_ppl:.2f}\n"
    f"Final Token Acc:  {final_acc:.4f}\n\n"
    f"Loss reduction:   {improvement:.4f}"
)
ax.text(0.05, 0.5, summary, fontsize=10.5, va="center", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.95])
chart_path = os.path.join(OUTPUT_DIR, "training_metrics.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Chart saved to {chart_path}")

# Also save a standalone loss curve with more detail
fig2, ax2 = plt.subplots(figsize=(10, 6))
if t_steps:
    ax2.plot(t_steps, t_loss, "b-", linewidth=1.5, label="Train Loss")
if e_steps:
    ax2.plot(e_steps, e_loss, "r-o", linewidth=1.5, markersize=4, label="Validation Loss")
ax2.set_xlabel("Step", fontsize=12)
ax2.set_ylabel("Loss", fontsize=12)
ax2.set_title("Loss Curve — Qwen3.5-0.8B-Base Domain Continual Pre-training", fontsize=13, fontweight="bold")
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"  Loss curve saved to loss_curve.png")

# Save final metrics as JSON
final_metrics = {
    "model": "Qwen3.5-0.8B-Base",
    "data_files": len(raw_texts),
    "data_chunks": len(all_chunks),
    "train_samples": len(split["train"]),
    "val_samples": len(split["test"]),
    "total_steps": len(t_steps),
    "initial_train_loss": float(t_loss[0]) if t_loss else None,
    "final_train_loss": float(t_loss[-1]) if t_loss else None,
    "final_val_loss": float(e_loss[-1]) if e_loss else None,
    "final_perplexity": float(ppl[-1]) if ppl else None,
    "final_token_accuracy": float(acc[-1]) if acc else None,
    "loss_improvement": float(improvement) if t_loss else None,
}
json_path = os.path.join(OUTPUT_DIR, "final_metrics.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(final_metrics, f, indent=2, ensure_ascii=False)
print(f"  Final metrics saved to final_metrics.json")

print(f"\n{'='*55}")
print("  TRAINING COMPLETE")
print(f"{'='*55}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Metrics: training_metrics.csv, final_metrics.json")
print(f"  Charts:  training_metrics.png, loss_curve.png")
print(f"  Model:   final_model/")

if final_ppl != "N/A":
    print(f"\n  Best Val Perplexity: {final_ppl:.2f}")
    print(f"  Best Token Accuracy: {final_acc:.4f}")
print()
