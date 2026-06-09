import os
import json
import torch
import gc
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 設置字體 (避免中文顯示問題)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer, 
    SentenceTransformerTrainer, 
    SentenceTransformerTrainingArguments,
    losses
)
from queryPos import qa_data

# ==========================================
# 1. Device configuration
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}")

if torch.cuda.is_available():
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"GPU memory: {total_mem:.2f} MB")

# ==========================================
# 2. Data preprocessing and split
# ==========================================
train_samples = []
for item in qa_data:
    train_samples.append({
        "anchor": item["query"],
        "positive": item["pos"]
    })

full_dataset = Dataset.from_list(train_samples)
print(f"Total data: {len(full_dataset)} samples")

# Split train/test set (80/20)
random.seed(42)
indices = list(range(len(full_dataset)))
random.shuffle(indices)

split_idx = int(len(indices) * 0.8)
train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

train_dataset = full_dataset.select(train_indices)
test_dataset = full_dataset.select(test_indices)

print(f"Train set: {len(train_dataset)} samples")
print(f"Test set: {len(test_dataset)} samples\n")

# ==========================================
# 3. Load model
# ==========================================
train_samples = []
for item in qa_data:
    train_samples.append({
        "anchor": item["query"],
        "positive": item["pos"]
    })

full_dataset = Dataset.from_list(train_samples)
print(f"Total data: {len(full_dataset)} samples")

# Split train/test set (80/20)
random.seed(42)
indices = list(range(len(full_dataset)))
random.shuffle(indices)

split_idx = int(len(indices) * 0.8)
train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

train_dataset = full_dataset.select(train_indices)
test_dataset = full_dataset.select(test_indices)

print(f"Train set: {len(train_dataset)} samples")
print(f"Test set: {len(test_dataset)} samples\n")

# ==========================================
# 3. 模型加載
# ==========================================
local_model_path = "/wt/code/models/BAAI/bge-m3"

print(f"Loading model...")
if os.path.exists(local_model_path):
    model = SentenceTransformer(
        local_model_path, 
        device=device,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.bfloat16}
    )
else:
    model = SentenceTransformer(
        "BAAI/bge-m3", 
        device=device,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.bfloat16},
        cache_dir="./models_cache"
    )

# ==========================================
# 4. Loss function and training args
# ==========================================
train_loss = losses.MultipleNegativesRankingLoss(model)

output_dir = "./models/finetuned-bge-m3"

training_args = SentenceTransformerTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_steps=20,
    learning_rate=2e-5,
    fp16=False,
    bf16=True,
    logging_steps=1,
    save_strategy="epoch",
    eval_strategy="epoch",
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    save_total_limit=2,
    logging_first_step=True,
)

# ==========================================
# 5. Training
# ==========================================
trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    loss=train_loss,
)

print("\n" + "="*50)
print("Starting training...")
print("="*50 + "\n")

trainer.train()

# Save Loss history
loss_history = trainer.state.log_history
loss_values = [item['loss'] for item in loss_history if 'loss' in item]
eval_history = [item for item in loss_history if 'eval_loss' in item]

os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "loss_history.json"), "w", encoding="utf-8") as f:
    json.dump(loss_values, f, ensure_ascii=False, indent=2)

print(f"\n✅ Loss saved to: {output_dir}/loss_history.json")
print(f"   Training Loss records: {len(loss_values)}")
print(f"   Evaluation Loss records: {len(eval_history)}")

# 繪製 Loss 曲線圖
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(loss_values) + 1), loss_values, linewidth=2, label='Training Loss', color='#2E86AB')
plt.xlabel('Step', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Curve', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)

# 添加統計信息
plt.annotate(f'Initial: {loss_values[0]:.4f}\nFinal: {loss_values[-1]:.4f}\n↓ {(loss_values[0] - loss_values[-1]) / loss_values[0] * 100:.1f}%',
             xy=(0.02, 0.98), xycoords='axes fraction',
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
loss_curve_path = os.path.join(output_dir, "loss_curve.png")
plt.savefig(loss_curve_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ Loss curve saved to: {loss_curve_path}")

# ==========================================
# 6. 保存模型
# ==========================================
model.save(output_dir)
print(f"\n✅ Model saved to: {output_dir}")

gc.collect()
torch.cuda.empty_cache()

# ==========================================
# 11. Plot evaluation comparison
# ==========================================
# 7. 評估函數
# ==========================================
def evaluate(model, dataset, name):
    """Evaluate model performance on dataset"""
    correct = 0
    sim_positive_list = []
    sim_negative_list = []
    
    print(f"\nEvaluating {name} ...")
    
    for i, item in enumerate(dataset):
        query = item["anchor"]
        positive = item["positive"]
        
        # 構建負樣本
        neg_candidates = [d["positive"] for d in dataset if d["anchor"] != query]
        if len(neg_candidates) > 0:
            negative = neg_candidates[i % len(neg_candidates)]
        else:
            negative = "這是一個無關的測試文本。"
        
        embeddings = model.encode([query, positive, negative], convert_to_tensor=True)
        sim_pos = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0).item()
        sim_neg = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[2], dim=0).item()
        
        sim_positive_list.append(sim_pos)
        sim_negative_list.append(sim_neg)
        
        if sim_pos > sim_neg:
            correct += 1
    
    accuracy = correct / len(dataset)
    avg_margin = torch.mean(torch.tensor(sim_positive_list) - torch.tensor(sim_negative_list)).item()
    
    print(f"  準確率：{accuracy:.2%} ({correct}/{len(dataset)})")
    print(f"  正樣本相似度：{torch.mean(torch.tensor(sim_positive_list)).item():.4f}")
    print(f"  負樣本相似度：{torch.mean(torch.tensor(sim_negative_list)).item():.4f}")
    print(f"  平均邊際：{avg_margin:.4f}")
    
    return {
        "name": name,
        "accuracy": accuracy,
        "avg_sim_positive": torch.mean(torch.tensor(sim_positive_list)).item(),
        "avg_sim_negative": torch.mean(torch.tensor(sim_negative_list)).item(),
        "avg_margin": avg_margin
    }

# ==========================================
# 8. Train/Test evaluation
# ==========================================
print("\n" + "="*50)
print("Model Evaluation")
print("="*50)

train_results = evaluate(trainer.model, train_dataset, "Train")
test_results = evaluate(trainer.model, test_dataset, "Test")

# ==========================================
# 9. Save evaluation report
# ==========================================
report = {
    "train_set_size": len(train_dataset),
    "test_set_size": len(test_dataset),
    "train_results": train_results,
    "test_results": test_results,
    "overfitting_check": {
        "accuracy_gap": train_results["accuracy"] - test_results["accuracy"],
        "margin_gap": train_results["avg_margin"] - test_results["avg_margin"]
    }
}

report_path = os.path.join(output_dir, "evaluation_report.json")
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n✅ Evaluation report saved to: {report_path}")

# ==========================================
# 10. Evaluation results comparison table
# ==========================================
print("\n" + "="*60)
print("Evaluation Results Comparison")
print("="*60)

print(f"\n{'Metric':<20} {'Train':<15} {'Test':<15} {'Gap':<10}")
print("-"*60)
print(f"{'Accuracy':<20} {train_results['accuracy']:<15.2%} {test_results['accuracy']:<15.2%} {train_results['accuracy'] - test_results['accuracy']:+.2%}")
print(f"{'Positive Sim':<20} {train_results['avg_sim_positive']:<15.4f} {test_results['avg_sim_positive']:<15.4f} {train_results['avg_sim_positive'] - test_results['avg_sim_positive']:+.4f}")
print(f"{'Negative Sim':<20} {train_results['avg_sim_negative']:<15.4f} {test_results['avg_sim_negative']:<15.4f} {train_results['avg_sim_negative'] - test_results['avg_sim_negative']:+.4f}")
print(f"{'Average Margin':<20} {train_results['avg_margin']:<15.4f} {test_results['avg_margin']:<15.4f} {train_results['avg_margin'] - test_results['avg_margin']:+.4f}")

# ==========================================
# 12. Overfitting check and summary
# ==========================================
print("\n" + "="*60)
print("Overfitting Check")
print("="*60)

accuracy_gap = train_results["accuracy"] - test_results["accuracy"]
margin_gap = train_results["avg_margin"] - test_results["avg_margin"]

print(f"\nAccuracy gap: {accuracy_gap:.2%}")
print(f"Margin gap: {margin_gap:.4f}")

if accuracy_gap > 0.1:
    print("\n⚠️ Warning: Train accuracy significantly higher than test, possible overfitting")
elif accuracy_gap < 0.02:
    print("\n✅ Good: Train and test performance consistent, good generalization")
else:
    print("\n✓ Normal: Slight gap within reasonable range")

# Fine-tuning summary
print("\n" + "="*60)
print("Fine-tuning Summary")
print("="*60)

if test_results["avg_margin"] > 0.3:
    print("\n✅ Fine-tuning successful: Margin reached good level")
else:
    print("\n⚠️ Suggestion: Continue optimizing, add more data or tune hyperparameters")

# ==========================================
# 11. Plot evaluation comparison
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Accuracy comparison
axes[0, 0].bar(['Train', 'Test'], 
            [train_results["accuracy"] * 100, test_results["accuracy"] * 100],
            color=['#2E86AB', '#E94F37'], alpha=0.8)
axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12)
axes[0, 0].set_title('Accuracy Comparison', fontsize=13)
axes[0, 0].set_ylim(0, 100)
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate([train_results["accuracy"] * 100, test_results["accuracy"] * 100]):
    axes[0, 0].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')

# Positive similarity comparison
axes[0, 1].bar(['Train', 'Test'], 
            [train_results["avg_sim_positive"], test_results["avg_sim_positive"]],
            color=['#2E86AB', '#E94F37'], alpha=0.8)
axes[0, 1].set_ylabel('Positive Similarity', fontsize=12)
axes[0, 1].set_title('Positive Similarity Comparison', fontsize=13)
axes[0, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate([train_results["avg_sim_positive"], test_results["avg_sim_positive"]]):
    axes[0, 1].text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')

# Negative similarity comparison
axes[1, 0].bar(['Train', 'Test'], 
            [train_results["avg_sim_negative"], test_results["avg_sim_negative"]],
            color=['#2E86AB', '#E94F37'], alpha=0.8)
axes[1, 0].set_ylabel('Negative Similarity', fontsize=12)
axes[1, 0].set_title('Negative Similarity Comparison', fontsize=13)
axes[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate([train_results["avg_sim_negative"], test_results["avg_sim_negative"]]):
    axes[1, 0].text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')

# Margin comparison
axes[1, 1].bar(['Train', 'Test'], 
            [train_results["avg_margin"], test_results["avg_margin"]],
            color=['#2E86AB', '#E94F37'], alpha=0.8)
axes[1, 1].set_ylabel('Average Margin', fontsize=12)
axes[1, 1].set_title('Margin Comparison', fontsize=13)
axes[1, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate([train_results["avg_margin"], test_results["avg_margin"]]):
    axes[1, 1].text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
eval_curve_path = os.path.join(output_dir, "evaluation_comparison.png")
plt.savefig(eval_curve_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ Evaluation comparison saved to: {eval_curve_path}")

# ==========================================
# 13. Model test
# ==========================================
print("\n" + "="*50)
print("Model Test")
print("="*50)

finetuned_model = SentenceTransformer(output_dir, device=device, trust_remote_code=True)

query = "OpenClaw 要錢嗎？"
candidate_1 = "OpenClaw 軟體本身完全免費且開源。"
candidate_2 = "水泵的維護週期通常是三個月。"

embeddings = finetuned_model.encode([query, candidate_1, candidate_2], convert_to_tensor=True)
similarity_1 = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0)
similarity_2 = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[2], dim=0)

print(f"\nQuery: {query}")
print(f"Similarity to relevant doc: {similarity_1.item():.4f}")
print(f"Similarity to irrelevant doc: {similarity_2.item():.4f}")

if similarity_1 > similarity_2:
    print("\n✅ Fine-tuning effective: Relevant doc scored higher")
else:
    print("\n⚠️ Note: Relevant doc did not score higher than irrelevant doc")