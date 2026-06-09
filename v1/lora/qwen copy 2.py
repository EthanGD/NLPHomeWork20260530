import os
import torch
import gc
import json
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer, 
    SentenceTransformerTrainer, 
    SentenceTransformerTrainingArguments,
    losses
)
from queryPos import qa_data  # 導入你的數據
# 嵌入模型（Embedding）Sentence Transformers + PEFT 中 sentence_transformers是主流工具（专用且方便）

# ==========================================
# 1. 顯存與設備配置 	設定使用 CUDA 裝置 6，監控顯存使用
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"當前使用設備：{device}")

if torch.cuda.is_available():
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**2
    allocated_mem = torch.cuda.memory_allocated(0) / 1024**2
    reserved_mem = torch.cuda.memory_reserved(0) / 1024**2
    print(f"顯存總量：{total_mem:.2f} MB")
    print(f"PyTorch 已分配：{allocated_mem:.2f} MB")
    print(f"PyTorch 已預留：{reserved_mem:.2f} MB")

# ==========================================
# 2. 數據預處理 從 queryPos.qa_data 讀取 QA 數據，轉換為 anchor/positive 格式
# ==========================================
train_samples = []
for item in qa_data:
    train_samples.append({
        "anchor": item["query"],
        "positive": item["pos"]
    })

train_dataset = Dataset.from_list(train_samples)
print(f"\n數據加載完成，共 {len(train_dataset)} 條樣本")

# ==========================================
# 3. 模型加載 (使用本地緩存)
# ==========================================
model_name = "BAAI/bge-m3"
local_model_path = "/wt/code/models/BAAI/bge-m3"

print(f"\n正在加載模型：{model_name} ...")
if os.path.exists(local_model_path):
    print(f"使用本地模型：{local_model_path}")
    model = SentenceTransformer(
        local_model_path, 
        device=device,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.bfloat16}
    )
else:
    print(f"本地模型不存在，嘗試從 HuggingFace 下載...")
    model = SentenceTransformer(
        model_name, 
        device=device,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.bfloat16},
        cache_dir="./models_cache"
    )

if torch.cuda.is_available():
    print(f"模型加載後顯存使用：{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

# ==========================================
# 4. 損失函數與訓練參數 使用 MultipleNegativesRankingLoss 進行對比學習
# ==========================================
train_loss = losses.MultipleNegativesRankingLoss(model)  # 高效使用正樣本 適合你的數據格式
'''
語義檢索任務的標準選擇

損失函數	適用場景	數據需求
MultipleNegativesRankingLoss	語義檢索/QA	(query, answer) 配對
CosineSimilarityLoss	語義相似度評分	(sentence1, sentence2, score)
TripletLoss	三元組對比	(anchor, positive, negative)
'''


output_dir = "./models/finetuned-lora-bge-m3"

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
    eval_strategy="no",
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,
    load_best_model_at_end=False,
    save_total_limit=2,
    save_steps=100,
    logging_first_step=True,
)

# ==========================================
# 5. 開始訓練 10 個 epoch，batch size=4，gradient accumulation=2
# ==========================================
trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    loss=train_loss,
)

print("\n" + "="*50)
print("開始微調...")
print("="*50 + "\n")

trainer.train()

loss_history = trainer.state.log_history
loss_values = [item['loss'] for item in loss_history if 'loss' in item]

loss_file = os.path.join(output_dir, "loss_history.json")
with open(loss_file, "w", encoding="utf-8") as f:
    json.dump(loss_values, f, ensure_ascii=False, indent=2)

print(f"\n✅ 損失值已保存至：{loss_file}")
print(f"   共記錄 {len(loss_values)} 個 loss 值")

# ==========================================
# 6. 保存模型 模型保存至 ./bge-m3-water-industry
# ==========================================
output_path = output_dir
model.save(output_path)
print(f"\n✅ 模型已保存至：{output_path}")

gc.collect()
torch.cuda.empty_cache()

# ==========================================
# 7. 簡單測試 驗證微調後模型的語義相似度效果
# ==========================================
print("\n" + "="*50)
print("模型測試")
print("="*50)

finetuned_model = SentenceTransformer(output_path, device=device, trust_remote_code=True)

query = "OpenClaw 要錢嗎？"
candidate_1 = "OpenClaw 軟體本身完全免費且開源。"
candidate_2 = "水泵的維護週期通常是三個月。"

embeddings = finetuned_model.encode([query, candidate_1, candidate_2], convert_to_tensor=True)
similarity_1 = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0)
similarity_2 = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[2], dim=0)

print(f"\nQuery: {query}")
print(f"與相關文檔相似度：{similarity_1.item():.4f}")
print(f"與無關文檔相似度：{similarity_2.item():.4f}")

if similarity_1 > similarity_2:
    print("\n✅ 微調生效：相關文檔得分更高")
else:
    print("\n⚠️ 注意：相關文檔得分未高於無關文檔")



import json
import matplotlib.pyplot as plt

with open(os.path.join(output_dir, "loss_history.json"), "r") as f:
    losses = json.load(f)

plt.figure(figsize=(10, 6))
plt.plot(losses, linewidth=2)
plt.xlabel("Step", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.title("Training Loss Curve", fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig("loss_curve.png", dpi=150)
plt.show()


'''
L20
訓練用了1分48秒

'''