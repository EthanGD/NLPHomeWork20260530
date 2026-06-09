import os
import torch
import gc
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer, 
    SentenceTransformerTrainer, 
    SentenceTransformerTrainingArguments,
    losses
)
from queryPos import qa_data  # 導入你的數據

# ==========================================
# 1. 顯存與設備配置
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
# 2. 數據預處理
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
# 3. 模型加載 (使用本地模型路徑)
# ==========================================
# 【關鍵修改】使用本地模型路徑
model_path = "/wt/code/models/BAAI/bge-m3"

print(f"\n正在加載本地模型：{model_path} ...")

# 檢查模型路徑是否存在
if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型路徑不存在：{model_path}\n請先運行 download_model.py 下載模型")

model = SentenceTransformer(
    model_path,  # 使用本地路徑而非模型名稱
    device=device,
    trust_remote_code=True,
    model_kwargs={"torch_dtype": torch.bfloat16}
)

if torch.cuda.is_available():
    print(f"模型加載後顯存使用：{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

# ==========================================
# 4. 損失函數與訓練參數
# ==========================================
train_loss = losses.MultipleNegativesRankingLoss(model)

training_args = SentenceTransformerTrainingArguments(
    output_dir="./bge-m3-finetuned",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_steps=20,
    learning_rate=2e-5,
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="no",
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,
    load_best_model_at_end=False,
    save_total_limit=2,
)

# ==========================================
# 5. 開始訓練
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

# ==========================================
# 6. 保存模型
# ==========================================
output_path = "./bge-m3-water-industry"
model.save(output_path)
print(f"\n✅ 模型已保存至：{output_path}")

gc.collect()
torch.cuda.empty_cache()

# ==========================================
# 7. 簡單測試
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