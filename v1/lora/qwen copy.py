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
# 3. 模型加載 (不使用 PEFT) 加載 BAAI/bge-m3 模型，使用 bfloat16 精度
# ==========================================
model_name = "BAAI/bge-m3"

print(f"\n正在加載模型：{model_name} ...")
model = SentenceTransformer(
    model_name, 
    device=device,
    trust_remote_code=True,
    model_kwargs={"torch_dtype": torch.bfloat16}
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


training_args = SentenceTransformerTrainingArguments(
    output_dir="./bge-m3-finetuned",
    num_train_epochs=10, # 微調任務不需要太多輪次，避免過擬合
    per_device_train_batch_size=4, # BGE-M3 模型較大，batch 太大會 OOM
    gradient_accumulation_steps=2, # 等效 batch size = 4×2 = 8，平衡顯存與穩定性
    warmup_steps=20, # 前 20 步緩慢增加學習率，避免初期梯度爆炸
    learning_rate=2e-5, # 微調大型語言模型通常使用較小的學習率 微調常用學習率，不會破壞預訓練知識
    fp16=False, # BGE-M3 模型在某些環境下可能不完全支持 fp16，改用 bfloat16
    bf16=True, # bfloat16 比 fp16 更穩定，適合 Transformer 模型
    logging_steps=10,
    save_strategy="epoch", # 每個 epoch 結束保存一次模型，方便後續選擇最佳模型
    eval_strategy="no",
    report_to="none",
    remove_unused_columns=False,
    dataloader_num_workers=0,
    load_best_model_at_end=False,
    save_total_limit=2, # 最多保留 2 個模型檔案，節省磁碟空間
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

# ==========================================
# 6. 保存模型 模型保存至 ./bge-m3-water-industry
# ==========================================
output_path = "./bge-m3-water-industry"
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