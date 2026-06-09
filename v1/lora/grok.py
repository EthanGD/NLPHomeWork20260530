import os
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

# ====================== 指定使用 6號 和 7號 GPU ======================
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

print(f"目前可見 GPU 數量: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)} - "
          f"已用記憶體: {torch.cuda.memory_allocated(i)/1024**2:.1f} MiB")

# ====================== 載入你的資料集 ======================
from queryPos import qa_data
print(f"成功從 queryPos.py 匯入 qa_data，共 {len(qa_data)} 條資料")

train_examples = [
    {
        "query": item["query"],
        "pos": item["pos"]
    }
    for item in qa_data
]

train_dataset = Dataset.from_list(train_examples)
print(f"已準備訓練資料集：{len(train_dataset)} 條正例對")

# ====================== 載入模型並加入 LoRA ======================
model_name = "BAAI/bge-m3"
model = SentenceTransformer(model_name, device="cuda")

peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules="all-linear",
    bias="none",
)

# 使用 add_adapter（Sentence Transformers 官方推薦方式）
model.add_adapter(peft_config)
print("已成功加入 LoRA adapter")

# 自訂函數顯示可訓練參數（因為 SentenceTransformer 沒有 print_trainable_parameters）
def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"trainable params: {trainable_params:,} || "
          f"all params: {all_params:,} || "
          f"trainable%: {100 * trainable_params / all_params:.4f}%")

print_trainable_parameters(model)

# ====================== Loss 與 Training Arguments ======================
train_loss = losses.MultipleNegativesRankingLoss(model=model)

training_args = SentenceTransformerTrainingArguments(
    output_dir="bge-m3-lora-water-domain",
    num_train_epochs=4,
    per_device_train_batch_size=4,           # 保守設定，避免 OOM
    gradient_accumulation_steps=8,           # 有效 batch size ≈ 64
    learning_rate=2e-4,
    warmup_steps=50,
    fp16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="no",
    report_to="none",
)

# ====================== 建立 Trainer 並開始訓練 ======================
trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    loss=train_loss,
)

print("開始訓練...")
trainer.train()

# 儲存 LoRA adapter
model.save_pretrained("bge-m3-lora-water-final")
print("✅ 訓練完成！LoRA adapter 已儲存至 bge-m3-lora-water-final")