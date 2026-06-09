# 中医针灸 LoRA 微调

## 概述

使用 Qwen3.5 API 生成中医针灸问答对，并对 Qwen3.5-0.8B-Base 进行 LoRA 微调。

## 文件结构

```
v4/
├── generate_qa_pairs.py    # 生成问答对脚本
├── train_lora.py           # LoRA 微调脚本
├── run_lora_finetune.py    # 一键运行脚本
├── Acupuncture/            # 中医书籍数据
│   ├── 针灸甲乙经.md
│   ├── 黄帝内经_灵枢.md
│   ├── 黄帝内经_素问.md
│   ├── 经络.md
│   ├── 腧穴.md
│   └── 十二经脉_动画图.md
├── models/                 # Qwen3.5-0.8B-Base 模型
└── results/                # 输出结果
    ├── lora_model/         # LoRA 微调模型
    ├── lora_training_metrics.csv
    ├── lora_training_metrics.png
    ├── lora_loss_curve.png
    └── lora_final_metrics.json
```

## 使用方法

### 方法 1: 一键运行

```bash
python run_lora_finetune.py
```

### 方法 2: 分步运行

#### 步骤 1: 生成问答对

```bash
python generate_qa_pairs.py
```

使用 `https://llm.ethanchenyansong.com/llm/v1` API 从中医书籍生成问答对。

#### 步骤 2: LoRA 微调

```bash
python train_lora.py
```

## 配置参数

### 问答对生成 (generate_qa_pairs.py)
- API URL: `https://llm.ethanchenyansong.com/llm/v1/chat/completions`
- 模型：Qwen3.5
- 文本块大小：3000 字符
- 重叠：500 字符

### LoRA 微调 (train_lora.py)
- 基础模型：`models/` (Qwen3.5-0.8B-Base)
- LoRA Rank: 8
- LoRA Alpha: 16
- 序列长度：512
- Batch Size: 4
- 学习率：1e-4
- Epochs: 3
- GPU: cuda:6

## 输出指标

### 训练指标
- **Train Loss**: 训练损失
- **Val Loss**: 验证损失
- **Perplexity**: 困惑度
- **Learning Rate**: 学习率

### 输出文件
1. `lora_training_metrics.csv` - 详细训练日志
2. `lora_training_metrics.png` - 四合一评估图表
3. `lora_loss_curve.png` - 损失曲线图
4. `lora_final_metrics.json` - 最终评估报告

## 评估指标说明

| 指标 | 说明 |
|------|------|
| Train Loss | 训练集上的平均损失，越低越好 |
| Val Loss | 验证集上的平均损失，用于检测过拟合 |
| Perplexity | 困惑度，衡量模型对数据的预测能力，越低越好 |
| Learning Rate | 学习率调度，用于观察训练动态 |

## 推理使用 LoRA 模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained("models", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("models", trust_remote_code=True)

# 加载 LoRA 权重
model = PeftModel.from_pretrained(model, "results/lora_model")

# 推理
prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n什么是针灸？<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```
