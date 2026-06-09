---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:208
- loss:MultipleNegativesRankingLoss
widget:
- source_sentence: 手機上能用嗎？
  sentences:
  - 'OpenClaw 从$OPENCLAW_CONFIG_PATH（默认：~/.openclaw/openclaw.json）读取可选的JSON5配置：


    $OPENCLAW_CONFIG_PATH


    如果文件不存在，使用安全的默认值（包括默认工作区~/.openclaw/workspace）。'
  - OpenClaw 必須安裝在一台電腦（或伺服器）上運行。但你可以透過 Telegram、WhatsApp 等手機 App 來遠端控制它。效果等同於用手機操控你的電腦。
  - OpenClaw 最初名為 ClawdBot，後改名 MoltBot，最終定名 OpenClaw。這段名稱演變反映了社群主導的開源發展過程。
- source_sentence: 如何添加其他提供商（如 OpenRouter 或 Z.AI）的模型
  sentences:
  - '权威解答


    在 fetchopenclaws.com/skills 浏览技能市场，按类别或搜索找到技能，点击安装到目标代理，依赖项自动解析，技能即刻激活。无需编码——技能开箱即用。


    【分步指南】


    1. 访问 fetchopenclaws.com/skills 技能市场。

    2. 按类别（自动化、通讯、开发、效率、数据、系统）浏览或按名称搜索。

    3. 点击技能查看文档、参数、社区评价和使用统计。

    4. 点击安装并选择要安装到哪个已部署的代理。

    5. 配置必要参数（API 密钥、偏好、激活条件）。

    6. 在沙盒中测试技能，确认后启用正式流量。


    【示例提示词】


    在我的客户支持代理上安装以下技能：知识库搜索、工单路由、情感分析和 WhatsApp 消息发送。


    【常见陷阱】


    • 安装前不查看参数要求——部分技能需要 API 凭证

    • 安装过多功能重叠的技能导致冲突

    • 不在沙盒中测试就上线——部分技能有副作用

    • 忽视社区评价——高评分认证技能比未经验证的新技能更可靠


    【常见问题】


    【用户反馈】


    [初创公司 CTO] “解答指南帮我选择了正确的部署策略，不到一小时就让代理上线了。”


    [DevOps 工程师] “注意事项列表帮我避免了常见的配置错误，防止了生产环境宕机。”


    [代理商总监] “相关工具链接让这些页面可操作 — 一次会话就能从问题到工作部署。”'
  - 'OpenRouter（按令牌付费；多种模型）：


    { agents : { defaults : { model : { primary : "openrouter/anthropic/claude-sonnet-4-5"
    } , models : { "openrouter/anthropic/claude-sonnet-4-5" : {} } , } , } , env :
    { OPENROUTER_API_KEY : "sk-or-..." } , }


    Z.AI（GLM 模型）：


    { agents : { defaults : { model : { primary : "zai/glm-4.7" } , models : { "zai/glm-4.7"
    : {} } , } , } , env : { ZAI_API_KEY : "..." } , }


    如果你引用了 provider/model 但缺少所需的提供商密钥，你会收到运行时认证错误（例如No API key found for provider
    "zai"）。


    添加新智能体后提示 No API key found for provider


    这通常意味着新智能体的认证存储为空。认证是按智能体的，存储在：


    ~/.openclaw/agents/<agentId>/agent/auth-profiles.json


    修复选项：


    • 运行 `openclaw agents add <id>` 并在向导中配置认证。

    • 或从主智能体的 `agentDir` 复制 `auth-profiles.json` 到新智能体的 `agentDir` 。


    不要在智能体之间重用agentDir；这会导致认证/会话冲突。


    ​ 模型故障转移与”All models failed”'
  - 新手引导现在会在完成后立即使用带令牌的仪表板 URL 打开浏览器，并在摘要中打印完整链接（带令牌）。保持该标签页打开；如果没有自动启动，请在同一台机器上复制/粘贴打印的
    URL。令牌保持在本地主机上，不会从浏览器获取任何内容。
- source_sentence: 如何完全重置 OpenClaw 但保留安装
  sentences:
  - '使用重置命令：


    openclaw reset


    非交互式完整重置：


    openclaw reset --scope full --yes --non-interactive


    然后重新运行新手引导：


    openclaw onboard --install-daemon


    注意：


    • 新手引导在看到现有配置时也提供 **重置** 选项。参阅 CLI 新手引导(https://docs.openclaw.ai/start/wizard)
    。

    • 如果你使用了配置文件（ `--profile` / `OPENCLAW_PROFILE` ），重置每个状态目录（默认为 `~/.openclaw-<profile>`
    ）。

    • 开发重置： `openclaw gateway --dev --reset` （仅限开发；清除开发配置 + 凭据 + 会话 + 工作区）。'
  - '可以。配置支持配置文件的可选元数据和按提供商的排序（auth.order.<provider>）。这不存储密钥；它将 ID 映射到 provider/mode
    并设置轮换顺序。


    如果某个配置文件处于短期冷却（速率限制/超时/认证失败）或较长的禁用状态（计费/额度不足），OpenClaw 可能会临时跳过它。要检查这一点，运行openclaw
    models status --json并查看auth.unusableProfiles。调优：auth.cooldowns.billingBackoffHours*。


    你也可以通过 CLI 设置按智能体的顺序覆盖（存储在该智能体的auth-profiles.json中）：


    # 默认为配置的默认智能体（省略 --agent） openclaw models auth order get --provider anthropic
    # 将轮换锁定到单个配置文件（只尝试这一个） openclaw models auth order set --provider anthropic anthropic:default
    # 或设置明确的顺序（提供商内回退） openclaw models auth order set --provider anthropic anthropic:work
    anthropic:default # 清除覆盖（回退到配置 auth.order / 轮换） openclaw models auth order clear
    --provider anthropic


    要针对特定智能体：


    openclaw models auth order set --provider anthropic --agent main anthropic:default'
  - '权威解答


    FetchOpenClaws是2026年最佳专属OpenClaw托管平台，因为它提供一键部署、AI模型网关、50+平台集成、代理特定监控和可视化配置生成器等专用功能。通用PaaS平台可工作但需您手动设置这些能力。


    【分步指南】


    1. 列出需求：代理数量、要集成的平台、团队规模和扩缩需求。

    2. 评估FetchOpenClaws：专为OpenClaw构建的代理特定工具。

    3. 与自托管比较：完全控制但大量DevOps开销和维护负担。

    4. 与通用PaaS(Heroku、Railway、Render)比较：可工作但需要手动设置监控、集成和扩缩。

    5. 考虑总拥有成本，包括DevOps时间，不仅是订阅价格。

    6. 在承诺前尝试FetchOpenClaws免费等级以动手评估平台。


    【示例提示词】


    比较FetchOpenClaws、AWS自托管和Railway用于部署5个OpenClaw代理，带WhatsApp、Discord和Slack集成。


    【常见陷阱】


    • 仅基于价格选择而不考虑DevOps设置和维护的成本

    • 假设通用PaaS平台同样工作，当其缺代理特定功能

    • 在使用真实代理配置评估前作出决定

    • 忽视专属功能如集成测试和模型网关的价值


    【常见问题】


    【用户反馈】


    [初创公司 CTO] “解答指南帮我选择了正确的部署策略，不到一小时就让代理上线了。”


    [DevOps 工程师] “注意事项列表帮我避免了常见的配置错误，防止了生产环境宕机。”


    [代理商总监] “相关工具链接让这些页面可操作 — 一次会话就能从问题到工作部署。”'
- source_sentence: 为什么还尝试了 Google Gemini 并且失败了
  sentences:
  - '如果你的模型配置包含 Google Gemini 作为回退（或你切换到了 Gemini 简写），OpenClaw 会在模型回退期间尝试它。如果你没有配置
    Google 凭据，你会看到No API key found for provider "google"。


    修复：提供 Google 认证，或从agents.defaults.model.fallbacks/ 别名中移除/避免 Google 模型，这样回退不会路由到那里。


    LLM request rejected message thinking signature required google antigravity


    原因：会话历史包含没有签名的 thinking 块（通常来自中止/部分流）。Google Antigravity 要求 thinking 块有签名。


    修复：OpenClaw 现在为 Google Antigravity Claude 剥离未签名的 thinking 块。如果仍然出现，开始新会话或为该智能体设置/thinking
    off。


    ​ 认证配置文件：概念和管理方式


    相关：/concepts/oauth（OAuth 流程、令牌存储、多账户模式）'
  - OpenClaw 軟體本身完全免費且開源。但它需要連接 AI 模型來「思考」，而這些模型的 API 呼叫需要付費。以 Claude Sonnet 為例，一般日常使用的月費大約在
    $5–$30 美元之間，取決於使用頻率。
  - '记忆文件保存在磁盘上，持久存在直到你删除它们。限制是你的存储空间，而不是模型。会话上下文仍然受模型上下文窗口限制，所以长对话可能会压缩或截断。这就是记忆搜索存在的原因——它只将相关部分拉回上下文。


    文档：记忆、上下文。


    ​ 磁盘上的文件位置'
- source_sentence: '为什么我看到 LLM request rejected: messages.N.content.X.tool_use.input:
    Field required'
  sentences:
  - '这是一个提供商验证错误：模型发出了一个没有必需input的tool_use块。通常意味着会话历史已过时或损坏（通常在长线程或工具/模式变更后发生）。


    修复：使用/new（独立消息）开始新会话。'
  - '目前支持的模式有：


    • **定时任务** ：隔离的任务可以为每个任务设置 `model` 覆盖。

    • **子智能体** ：将任务路由到具有不同默认模型的独立智能体。

    • **按需切换** ：使用 `/model` 随时切换当前会话模型。


    参阅定时任务、多智能体路由和斜杠命令。'
  - '不是必需的，但推荐用于可靠性和隔离。


    • **专用主机（VPS/Mac mini/Pi）：** 常开，更少的休眠/重启中断，更干净的权限，更容易保持运行。

    • **共享的笔记本/台式机：** 完全适合测试和活跃使用，但当机器休眠或更新时预期会有暂停。


    如果你想要两全其美，将 Gateway 网关保持在专用主机上，并将笔记本配对为节点以获取本地屏幕/摄像头/执行工具。参阅节点。

    安全指南请阅读安全。'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer

This is a [sentence-transformers](https://www.SBERT.net) model trained. It maps sentences & paragraphs to a 1024-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Maximum Sequence Length:** 8192 tokens
- **Output Dimensionality:** 1024 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 8192, 'do_lower_case': False, 'architecture': 'XLMRobertaModel'})
  (1): Pooling({'word_embedding_dimension': 1024, 'pooling_mode_cls_token': True, 'pooling_mode_mean_tokens': False, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
  (2): Normalize()
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    '为什么我看到 LLM request rejected: messages.N.content.X.tool_use.input: Field required',
    '这是一个提供商验证错误：模型发出了一个没有必需input的tool_use块。通常意味着会话历史已过时或损坏（通常在长线程或工具/模式变更后发生）。\n\n修复：使用/new（独立消息）开始新会话。',
    '不是必需的，但推荐用于可靠性和隔离。\n\n• **专用主机（VPS/Mac mini/Pi）：** 常开，更少的休眠/重启中断，更干净的权限，更容易保持运行。\n• **共享的笔记本/台式机：** 完全适合测试和活跃使用，但当机器休眠或更新时预期会有暂停。\n\n如果你想要两全其美，将 Gateway 网关保持在专用主机上，并将笔记本配对为节点以获取本地屏幕/摄像头/执行工具。参阅节点。\n安全指南请阅读安全。',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 1024]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.5746, 0.2210],
#         [0.5746, 1.0000, 0.2096],
#         [0.2210, 0.2096, 1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 208 training samples
* Columns: <code>anchor</code> and <code>positive</code>
* Approximate statistics based on the first 208 samples:
  |         | anchor                                                                            | positive                                                                            |
  |:--------|:----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|
  | type    | string                                                                            | string                                                                              |
  | details | <ul><li>min: 2 tokens</li><li>mean: 14.44 tokens</li><li>max: 49 tokens</li></ul> | <ul><li>min: 2 tokens</li><li>mean: 180.24 tokens</li><li>max: 712 tokens</li></ul> |
* Samples:
  | anchor                      | positive                                                                                                                                                                                                                                           |
  |:----------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>OpenClaw 要錢嗎？</code>  | <code>OpenClaw 軟體本身完全免費且開源。但它需要連接 AI 模型來「思考」，而這些模型的 API 呼叫需要付費。以 Claude Sonnet 為例，一般日常使用的月費大約在 $5–$30 美元之間，取決於使用頻率。</code>                                                                                                                         |
  | <code>我的電腦跑得動嗎？</code>      | <code>OpenClaw 對硬體的要求很低——因為「思考」的工作由雲端 AI 模型完成，你的電腦只負責執行指令。基本需求：[6]<br><br>作業系統：macOS、Linux（含 Windows WSL2）<br>Node.js：22 或以上<br>記憶體：2GB 以上（推薦 4GB）<br>磁碟空間：約 500MB<br>網路：需要穩定的網際網路連線（用於呼叫 AI 模型 API）<br>甚至一台樹莓派（Raspberry Pi）也能運行 OpenClaw。</code> |
  | <code>安全嗎？我的資料會被看到嗎？</code> | <code>安全性取決於你的配置。[5] 需要注意兩件事：<br><br>代理權限：OpenClaw 以你的用戶身份運行，能存取你能存取的所有檔案。使用 Docker 沙盒可以限制其存取範圍<br>資料傳輸：代理會將你的指令和相關檔案內容發送到 AI 模型供應商（如 Anthropic）進行處理。敏感資料不應讓代理接觸<br>詳細的安全指南請參閱我們的《安全性完全指南》。</code>                                                 |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 4
- `num_train_epochs`: 10
- `learning_rate`: 2e-05
- `warmup_steps`: 20
- `gradient_accumulation_steps`: 2
- `bf16`: True
- `remove_unused_columns`: False

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 4
- `num_train_epochs`: 10
- `max_steps`: -1
- `learning_rate`: 2e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 20
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 2
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1.0
- `label_smoothing_factor`: 0.0
- `bf16`: True
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: trackio
- `eval_strategy`: no
- `per_device_eval_batch_size`: 8
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: False
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch  | Step | Training Loss |
|:------:|:----:|:-------------:|
| 0.0385 | 1    | 0.1817        |
| 0.0769 | 2    | 0.1655        |
| 0.1154 | 3    | 0.1211        |
| 0.1538 | 4    | 0.2993        |
| 0.1923 | 5    | 0.2423        |
| 0.2308 | 6    | 0.3498        |
| 0.2692 | 7    | 0.2387        |
| 0.3077 | 8    | 0.4324        |
| 0.3462 | 9    | 0.2882        |
| 0.3846 | 10   | 0.4726        |
| 0.4231 | 11   | 0.1541        |
| 0.4615 | 12   | 0.1159        |
| 0.5    | 13   | 0.1987        |
| 0.5385 | 14   | 0.1731        |
| 0.5769 | 15   | 0.4158        |
| 0.6154 | 16   | 0.0620        |
| 0.6538 | 17   | 0.4192        |
| 0.6923 | 18   | 0.1253        |
| 0.7308 | 19   | 0.1840        |
| 0.7692 | 20   | 0.1547        |
| 0.8077 | 21   | 0.3482        |
| 0.8462 | 22   | 0.3398        |
| 0.8846 | 23   | 0.0817        |
| 0.9231 | 24   | 0.1160        |
| 0.9615 | 25   | 0.0714        |
| 1.0    | 26   | 0.1563        |
| 1.0385 | 27   | 0.0873        |
| 1.0769 | 28   | 0.0800        |
| 1.1154 | 29   | 0.1991        |
| 1.1538 | 30   | 0.0777        |
| 1.1923 | 31   | 0.0710        |
| 1.2308 | 32   | 0.0978        |
| 1.2692 | 33   | 0.1235        |
| 1.3077 | 34   | 0.2529        |
| 1.3462 | 35   | 0.1559        |
| 1.3846 | 36   | 0.1644        |
| 1.4231 | 37   | 0.3412        |
| 1.4615 | 38   | 0.3170        |
| 1.5    | 39   | 0.0370        |
| 1.5385 | 40   | 0.0453        |
| 1.5769 | 41   | 0.0715        |
| 1.6154 | 42   | 0.1498        |
| 1.6538 | 43   | 0.0699        |
| 1.6923 | 44   | 0.0739        |
| 1.7308 | 45   | 0.0845        |
| 1.7692 | 46   | 0.1876        |
| 1.8077 | 47   | 0.1116        |
| 1.8462 | 48   | 0.0688        |
| 1.8846 | 49   | 0.0714        |
| 1.9231 | 50   | 0.1871        |
| 1.9615 | 51   | 0.0217        |
| 2.0    | 52   | 0.0337        |
| 2.0385 | 53   | 0.0136        |
| 2.0769 | 54   | 0.0603        |
| 2.1154 | 55   | 0.0306        |
| 2.1538 | 56   | 0.0194        |
| 2.1923 | 57   | 0.0770        |
| 2.2308 | 58   | 0.0401        |
| 2.2692 | 59   | 0.0854        |
| 2.3077 | 60   | 0.1019        |
| 2.3462 | 61   | 0.1897        |
| 2.3846 | 62   | 0.0589        |
| 2.4231 | 63   | 0.0963        |
| 2.4615 | 64   | 0.6751        |
| 2.5    | 65   | 0.0330        |
| 2.5385 | 66   | 0.0521        |
| 2.5769 | 67   | 0.0539        |
| 2.6154 | 68   | 0.0281        |
| 2.6538 | 69   | 0.1268        |
| 2.6923 | 70   | 0.1374        |
| 2.7308 | 71   | 0.2884        |
| 2.7692 | 72   | 0.1940        |
| 2.8077 | 73   | 0.1024        |
| 2.8462 | 74   | 0.0307        |
| 2.8846 | 75   | 0.3319        |
| 2.9231 | 76   | 0.0209        |
| 2.9615 | 77   | 0.0308        |
| 3.0    | 78   | 0.0225        |
| 3.0385 | 79   | 0.1901        |
| 3.0769 | 80   | 0.0594        |
| 3.1154 | 81   | 0.0382        |
| 3.1538 | 82   | 0.0473        |
| 3.1923 | 83   | 0.0628        |
| 3.2308 | 84   | 0.1749        |
| 3.2692 | 85   | 0.0371        |
| 3.3077 | 86   | 0.0198        |
| 3.3462 | 87   | 0.2716        |
| 3.3846 | 88   | 0.0629        |
| 3.4231 | 89   | 0.0243        |
| 3.4615 | 90   | 0.0637        |
| 3.5    | 91   | 0.0463        |
| 3.5385 | 92   | 0.0271        |
| 3.5769 | 93   | 0.0326        |
| 3.6154 | 94   | 0.0252        |
| 3.6538 | 95   | 0.0241        |
| 3.6923 | 96   | 0.1839        |
| 3.7308 | 97   | 0.0839        |
| 3.7692 | 98   | 0.2094        |
| 3.8077 | 99   | 0.0130        |
| 3.8462 | 100  | 0.0687        |
| 3.8846 | 101  | 0.0312        |
| 3.9231 | 102  | 0.0704        |
| 3.9615 | 103  | 0.0101        |
| 4.0    | 104  | 0.0329        |
| 4.0385 | 105  | 0.0367        |
| 4.0769 | 106  | 0.0139        |
| 4.1154 | 107  | 0.0511        |
| 4.1538 | 108  | 0.0356        |
| 4.1923 | 109  | 0.0086        |
| 4.2308 | 110  | 0.0544        |
| 4.2692 | 111  | 0.0370        |
| 4.3077 | 112  | 0.0089        |
| 4.3462 | 113  | 0.1223        |
| 4.3846 | 114  | 0.0384        |
| 4.4231 | 115  | 0.3160        |
| 4.4615 | 116  | 0.0481        |
| 4.5    | 117  | 0.0840        |
| 4.5385 | 118  | 0.0468        |
| 4.5769 | 119  | 0.0204        |
| 4.6154 | 120  | 0.0144        |
| 4.6538 | 121  | 0.0346        |
| 4.6923 | 122  | 0.0208        |
| 4.7308 | 123  | 0.0505        |
| 4.7692 | 124  | 0.1888        |
| 4.8077 | 125  | 0.1264        |
| 4.8462 | 126  | 0.0270        |
| 4.8846 | 127  | 0.0780        |
| 4.9231 | 128  | 0.1350        |
| 4.9615 | 129  | 0.0516        |
| 5.0    | 130  | 0.0111        |
| 5.0385 | 131  | 0.0405        |
| 5.0769 | 132  | 0.0235        |
| 5.1154 | 133  | 0.0945        |
| 5.1538 | 134  | 0.0066        |
| 5.1923 | 135  | 0.0427        |
| 5.2308 | 136  | 0.0194        |
| 5.2692 | 137  | 0.0411        |
| 5.3077 | 138  | 0.0126        |
| 5.3462 | 139  | 0.0583        |
| 5.3846 | 140  | 0.0356        |
| 5.4231 | 141  | 0.0339        |
| 5.4615 | 142  | 0.0223        |
| 5.5    | 143  | 0.0300        |
| 5.5385 | 144  | 0.1319        |
| 5.5769 | 145  | 0.0044        |
| 5.6154 | 146  | 0.0041        |
| 5.6538 | 147  | 0.0587        |
| 5.6923 | 148  | 0.0563        |
| 5.7308 | 149  | 0.0383        |
| 5.7692 | 150  | 0.1654        |
| 5.8077 | 151  | 0.0465        |
| 5.8462 | 152  | 0.0405        |
| 5.8846 | 153  | 0.0602        |
| 5.9231 | 154  | 0.0176        |
| 5.9615 | 155  | 0.0960        |
| 6.0    | 156  | 0.0075        |
| 6.0385 | 157  | 0.0166        |
| 6.0769 | 158  | 0.0398        |
| 6.1154 | 159  | 0.0298        |
| 6.1538 | 160  | 0.0340        |
| 6.1923 | 161  | 0.0103        |
| 6.2308 | 162  | 0.0638        |
| 6.2692 | 163  | 0.0029        |
| 6.3077 | 164  | 0.0649        |
| 6.3462 | 165  | 0.0043        |
| 6.3846 | 166  | 0.0546        |
| 6.4231 | 167  | 0.2316        |
| 6.4615 | 168  | 0.0215        |
| 6.5    | 169  | 0.0110        |
| 6.5385 | 170  | 0.0376        |
| 6.5769 | 171  | 0.0076        |
| 6.6154 | 172  | 0.0096        |
| 6.6538 | 173  | 0.0069        |
| 6.6923 | 174  | 0.1448        |
| 6.7308 | 175  | 0.0125        |
| 6.7692 | 176  | 0.0267        |
| 6.8077 | 177  | 0.1370        |
| 6.8462 | 178  | 0.0187        |
| 6.8846 | 179  | 0.0646        |
| 6.9231 | 180  | 0.0162        |
| 6.9615 | 181  | 0.0210        |
| 7.0    | 182  | 0.0220        |
| 7.0385 | 183  | 0.0475        |
| 7.0769 | 184  | 0.0359        |
| 7.1154 | 185  | 0.0350        |
| 7.1538 | 186  | 0.0275        |
| 7.1923 | 187  | 0.0227        |
| 7.2308 | 188  | 0.0527        |
| 7.2692 | 189  | 0.1779        |
| 7.3077 | 190  | 0.0222        |
| 7.3462 | 191  | 0.0530        |
| 7.3846 | 192  | 0.0264        |
| 7.4231 | 193  | 0.0045        |
| 7.4615 | 194  | 0.0106        |
| 7.5    | 195  | 0.0614        |
| 7.5385 | 196  | 0.0699        |
| 7.5769 | 197  | 0.0097        |
| 7.6154 | 198  | 0.0155        |
| 7.6538 | 199  | 0.0847        |
| 7.6923 | 200  | 0.0131        |
| 7.7308 | 201  | 0.1222        |
| 7.7692 | 202  | 0.0095        |
| 7.8077 | 203  | 0.0161        |
| 7.8462 | 204  | 0.0158        |
| 7.8846 | 205  | 0.0281        |
| 7.9231 | 206  | 0.0489        |
| 7.9615 | 207  | 0.0371        |
| 8.0    | 208  | 0.0383        |
| 8.0385 | 209  | 0.0132        |
| 8.0769 | 210  | 0.0050        |
| 8.1154 | 211  | 0.0602        |
| 8.1538 | 212  | 0.0825        |
| 8.1923 | 213  | 0.2327        |
| 8.2308 | 214  | 0.0789        |
| 8.2692 | 215  | 0.0387        |
| 8.3077 | 216  | 0.0196        |
| 8.3462 | 217  | 0.0214        |
| 8.3846 | 218  | 0.0062        |
| 8.4231 | 219  | 0.0338        |
| 8.4615 | 220  | 0.0200        |
| 8.5    | 221  | 0.0116        |
| 8.5385 | 222  | 0.0617        |
| 8.5769 | 223  | 0.2851        |
| 8.6154 | 224  | 0.1016        |
| 8.6538 | 225  | 0.0156        |
| 8.6923 | 226  | 0.0055        |
| 8.7308 | 227  | 0.0248        |
| 8.7692 | 228  | 0.0373        |
| 8.8077 | 229  | 0.0312        |
| 8.8462 | 230  | 0.0863        |
| 8.8846 | 231  | 0.0427        |
| 8.9231 | 232  | 0.0168        |
| 8.9615 | 233  | 0.0382        |
| 9.0    | 234  | 0.0340        |
| 9.0385 | 235  | 0.0221        |
| 9.0769 | 236  | 0.0139        |
| 9.1154 | 237  | 0.0158        |
| 9.1538 | 238  | 0.1216        |
| 9.1923 | 239  | 0.0089        |
| 9.2308 | 240  | 0.0492        |
| 9.2692 | 241  | 0.0127        |
| 9.3077 | 242  | 0.0075        |
| 9.3462 | 243  | 0.0046        |
| 9.3846 | 244  | 0.0693        |
| 9.4231 | 245  | 0.0103        |
| 9.4615 | 246  | 0.0498        |
| 9.5    | 247  | 0.0048        |
| 9.5385 | 248  | 0.0251        |
| 9.5769 | 249  | 0.0170        |
| 9.6154 | 250  | 0.0725        |
| 9.6538 | 251  | 0.0504        |
| 9.6923 | 252  | 0.1176        |
| 9.7308 | 253  | 0.0726        |
| 9.7692 | 254  | 0.0072        |
| 9.8077 | 255  | 0.0908        |
| 9.8462 | 256  | 0.0237        |
| 9.8846 | 257  | 0.0675        |
| 9.9231 | 258  | 0.0716        |
| 9.9615 | 259  | 0.0037        |
| 10.0   | 260  | 0.0200        |

</details>

### Framework Versions
- Python: 3.11.15
- Sentence Transformers: 5.3.0
- Transformers: 5.4.0
- PyTorch: 2.11.0+cu130
- Accelerate: 1.13.0
- Datasets: 4.8.4
- Tokenizers: 0.22.2

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->