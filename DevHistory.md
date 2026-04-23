# Mistral-7B SAE 训练 — 开发日志

## 2026-04-10

### 起因

ZL用 LEACE 做 Mistral-7B 的 SC/SemC 双分离实验全翻车——20+ 配置都是尖锐相变，结构和语义一起崩。诊断结论：Mistral 的结构子空间和语义子空间纠缠（EUO=1.59），线性刀切不开。方案：训 SAE 升维解耦，精确干预 JSON 结构特征。

但其实就是想玩玩 SAE 😄

### 环境

- **硬件**：DGX Spark GB10, 128GB 统一内存
- **容器**：`d2l_exp`（`nvcr.io/nvidia/pytorch:25.11-py3`）
  - torch 2.10.0 + CUDA 13.0
  - transformers 5.3.0
  - SAELens 6.39.0 + transformer-lens 2.16.1
- **注意**：容器没网（NAT 不通），大文件在宿主机下，通过 `/workspace` 映射进去

### 踩坑记录

#### 1. GPU 检测失败
容器启动后 `torch.cuda.is_available() = False`。`docker restart d2l_exp` 后恢复。Spark 的 GB10 偶尔会掉 GPU 连接，重启容器就好。

#### 2. TransformerLens 不认 Mistral v0.3
`mistralai/Mistral-7B-Instruct-v0.3` 不在 TransformerLens 的 `OFFICIAL_MODEL_NAMES` 里。换成 v0.1（架构一样，都是 32 layers / 4096 hidden dim）。

#### 3. 容器离线加载模型
SAELens → TransformerLens → `AutoConfig.from_pretrained("mistralai/...")` 会试图联网。解法：
- 设 `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`
- 在 HF cache 目录创建软链指向本地模型文件（模拟 cache 结构）
- 用 `AutoModelForCausalLM.from_pretrained(本地路径)` 预加载模型，通过 `hf_model` 参数传给 TransformerLens

#### 4. transformers 5.x 删了 rope_theta
TransformerLens 2.16.1 读 `hf_config.rope_theta`，但 transformers 5.3.0 的 `MistralConfig` 已经移除了这个属性（换成了 `rope_parameters` 体系）。解法：monkey-patch `AutoConfig.from_pretrained`，从原始 config.json 里补回 `rope_theta`。

### 当前状态（2026-04-10）

Layer 16 SAE 训练中，但 Spark 反复卡死（见 4-11 记录）。

配置：
- **架构**：JumpReLU SAE（Anthropic Scaling Monosemanticity 同款）
- **字典大小**：65536（16x expansion）
- **训练 tokens**：409,600,000（~400M）
- **学习率**：5e-5，常数 + warmup 1000 步 + 末尾 20% 衰减
- **模型精度**：float16（autocast_lm），SAE 精度：float32

### 文件

| 文件 | 用途 |
|---|---|
| `train_sae.py` | SAE 训练脚本，支持单层/批量训练 |
| `sae_checkpoints/` | 输出目录 |

---

## 2026-04-11

### Spark 反复卡死

4-10 到 4-11 上午，layer 16 训练多次跑到一半 Spark 整机僵死。没有明确报错，大概率是 OOM——Grace CPU 的共享内存没有 CUDA 那种显式 OOM 报错，耗尽后直接僵死。

### 内存分析

原配置的内存开销估算：
| 项目 | 估算 |
|---|---|
| Mistral-7B 模型 (fp16) | ~14 GB |
| SAE 权重 65536×4096 (fp32) | ~2.1 GB |
| Adam 优化器状态 (×3) | ~6.3 GB |
| 激活缓存 n_batches_in_buffer=64 | ~4.3 GB |
| TransformerLens 包装开销 | 不确定，至少几 GB |
| 数据集加载 + 前向传播中间激活 | 不确定 |
| **合计** | **>30 GB + 不确定项** |

128GB 看起来够，但 Grace CPU 统一内存的碎片化行为和 GPU 显存不同，峰值远超稳态。

### 改参数降内存

三处改动，总共省约 8-9 GB 稳态 + 降低峰值：

| 参数 | 改前 | 改后 | 省多少 |
|---|---|---|---|
| `D_SAE`（字典大小） | 65536 (16x) | **16384 (4x)** | ~4.7 GB (SAE权重+optimizer) |
| `n_batches_in_buffer` | 64 | **8** | ~3.8 GB (激活缓存) |
| `store_batch_size_prompts` | 16 | **4** | 降低前向传播峰值 |

**16K 字典 vs 64K**：Anthropic 原始论文用过 4x expansion，特征少但更容易人工检查。64K 有 65536 个特征，找 JSON 结构特征要翻到天荒地老。16K 对写公众号够了。

### buffer 报错修复

`n_batches_in_buffer=8` 太小，SAELens 的 mixing_buffer 要求 buffer size ≥ batch size，报错：
```
ValueError: Buffer size must be greater than or equal to batch size
```
改 `n_batches_in_buffer` 8→16，修复。

### ✅ Layer 16 训练成功（2026-04-11 夜间 → 4-12 凌晨）

改参数后重跑，12000 步全部跑完。中间 Windows 电脑重启导致 ssh 断开，但训练在容器里已经跑完了。

**输出文件**（注意：实际存在 `output/` 而非 `sae_checkpoints/`，SAELens 默认行为）：
```
output/
├── cfg.json                  # 训练配置
└── sae_weights.safetensors   # SAE 权重
```

**最终训练配置**：
| 参数 | 值 |
|---|---|
| 模型 | Mistral-7B-Instruct-v0.1 |
| 目标层 | Layer 16 (hook_resid_post) |
| SAE 架构 | JumpReLU |
| 字典大小 | 16384 (4x expansion) |
| 训练 tokens | ~49M |
| 训练步数 | 12000 |
| n_batches_in_buffer | 16 |
| store_batch_size_prompts | 4 |
| 训练时长 | 约 10 小时 |

**训练中间指标**（800 步时）：
- mse_loss: 2060（重建误差，正常范围）
- l0_loss: 293（平均每 token 激活 293/16384 = 1.8% 特征，稀疏度合理）

### 教训

**长任务必须用 tmux**：ssh 断开 = 进程被杀（SIGHUP）。下次：
```
tmux new -s sae
python train_sae.py
# Ctrl+B, D 退出，关电脑不影响
# 回来: tmux attach -t sae
```
这次是运气好跑完了才断的。

### 文件（更新）

| 文件 | 用途 |
|---|---|
| `train_sae.py` | SAE 训练脚本 |
| `output/cfg.json` | Layer 16 训练配置 |
| `output/sae_weights.safetensors` | **Layer 16 SAE 权重 ✅** |
| `sae_checkpoints/` | 空（SAELens 没用这个目录） |

### 下一步

- 干预实验：关掉某个 JSON 结构特征，看模型生成怎么变
- 写公众号（SAE 科普 + Mistral 内部结构可视化）

---

## 2026-04-12

### SAE 验证

写了两版验证脚本，确认 SAE 训练质量：

**v1 (`validate_sae.py`)**：三板斧全过，但 top-20 特征被 BOS token (`<s>`) 霸占（激活值 60~115，远超正常 token）。

**v2 (`validate_sae_v2.py`)**：排除 BOS token 后重跑。

**定量指标**：
| 指标 | v1（含 BOS） | v2（排除 BOS） | 说明 |
|------|-------------|---------------|------|
| Explained Variance | **0.9622** ✅ | 0.4747 | v2 下降是因为去掉 BOS 后方差从 0.30 暴跌到 0.02，MSE 几乎不变（0.0114→0.0116），EV 在低方差数据上不稳定，不代表质量差 |
| MSE | 0.0114 | 0.0116 | 几乎一样，重建质量没问题 |
| L0 | 79.4 ✅ | 56.9 ✅ | 每 token 平均激活 57~79 / 16384 个特征，稀疏度好 |

**特征可解释性（v2，排除 BOS 后的 top-20）**：

JSON 侧找到了清晰的结构特征：
- Feature 1000：`[` 方括号
- Feature 11062：`{"` JSON 对象开头
- Feature 4765：`}`, `}}}`, `}}` 闭合括号
- Feature 2918：`"`, `":` 引号/键值
- Feature 200：`id`, `title`, `tags` — JSON 键名
- Feature 9459：`_` 下划线（`user_id`, `app_user`）

自然语言侧完全不同的特征分布：
- Feature 5088：句首词 `In`, `The`
- Feature 16167：`early`, `morning` 时间意象
- Feature 8350：`morning`, `night`, `light` 时间/光线语义簇

**两侧 top-20 几乎零重叠** — SAE 确实把 JSON 结构和自然语言语义分到了不同特征上。

JSON 标点偏好特征 top-10 全部 ratio > 1e8（只在 JSON 标点上亮，非标点激活为零）：
- Feature 14474 / 15086：`"` 引号
- Feature 442：`"`, `{"` 对象开头
- Feature 1978：`,`, `",` 分隔符
- Feature 2122：`":` 键值分隔符
- Feature 9353：`{"` 对象开头
- Feature 9582：`"]`, `}]` 闭合

### cfg.json 损坏修复

SAELens 保存 cfg.json 时试图序列化 `hf_model`（整个 PyTorch 模型对象），导致文件在 546 字节处截断。手动重建了合法的 cfg.json。

### 上传 HuggingFace

权重上传到 `lmxxf/mistral-7b-sae-layer16`。

### SAE 特征干预实验 ✅

用 `intervene_sae.py` 在 Layer 16 hook 点做特征消融：SAE encode → clamp 目标特征为 0 → SAE decode → 替换原始激活。三级干预逐步加码。

**干预特征配置**：
- Light (1 feat)：Feature 2122（`":` 键值分隔符）
- Medium (3 feat)：+9353（`{"`）+1000（`[`）
- Heavy (10 feat)：全部 top-10 JSON 标点偏好特征

**测试 prompt**：5 个语义类（car/person/book/restaurant/school）+ 2 个无意义类（blonf/trelm），贪婪解码 max 200 token。

**结果：发现两类行为模式**

**模式 A：完全崩溃（car、school、blonf）**
- light 干预就直接输出空 code block（` ```\n\n``` `），SC 崩、SemC 也崩
- 和 LEACE 的"尖锐相变"行为一致

**模式 B：渐进退化（person、book、restaurant）** ← 关键发现
- light/medium 干预后 SC 依然通过（json.loads 成功），格式略变（缩进减少）
- heavy 干预后 SC 崩溃，但 **SemC 完全保留**
- 例：book + heavy → `"title": "The Great Gatsby", "author: "F. Scott Fitzgerald "pages: 176` → 引号/冒号打乱（SC❌），但书名作者页数全在（SemC 保留）
- 例：person + heavy → `"name": "John", "age: 25 "city": "New York"` → 结构破损，语义完整

**这是 LEACE 在 Mistral-7B 上 20+ 配置都做不到的事：SC 崩溃而 SemC 保留。**

| | LEACE（ZL 20+ 配置） | SAE 干预（本实验） |
|---|---|---|
| Mistral-7B 双分离 | ❌ 全部尖锐相变 | ✅ 部分 prompt 实现 |
| 行为模式 | 只有"不穿"和"全穿" | 存在渐进退化区间 |
| 语义保留 | 结构崩时语义同步崩 | 结构崩时语义可保留 |

**SemC 评分说明**：绝对值偏低（0.15~0.55）是因为 content_words 列表过宽（car 词表含 toyota/honda/ford/bmw/tesla 等十几个，模型只输出一个）。关键指标是干预前后 SemC 不变。

**模式 A 的原因推测**：这些 prompt 的 JSON 生成可能强依赖 Layer 16 的结构特征，一旦被干预就无法启动生成（直接输出空 code block）。模式 B 的 prompt 可能有其他层的冗余路径支撑部分生成能力。

### 文件（更新）

| 文件 | 用途 |
|---|---|
| `train_sae.py` | SAE 训练脚本 |
| `validate_sae.py` | SAE 验证 v1（含 BOS） |
| `validate_sae_v2.py` | SAE 验证 v2（排除 BOS，支持 --layer 和 --sae-path 参数） |
| `intervene_sae.py` | **SAE 特征干预实验 ✅** |
| `output/cfg.json` | Layer 16 训练配置（已手动修复） |
| `output/sae_weights.safetensors` | Layer 16 SAE 权重 |
| `output/intervention_results.json` | **干预实验原始结果** |
| `temp-cli.md` | 临时命令备忘（可删） |

### 下一步

- 三层（L8+L16+L22）联合干预实验
- 公众号后续期数：SAE 干预实验可视化

---

## 2026-04-13

### L8 + L22 SAE 训练

训练 Layer 8 和 Layer 22 的 SAE，与之前的 Layer 16 组成三层覆盖（浅层/中层/深层）。

**踩坑记录**：

1. **L8 第一次训练（4-12）**：训完 12000 步后崩在 SAELens 保存 cfg.json（同 L16 的 bug），但 fallback 代码里 `import json` 忘了加 → 权重没存下来，白跑 5 小时
2. **L8 第二次训练（4-12~13）**：修了 `import json`，但 fallback 保存时用 `sae.state_dict()` 存的是训练态 key（`log_threshold`），加载时 SAELens 期望推理态 key（`threshold`）→ 权重存下来了但加载报错
3. **L8 第三次训练（4-13）**：修了 `log_threshold → threshold` 转换，终于成功。同时用 `safetensors` 转换了已有的 L8/L22 权重文件

**train_sae.py 修复内容**：
- 加了 `import json`（顶部）
- fallback 保存时 `log_threshold` → `torch.exp()` → `threshold`（训练态转推理态）

**validate_sae_v2.py 改进**：
- 加了 `--layer` 和 `--sae-path` 命令行参数，不再硬编码 Layer 16

**训练结果**：

| 层 | MSE (训练末) | L0 (训练末) | 训练时长 | 保存位置 |
|---|---|---|---|---|
| L8 | 911 | 267 | 5h | `sae_checkpoints/mistral7b_sae_L8_64k/` |
| L16 | 903 | 266 | 10h | `output/` |
| L22 | 1326 | 327 | 11.5h | `sae_checkpoints/mistral7b_sae_L22_64k/` |

L22 比 L8/L16 慢一倍（1197 vs 2710 it/s）且 MSE/L0 更高——深层残差流信息更密，重建更难。

### 三层验证结果

| 层 | MSE (推理) | L0 (推理) | JSON标点特征 | 特征可解释性 |
|---|---|---|---|---|
| L8 | 0.0019 | 33.4 | ratio>1e8 ✅ | ✅ |
| L16 | 0.0116 | 56.9 | ratio>1e8 ✅ | ✅ |
| L22 | 0.0525 | 57.7 | ratio>1e8 ✅ | ✅ |

MSE 从浅到深递增（0.002 → 0.012 → 0.052），越深层信息越密重建越难。三层都找到了 JSON 结构特征。

**关键发现：浅层 vs 深层的 JSON 特征类型不同**

| | L8 JSON top-20 | L22 JSON top-20 |
|---|---|---|
| 特征类型 | 语法标点级（`{`, `"`, `:`, `[`） | 语义概念级（system, credentials, Post, Alice） |
| 说明 | 浅层编码的是"这里要放个引号" | 深层编码的是"这个值的含义是什么" |

这印证了 Mistral "渐进展开型"——浅层压缩（eff_rank 低），深层展开（语义级特征涌现）。

### 文件（更新）

| 文件 | 用途 |
|---|---|
| `train_sae.py` | SAE 训练脚本（已修 import json + log_threshold 转换） |
| `validate_sae.py` | SAE 验证 v1（含 BOS） |
| `validate_sae_v2.py` | SAE 验证 v2（排除 BOS，支持 --layer/--sae-path） |
| `intervene_sae.py` | SAE 特征干预实验 |
| `output/` | Layer 16 SAE 权重 + cfg + 干预结果 |
| `sae_checkpoints/mistral7b_sae_L8_64k/` | **Layer 8 SAE 权重 ✅** |
| `sae_checkpoints/mistral7b_sae_L22_64k/` | **Layer 22 SAE 权重 ✅** |

### 三层联合干预实验 ✅

`intervene_multilayer.py`：5 个干预级别 × 7 个 prompt = 35 次生成。

**干预级别**：
- baseline：无干预
- L16_only：L16 top-10 JSON 标点特征
- shallow_only：L8 top-10
- deep_only：L22 top-10
- all_layers：L8+L16+L22 top-10（共 30 个特征）

**汇总结果（语义类 prompt）**：

| 级别 | SC | Avg SemC | Collapse | 特点 |
|---|---|---|---|---|
| baseline | 100% | 0.28 | 0% | 正常 |
| L16_only | 0% | 0.00 | 80% | 大面积崩溃，输出 `` ```---``` `` |
| shallow_only (L8) | 0% | 0.07 | 60% | 死循环结构符号 `{{{[[[((( ` |
| **deep_only (L22)** | **0%** | **0.12** | **0%** | **SC 全崩但零崩溃！语义保留** |
| all_layers | 0% | 0.07 | 20% | 加了 L8+L16 反而崩溃率上升 |

**核心发现：L22 深层干预是唯一实现"零崩溃"的方案**

L22 干预的具体输出：
- restaurant: `"name": " The Fat Cat" "cuisine": " New American" "rating": "3.5" "address": " 123 Main St. "` → 格式坏了但语义全在
- school: `"name": " Lincoln High School", "address": " 12th Avenue", "founded": "1968,,", "teachers": ""` → 结构破损但学校名地址年份都在
- book: `"title" : " The Road to Rio ", "author" : " Clive Barkham ", "pages" : " 100 pages "` → 键值对格式变了但语义丰富

**为什么不同层的干预效果差异这么大**：
- L8（浅层）编码的是语法标点级特征——关掉后模型连基本的 token 序列都组织不了，陷入结构符号死循环
- L16（中层）类似——干预太底层，模型失去基本输出能力
- L22（深层）编码的是输出规划级特征——关掉后模型不知道"该用 JSON 格式输出"，但还能生成有意义的内容

**结论：要做精确的结构/语义分离，得在深层动刀。浅层/中层的标点特征太基础，关掉等于把嘴缝上了。深层的规划特征是"格式决策"层面的，关掉只影响输出格式不影响内容。**

### 文件（更新）

| 文件 | 用途 |
|---|---|
| `train_sae.py` | SAE 训练脚本（已修 import json + log_threshold 转换） |
| `validate_sae.py` | SAE 验证 v1（含 BOS） |
| `validate_sae_v2.py` | SAE 验证 v2（排除 BOS，支持 --layer/--sae-path） |
| `intervene_sae.py` | SAE 特征干预实验（单层 L16） |
| `intervene_multilayer.py` | **三层联合干预实验 ✅** |
| `output/` | Layer 16 SAE 权重 + cfg + 干预结果 |
| `output/intervention_multilayer_results.json` | **三层干预原始结果** |
| `sae_checkpoints/mistral7b_sae_L8_64k/` | Layer 8 SAE 权重 |
| `sae_checkpoints/mistral7b_sae_L22_64k/` | Layer 22 SAE 权重 |

### 下一步

- L22 单层精细干预：不用 top-10 全关，试 top-3 / top-5，找最小必要干预集
- 公众号素材：三层对比的可视化
- 更新给ZL的结论：深层干预 > 浅层干预，L22 是做双分离的最佳层

---

## 2026-04-20

### EMNLP 2026 冲刺：重训 SAE（两模型六层）

**背景**：ZL定了 EMNLP 2026 主会（5/25 截止）。之前的 SAE 有两个方法论漏洞，审稿人必抓：

1. **模型版本不一致**：train_sae.py 里 MODEL_PATH 写的是 v0.1，但ZL DAS 实验全在 v0.3 上跑。激活分布不一致，SAE 特征 clamping 对不上
2. **训练数据分布不匹配**：原来用 OpenWebText（纯文本），但 DAS 实验在 chat template 下做（`[INST]...[/INST]`）。SAE 没见过 chat format 的激活分布，特征 clamping 脱分布

**计划**：
- Mistral-7B-Instruct-v0.3：L8 / L16 / L22 三层，~30h
- Llama-3.1-8B-Instruct：L8 / L16 / L22 三层，~30h（跨模型复现）
- Qwen 砍了，放 future work（36 天窗口精不必全）
- ZL同步推 format specificity 对照实验（JSON vs XML/YAML），Codex 点名的 R2 防御

### 改动 1：模型版本 v0.1 → v0.3

train_sae.py：
- `MODEL_PATH` 改为 `/workspace/models/Mistral-7B-Instruct-v0.3`
- 新增 `TRANSFORMERLENS_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"`

**TransformerLens 不认 v0.3 的原因**：不是单纯的白名单限制。TransformerLens 内部对每个模型架构维护了一套权重 key 转换映射（HuggingFace 的 `model.layers.0.self_attn.q_proj.weight` → TransformerLens 的 `blocks.0.attn.W_Q`），白名单（`OFFICIAL_MODEL_NAMES`）本质是"这个模型的转换映射我写好了"。v0.3 不在列表里只是因为维护者没跟上 Mistral 发版节奏——v0.1 和 v0.3 架构完全一样（32 层 / 4096 hidden dim / 同样的 attention），只是词表从 32000 扩到 32768（多了 function calling tokens）。所以用 v0.1 的名字注册、实际塞 v0.3 的权重完全可行，转换映射通用。

**具体绕法**：
- HF cache 软链：用 v0.1 的目录名，软链指向 v0.3 的文件
- monkey-patch `AutoConfig.from_pretrained`：从 v0.3 的 config.json 补 `rope_theta` + 覆盖 `vocab_size=32768`
- `hf_model` 参数：直接传 v0.3 的 PyTorch 模型，TransformerLens 从权重读 shape
- cfg.json 里记录真实模型名 `mistralai/Mistral-7B-Instruct-v0.3`

### 改动 2：训练数据 OpenWebText → lmsys-chat-1m

**数据集**：`science-of-finetuning/lmsys-chat-1m-chat-formatted`（HuggingFace datasets repo，下载时要加 `--repo-type dataset`）

**预处理脚本 `prepare_lmsys_dataset.py`**：
- 从 lmsys 数据集的 `conversation` 字段取多轮对话 `[{role, content}, ...]`
- 用 Mistral v0.3 的 `tokenizer.apply_chat_template(conv, tokenize=False)` 渲染成 `<s>[INST] 问题[/INST] 回答</s>` 格式
- 过滤空/短文本后存成 parquet（`data/train-00000-of-00001.parquet`），SAELens 的 `load_dataset` 直接读 `text` 字段

**为什么需要这步**：lmsys 数据集只预渲染了 llama3 和 qwen2.5 的 chat template（`text_llama3`、`text_qwen2_5` 字段），没有 Mistral 的。必须自己用 Mistral tokenizer 重新渲染。

**预处理结果**：900,000 条对话 → 887,329 条（12,671 条过滤掉）

**不加 Pile / JSON-heavy 数据的原因**：否则 SAE 会学成 JSON 专家，下游 JSON 干预实验循环论证。

### 不改的部分

| 参数 | 值 | 原因 |
|---|---|---|
| SAE 架构 | JumpReLU | Anthropic 同款，审稿人认 |
| 字典大小 | 16384 (4x) | 够用且容易检查 |
| 三层 | L8 / L16 / L22 | 浅/中/深覆盖，已验证有效 |
| ~49M tokens/层 | 12000 步 × 4096 batch | 上次实测够收敛 |
| LR / warmup / decay | 5e-5 / 1000 / 20% | 不动 |

### ✅ L8 训练完成（2026-04-20 夜间）

SAELens 保存 cfg.json 时触发已知 bug（序列化 hf_model），fallback 代码兜住了。

**验证结果**：

| 指标 | 值 | 说明 |
|---|---|---|
| MSE | 0.0016 | 比旧版 v0.1 的 0.0019 更好 |
| L0 | 71.2 | 稀疏度正常 |
| EV | 0.33 | 排除 BOS 后低方差不稳定，不影响质量 |
| JSON 标点特征 | ratio > 1e7 ✅ | top-10 全部通过 |

JSON 侧 top-20 vs 自然语言侧 top-20 清晰分开，v0.3 + lmsys chat 数据训出的 SAE 质量确认没问题。

### ✅ L16 训练完成（2026-04-21 凌晨）

同样触发 SAELens cfg.json bug，fallback 兜住。

### ✅ L22 训练完成（2026-04-21 夜间 → 4-22）

同样触发 SAELens cfg.json bug，fallback 兜住。

### 三层验证结果（v0.3 + lmsys）

| 层 | MSE | L0 | JSON 标点 ratio | 说明 |
|---|---|---|---|---|
| L8 | 0.0016 | 71.2 | > 1e7 ✅ | 浅层，重建最好 |
| L16 | 0.0101 | 65.2 | > 1e8 ✅ | 中层 |
| L22 | 0.0409 | 76.6 | > 1e9 ✅ | 深层，MSE 最高但特征最清晰 |

与旧版 v0.1 趋势一致：MSE 从浅到深递增（深层信息越密重建越难），JSON 标点特征从浅到深越来越强。L22 的 `{"` 激活值 78.3、`":` 激活值 64.3，比 L8/L16 强一个数量级。

### Mistral v0.3 HuggingFace 上传（2026-04-22）

- [lmxxf/mistral-7b-v03-sae-layer8](https://huggingface.co/lmxxf/mistral-7b-v03-sae-layer8)
- [lmxxf/mistral-7b-v03-sae-layer16](https://huggingface.co/lmxxf/mistral-7b-v03-sae-layer16)
- [lmxxf/mistral-7b-v03-sae-layer22](https://huggingface.co/lmxxf/mistral-7b-v03-sae-layer22)

---

## 2026-04-22 ~ 04-24

### Llama-3.1-8B-Instruct SAE 训练（跨模型复现）

**目的**：EMNLP 审稿要求跨模型复现，证明 SAE 双分离不是 Mistral 的个例。

**模型下载**：Meta 官方需要申请权限，用 ModelScope 镜像（`LLM-Research/Meta-Llama-3.1-8B-Instruct`）免审批下载。

**TransformerLens 兼容性**：Llama-3.1-8B-Instruct 在官方列表里，不需要像 Mistral v0.3 那样骗。

**新增脚本**：
- `prepare_lmsys_llama.py`：lmsys 数据集用 Llama 3.1 chat template 渲染（`<|begin_of_text|><|start_header_id|>...`）
- `train_sae_llama.py`：Llama SAE 训练脚本，超参与 Mistral 一致
- `validate_sae_v2.py`：加了 `--model llama` 参数，支持双模型验证

**预处理结果**：900,000 条对话 → 900,000 条（零丢弃，Llama chat template 比 Mistral 长，无短文本）

**训练踩坑**：
- L8 训完后 GPU 掉线（Spark 老毛病），`docker restart` 恢复后继续 L16+L22
- L16+L22 一口气跑完没再掉

**训练时长**：L8 5h + L16 8.7h + L22 11.4h ≈ 25h

### Llama 三层验证结果

| 层 | MSE | L0 | JSON 标点 ratio | 说明 |
|---|---|---|---|---|
| L8 | 0.0046 | 28.7 | > 1e7 ✅ | L0 偏低但特征清晰 |
| L16 | 0.0124 | 35.5 | > 1e8 ✅ | 同上 |
| L22 | 0.0394 | 40.2 | > 1e8 ✅ | 深层 MSE 最高，符合预期 |

**跨模型 L0 对比**：

| 层 | Mistral v0.3 L0 | Llama 3.1 L0 |
|---|---|---|
| L8 | 71.2 | 28.7 |
| L16 | 65.2 | 35.5 |
| L22 | 76.6 | 40.2 |

Llama 整体比 Mistral 稀疏约一倍——同样 4x expansion，Llama 每 token 只需一半特征。可能原因：Llama 128K 词表的 tokenizer 让每个 token 承载更多信息，减少冗余。

**两个模型都能干净分离 JSON 结构特征，跨模型复现成立。**

### Llama HuggingFace 上传

- [lmxxf/llama31-8b-sae-layer8](https://huggingface.co/lmxxf/llama31-8b-sae-layer8)
- [lmxxf/llama31-8b-sae-layer16](https://huggingface.co/lmxxf/llama31-8b-sae-layer16)
- [lmxxf/llama31-8b-sae-layer22](https://huggingface.co/lmxxf/llama31-8b-sae-layer22)

### 当前状态（2026-04-24）

- ✅ Mistral-7B-Instruct-v0.3 三层 SAE 训练 + 验证 + 上传完成
- ✅ Llama-3.1-8B-Instruct 三层 SAE 训练 + 验证 + 上传完成
- **两模型六个 SAE 全部就位**

### 文件（更新）

| 文件 | 用途 |
|---|---|
| `train_sae.py` | Mistral SAE 训练脚本（v0.3 + lmsys） |
| `train_sae_llama.py` | **Llama SAE 训练脚本（新增）** |
| `prepare_lmsys_dataset.py` | lmsys 数据集预处理（Mistral） |
| `prepare_lmsys_llama.py` | **lmsys 数据集预处理（Llama，新增）** |
| `validate_sae.py` | SAE 验证 v1 |
| `validate_sae_v2.py` | SAE 验证 v2（**已改：支持 --model llama**） |
| `intervene_sae.py` | SAE 特征干预实验（单层 L16） |
| `intervene_multilayer.py` | 三层联合干预实验 |
| `output/` | Layer 16 SAE 权重（旧版 v0.1） |
| `sae_checkpoints/mistral7b_sae_L{8,16,22}_64k/` | **Mistral v0.3 SAE 权重 ✅** |
| `sae_checkpoints/llama31_8b_sae_L{8,16,22}_16k/` | **Llama 3.1 SAE 权重 ✅** |

### 下一步

- ZL 用新 SAE 重跑干预实验（Mistral + Llama 跨模型对比）
- ZL 同步推 format specificity 对照实验（JSON vs XML/YAML）
- EMNLP 2026 主会投稿（5/25 截止）
