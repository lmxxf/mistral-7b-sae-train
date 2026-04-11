# Mistral-7B SAE 训练 — 开发日志

## 2026-04-10

### 起因

赵磊用 LEACE 做 Mistral-7B 的 SC/SemC 双分离实验全翻车——20+ 配置都是尖锐相变，结构和语义一起崩。诊断结论：Mistral 的结构子空间和语义子空间纠缠（EUO=1.59），线性刀切不开。方案：训 SAE 升维解耦，精确干预 JSON 结构特征。

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

### 下一步

- 用新参数重跑 `python train_sae.py`（默认 layer 16），看能不能跑完 12000 步
- 跑通后再考虑扩层
- 目的已从赵磊的论文实验转为**写公众号**（SAE 科普 + Mistral 内部结构可视化）
