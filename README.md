# Mistral-7B Sparse Autoencoders (JumpReLU) — Layer 8, 16, 22

Sparse Autoencoders (SAE) trained on residual stream activations of Mistral-7B-Instruct-v0.1 at three layers (shallow / middle / deep), using the JumpReLU architecture from Anthropic's *Scaling Monosemanticity*.

在 Mistral-7B-Instruct-v0.1 的三层残差流（浅层/中层/深层）上训练的稀疏自编码器（SAE），使用 Anthropic *Scaling Monosemanticity* 论文中的 JumpReLU 架构。

**Weights / 权重**:
- Layer 8: [lmxxf/mistral-7b-sae-layer8](https://huggingface.co/lmxxf/mistral-7b-sae-layer8)
- Layer 16: [lmxxf/mistral-7b-sae-layer16](https://huggingface.co/lmxxf/mistral-7b-sae-layer16)
- Layer 22: [lmxxf/mistral-7b-sae-layer22](https://huggingface.co/lmxxf/mistral-7b-sae-layer22)

## Motivation / 动机

The 4096-dim residual stream of Mistral-7B encodes thousands of concepts in superposition — JSON structure, natural language semantics, syntax, etc. all tangled together. This SAE expands the residual stream to 16384 dimensions, disentangling these concepts into distinct sparse features suitable for targeted intervention.

Mistral-7B 的 4096 维残差流把成千上万个概念叠加在一起——JSON 结构、自然语言语义、语法等全搅在一起。这个 SAE 把残差流升维到 16384 维，将纠缠的概念解耦到不同的稀疏特征上，使精确干预成为可能。

## Training Configuration / 训练配置

| Parameter / 参数 | Value / 值 |
|---|---|
| Base model / 基座模型 | `mistralai/Mistral-7B-Instruct-v0.1` (HuggingFace) |
| Training dataset / 训练数据 | `Skylion007/openwebtext` (HuggingFace) |
| Target layers / 目标层 | Layer 8, 16, 22 (`blocks.{8,16,22}.hook_resid_post`) |
| SAE architecture / SAE 架构 | JumpReLU |
| Dictionary size / 字典大小 | 16,384 (4x expansion) |
| Input dimension / 输入维度 | 4,096 |
| Training tokens / 训练 token 数 | ~49M |
| Training steps / 训练步数 | 12,000 |
| Learning rate / 学习率 | 5e-5 (constant + 1000-step warmup + 20% tail decay) |
| Model precision / 模型精度 | float16 (autocast) |
| SAE precision / SAE 精度 | float32 |
| Context size / 上下文长度 | 256 |
| Activation normalization / 激活归一化 | `expected_average_only_in` |
| Training time / 训练时长 | 5~11.5 hours per layer on DGX Spark GB10 |

## Validation Results / 验证结果

### Quantitative Metrics / 定量指标 (excluding BOS / 排除 BOS)

| Layer / 层 | MSE / 均方误差 | L0 | JSON features / JSON 特征 |
|---|---|---|---|
| **L8** (shallow / 浅层) | 0.0019 | 33.4 | ratio > 1e8 ✅ |
| **L16** (middle / 中层) | 0.0116 | 56.9 | ratio > 1e8 ✅ |
| **L22** (deep / 深层) | 0.0525 | 57.7 | ratio > 1e8 ✅ |

MSE increases with depth (deeper layers encode denser information). All three layers have clean JSON structure features with activation ratios > 100 million.

MSE 随深度递增（深层编码更密集的信息）。三层都找到了干净的 JSON 结构特征，激活比值超过 1 亿。

### Feature Interpretability / 特征可解释性

The SAE cleanly separates JSON structural features from natural language semantic features. Top-20 features for each domain show **near-zero overlap**. Shallow and deep layers encode different types of JSON features.

SAE 将 JSON 结构特征和自然语言语义特征干净地分离。两类文本的 top-20 特征**几乎零重叠**。浅层和深层编码了不同类型的 JSON 特征。

**Shallow (L8) — syntax-level / 浅层——语法级**:

| Feature / 特征 | Activation pattern / 激活模式 |
|---|---|
| 5965 / 8734 | `":` — key-value separator / 键值分隔符 |
| 15265 | `}}`, `}}}` — closing braces / 闭合括号 |
| 6302 | `,` — comma / 逗号 |

**Middle (L16) — syntax-level / 中层——语法级**:

| Feature / 特征 | Activation pattern / 激活模式 |
|---|---|
| 1000 | `[` — array brackets / 数组括号 |
| 11062 | `{"` — JSON object opening / JSON 对象开头 |
| 4765 | `}`, `}}}` — closing braces / 闭合括号 |
| 2918 | `"`, `":` — quotes / key-value separators / 引号/键值分隔符 |
| 200 | `id`, `title`, `tags` — JSON key names / JSON 键名 |

**Deep (L22) — semantic-level / 深层——语义级**:

| Feature / 特征 | Activation pattern / 激活模式 |
|---|---|
| 454 | `system`, `app` — programming concepts / 编程概念 |
| 14251 | `Al`, `alice` — named entities / 人名实体 |
| 13416 | `cred`, `entials` — credentials / 凭证 |
| 6436 | `First`, `Second` — ordinals / 序数 |

**Natural language features / 自然语言特征**:

| Feature / 特征 | Activation pattern / 激活模式 |
|---|---|
| 5088 | Sentence-initial words / 句首词: `In`, `The` |
| 16167 | Temporal concepts / 时间概念: `early`, `morning` |
| 8350 | Time/light cluster / 时间/光线语义簇: `morning`, `night`, `light` |

### Feature Intervention / 特征干预实验

Ablating JSON structure features at the Layer 16 hook point (SAE encode → clamp features to 0 → SAE decode → replace activations) successfully disrupts JSON formatting while preserving semantic content:

在 Layer 16 hook 点消融 JSON 结构特征（SAE 编码 → 目标特征置零 → SAE 解码 → 替换激活），成功破坏了 JSON 格式，同时语义内容完整保留：

- `book` prompt + heavy ablation (10 features): `"title": "The Great Gatsby", "author: "F. Scott Fitzgerald "pages: 176` — SC fails, semantics intact / 结构崩，语义完整
- `person` prompt + heavy ablation: `"name": "John", "age: 25 "city": "New York"` — same pattern / 同上

## Usage / 使用方法

```python
from sae_lens import SAE

sae = SAE.load_from_pretrained(
    "path/to/downloaded/weights/",
    device="cuda",
)

# sae.encode(activations) -> sparse features / 稀疏特征
# sae.decode(features) -> reconstructed activations / 重建激活
```

Full pipeline with TransformerLens / 完整流程：

```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
_, cache = model.run_with_cache("Hello world", prepend_bos=True)

layer_16_act = cache["blocks.16.hook_resid_post"]
features = sae.encode(layer_16_act)  # (batch, seq, 16384)
reconstructed = sae.decode(features)
```

## Environment / 环境

| Component / 组件 | Version / 版本 |
|---|---|
| Hardware / 硬件 | NVIDIA DGX Spark GB10, 128GB unified memory / 统一内存 |
| PyTorch | 2.10.0 + CUDA 13.0 |
| SAELens | 6.39.0 |
| TransformerLens | 2.16.1 |
| Transformers | 5.3.0 |
| Container / 容器 | `nvcr.io/nvidia/pytorch:25.11-py3` |

## License / 许可

This SAE is released for research purposes. The base model (Mistral-7B-Instruct-v0.1) is subject to its own [license](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1).

本 SAE 用于研究目的开源。基座模型 Mistral-7B-Instruct-v0.1 遵循其自身的[许可协议](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)。
