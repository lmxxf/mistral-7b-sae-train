# Mistral-7B Layer 16 Sparse Autoencoder (JumpReLU)

A Sparse Autoencoder (SAE) trained on layer 16 residual stream activations of Mistral-7B-Instruct-v0.1, using the JumpReLU architecture from Anthropic's *Scaling Monosemanticity*.

**Weights**: [lmxxf/mistral-7b-sae-layer16](https://huggingface.co/lmxxf/mistral-7b-sae-layer16)

## Motivation

LEACE-based linear erasure experiments on Mistral-7B failed to separate JSON structural features from natural language semantic features — all 20+ configurations exhibited sharp phase transitions where both capabilities collapsed simultaneously (subspace entanglement EUO = 1.59). A linear knife cannot cut entangled subspaces. This SAE expands the 4096-dim residual stream to 16384 dimensions, disentangling structure and semantics into distinct sparse features suitable for targeted intervention.

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | `mistralai/Mistral-7B-Instruct-v0.1` |
| Target layer | Layer 16 (`blocks.16.hook_resid_post`) |
| SAE architecture | JumpReLU |
| Dictionary size | 16,384 (4x expansion) |
| Input dimension | 4,096 |
| Training tokens | ~49M |
| Training steps | 12,000 |
| Learning rate | 5e-5 (constant + 1000-step warmup + 20% tail decay) |
| Model precision | float16 (autocast) |
| SAE precision | float32 |
| Context size | 256 |
| Dataset | OpenWebText |
| Activation normalization | `expected_average_only_in` |
| Training time | ~10 hours |

## Validation Results

### Quantitative Metrics

| Metric | With BOS | Excluding BOS | Notes |
|---|---|---|---|
| Explained Variance | **0.9622** | 0.4747 | EV drops after BOS removal due to low residual variance (0.30 → 0.02), not reconstruction quality |
| MSE | 0.0114 | 0.0116 | Nearly identical — reconstruction quality is robust |
| L0 (avg active features) | 79.4 | 56.9 | 57–79 / 16,384 features active per token — good sparsity |

### Feature Interpretability

The SAE cleanly separates JSON structural features from natural language semantic features. Top-20 features for each domain show **near-zero overlap**.

**JSON structure features** (examples):
| Feature | Activation pattern |
|---|---|
| 1000 | `[` — array brackets |
| 11062 | `{"` — JSON object opening |
| 4765 | `}`, `}}}` — closing braces |
| 2918 | `"`, `":` — quotes / key-value separators |
| 200 | `id`, `title`, `tags` — JSON key names |
| 9459 | `_` — underscores in `user_id`, `app_user` |

**Natural language features** (examples):
| Feature | Activation pattern |
|---|---|
| 5088 | Sentence-initial words: `In`, `The` |
| 16167 | Temporal concepts: `early`, `morning` |
| 8350 | Time/light semantic cluster: `morning`, `night`, `light` |

JSON punctuation–preferring features (top-10) all have activation ratio > 1e8 — they fire exclusively on JSON punctuation with zero activation on non-punctuation tokens.

## Usage

```python
from sae_lens import SAE

sae = SAE.from_pretrained(
    release="lmxxf/mistral-7b-sae-layer16",
    sae_id=".",
)

# sae.encode(activations) -> sparse features
# sae.decode(features) -> reconstructed activations
```

For a full pipeline with TransformerLens:

```python
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
_, cache = model.run_with_cache("Hello world", prepend_bos=True)

layer_16_act = cache["blocks.16.hook_resid_post"]
features = sae.encode(layer_16_act)  # (batch, seq, 16384)
reconstructed = sae.decode(features)
```

## Environment

| Component | Version |
|---|---|
| Hardware | NVIDIA DGX Spark GB10, 128GB unified memory |
| PyTorch | 2.10.0 + CUDA 13.0 |
| SAELens | 6.39.0 |
| TransformerLens | 2.16.1 |
| Transformers | 5.3.0 |
| Container | `nvcr.io/nvidia/pytorch:25.11-py3` |

## License

This SAE is released for research purposes. The base model (Mistral-7B-Instruct-v0.1) is subject to its own [license](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1).
