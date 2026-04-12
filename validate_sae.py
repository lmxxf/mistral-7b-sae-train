"""
Mistral-7B SAE 验证脚本
验证已训好的 Layer 16 SAE 的质量：重建质量、稀疏度、特征可解释性

用法：
  python validate_sae.py

硬件：DGX Spark GB10, 128GB 统一内存
"""

import os

# 强制离线模式，容器里没网
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import torch
import numpy as np

# ===== 路径 =====
MODEL_PATH = "/workspace/models/Mistral-7B-Instruct-v0.1"
SAE_PATH = "/workspace/mistral-7b-sae-train/output/"
HOOK_NAME = "blocks.16.hook_resid_post"


# ===== 环境配置（复用 train_sae.py 的逻辑） =====

def setup_environment():
    """设置离线环境：rope_theta monkey-patch + HF cache 软链"""
    import transformers

    # 修补 transformers 5.x 兼容性：新版删了 rope_theta，TransformerLens 还在用
    _orig_from_pretrained = transformers.AutoConfig.from_pretrained

    @staticmethod
    def _patched_from_pretrained(*args, **kwargs):
        result = _orig_from_pretrained(*args, **kwargs)
        # 有些调用方传 return_unused_kwargs=True，返回 (config, kwargs) tuple
        if isinstance(result, tuple):
            config = result[0]
        else:
            config = result
        if not hasattr(config, "rope_theta"):
            config_path = os.path.join(MODEL_PATH, "config.json")
            with open(config_path) as f:
                raw = json.load(f)
            config.rope_theta = raw.get("rope_theta", 10000.0)
        return result

    transformers.AutoConfig.from_pretrained = _patched_from_pretrained

    # 把本地模型路径注入 HF cache，让 TransformerLens 的 AutoConfig 能离线找到
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    link_name = os.path.join(cache_dir, "models--mistralai--Mistral-7B-Instruct-v0.1")
    if not os.path.exists(link_name):
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(link_name, exist_ok=True)
        snapshot_dir = os.path.join(link_name, "snapshots", "local")
        os.makedirs(snapshot_dir, exist_ok=True)
        for f in os.listdir(MODEL_PATH):
            src = os.path.join(MODEL_PATH, f)
            dst = os.path.join(snapshot_dir, f)
            if not os.path.exists(dst) and os.path.isfile(src):
                os.symlink(src, dst)
        refs_dir = os.path.join(link_name, "refs")
        os.makedirs(refs_dir, exist_ok=True)
        with open(os.path.join(refs_dir, "main"), "w") as fp:
            fp.write("local")
        print(f"已创建 HF cache 软链: {link_name}")


# ===== 测试文本 =====

JSON_TEXTS = [
    # 典型 API response
    '{"status": "success", "data": {"user_id": 12345, "name": "Alice", "email": "alice@example.com", "roles": ["admin", "editor"], "settings": {"theme": "dark", "notifications": true}}}',

    # 配置文件风格
    '{"version": "2.1.0", "database": {"host": "localhost", "port": 5432, "name": "mydb", "credentials": {"username": "app_user", "password": "secret123"}}, "cache": {"enabled": true, "ttl": 3600, "backend": "redis"}}',

    # 嵌套数组
    '{"results": [{"id": 1, "title": "First Post", "tags": ["python", "ml"]}, {"id": 2, "title": "Second Post", "tags": ["rust", "systems"]}, {"id": 3, "title": "Third Post", "tags": ["javascript", "web"]}], "total": 3, "page": 1}',

    # 简短 JSON
    '{"error": {"code": 404, "message": "Resource not found", "details": [{"field": "id", "issue": "invalid format"}]}}',
]

NATURAL_TEXTS = [
    "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet and has been used as a typing exercise for many years.",

    "Machine learning models have revolutionized natural language processing. Transformer architectures, introduced in 2017, enabled unprecedented performance on tasks like translation, summarization, and question answering.",

    "In the early morning light, the city slowly awakens. Coffee shops open their doors, buses begin their routes, and joggers appear on the sidewalks. The rhythm of urban life resumes after the quiet of night.",

    "The theory of relativity fundamentally changed our understanding of space and time. Einstein showed that the speed of light is constant in all reference frames, leading to surprising consequences like time dilation and length contraction.",
]


def load_model():
    """加载 Mistral-7B 模型（HookedTransformer）"""
    from transformers import AutoModelForCausalLM
    from transformer_lens import HookedTransformer

    print("加载 HF 模型...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16
    )

    print("转换为 HookedTransformer...")
    model = HookedTransformer.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.1",
        hf_model=hf_model,
        center_writing_weights=False,
        device="cuda",
        dtype=torch.float16,
    )
    model.eval()

    # 释放 HF 模型，节省内存
    del hf_model
    torch.cuda.empty_cache()

    print("模型加载完成\n")
    return model


def load_sae():
    """加载训好的 SAE"""
    from sae_lens import SAE

    print("加载 SAE...")
    sae = SAE.load_from_pretrained(SAE_PATH, device="cuda")
    sae.eval()
    print(f"SAE 加载完成: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}\n")
    return sae


def get_activations(model, texts):
    """获取一批文本在指定 hook 的激活"""
    all_activations = []
    all_tokens = []

    for text in texts:
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(tokens, names_filter=[HOOK_NAME])
        activations = cache[HOOK_NAME]  # [1, seq_len, d_in]
        all_activations.append(activations.squeeze(0))  # [seq_len, d_in]
        all_tokens.append(tokens.squeeze(0))  # [seq_len]

    return all_activations, all_tokens


# ===== 验证 1：重建质量 =====

def validate_reconstruction(sae, all_activations):
    """计算 SAE 重建质量：explained variance 和 MSE"""
    print("=" * 60)
    print("  验证 1：重建质量（Reconstruction Quality）")
    print("=" * 60)

    # 把所有激活拼起来
    activations = torch.cat(all_activations, dim=0).to(torch.float32)  # [total_tokens, d_in]

    with torch.no_grad():
        # SAE 前向：编码 + 解码
        sae_out = sae(activations)
        # sae() 可能返回 tuple (reconstructed, ...) 或直接返回 tensor
        if isinstance(sae_out, tuple):
            reconstructed = sae_out[0]
        else:
            reconstructed = sae_out

    # MSE
    mse = torch.mean((activations - reconstructed) ** 2).item()

    # Explained variance = 1 - MSE / Var(original)
    var_original = torch.var(activations).item()
    explained_variance = 1.0 - mse / var_original

    print(f"\n  总 token 数:       {activations.shape[0]}")
    print(f"  MSE:               {mse:.6f}")
    print(f"  原始方差:          {var_original:.6f}")
    print(f"  Explained Variance: {explained_variance:.4f}")

    if explained_variance > 0.85:
        print(f"\n  [PASS] Explained Variance = {explained_variance:.4f} > 0.85")
    else:
        print(f"\n  [FAIL] Explained Variance = {explained_variance:.4f} < 0.85")

    print()
    return explained_variance, mse


# ===== 验证 2：稀疏度 =====

def validate_sparsity(sae, all_activations):
    """计算 L0 稀疏度"""
    print("=" * 60)
    print("  验证 2：稀疏度（Sparsity / L0）")
    print("=" * 60)

    activations = torch.cat(all_activations, dim=0).to(torch.float32)

    with torch.no_grad():
        # 获取 SAE 编码后的特征激活
        feature_acts = sae.encode(activations)  # [total_tokens, d_sae]

    # L0 = 每个 token 的非零特征数的平均值
    nonzero_per_token = (feature_acts > 0).float().sum(dim=-1)  # [total_tokens]
    l0_mean = nonzero_per_token.mean().item()
    l0_std = nonzero_per_token.std().item()
    l0_min = nonzero_per_token.min().item()
    l0_max = nonzero_per_token.max().item()

    print(f"\n  总 token 数:   {activations.shape[0]}")
    print(f"  字典大小:     {sae.cfg.d_sae}")
    print(f"  L0 均值:       {l0_mean:.1f}")
    print(f"  L0 标准差:     {l0_std:.1f}")
    print(f"  L0 范围:       [{l0_min:.0f}, {l0_max:.0f}]")
    print(f"  L0/d_sae:      {l0_mean / sae.cfg.d_sae:.4f}")

    if 50 <= l0_mean <= 500:
        print(f"\n  [PASS] L0 = {l0_mean:.1f}, 在合理范围 [50, 500]")
    else:
        print(f"\n  [WARN] L0 = {l0_mean:.1f}, 超出典型范围 [50, 500]")

    print()
    return l0_mean


# ===== 验证 3：特征可解释性 =====

def validate_interpretability(model, sae, json_texts, natural_texts):
    """对比 JSON vs 自然语言文本的特征激活模式"""
    print("=" * 60)
    print("  验证 3：特征可解释性（Feature Interpretability）")
    print("=" * 60)

    for label, texts in [("JSON", json_texts), ("自然语言", natural_texts)]:
        print(f"\n{'─' * 50}")
        print(f"  文本类型: {label}")
        print(f"{'─' * 50}")

        # 收集所有 token 的特征激活
        all_feature_acts = []
        all_token_strs = []

        for text in texts:
            tokens = model.to_tokens(text)
            _, cache = model.run_with_cache(tokens, names_filter=[HOOK_NAME])
            activations = cache[HOOK_NAME].squeeze(0).to(torch.float32)  # [seq_len, d_in]

            with torch.no_grad():
                feature_acts = sae.encode(activations)  # [seq_len, d_sae]

            # 解码 token 为字符串
            token_ids = tokens.squeeze(0)
            token_strs = [model.tokenizer.decode([t.item()]) for t in token_ids]

            all_feature_acts.append(feature_acts)
            all_token_strs.extend(token_strs)

        # 拼接所有特征激活
        all_feature_acts = torch.cat(all_feature_acts, dim=0)  # [total_tokens, d_sae]

        # 找每个特征在所有 token 上的最大激活值，然后取 top-20 特征
        max_act_per_feature = all_feature_acts.max(dim=0).values  # [d_sae]
        top_k = 20
        top_feature_indices = torch.topk(max_act_per_feature, top_k).indices

        print(f"\n  Top-{top_k} 特征（按最大激活值排序）:\n")

        for rank, feat_idx in enumerate(top_feature_indices):
            feat_idx = feat_idx.item()
            feat_acts_for_this = all_feature_acts[:, feat_idx]  # [total_tokens]
            max_val = feat_acts_for_this.max().item()

            # 找该特征激活最强的 top-5 token
            top_token_k = min(5, len(all_token_strs))
            top_token_indices = torch.topk(feat_acts_for_this, top_token_k).indices

            top_tokens_info = []
            for ti in top_token_indices:
                ti = ti.item()
                token_str = all_token_strs[ti]
                act_val = feat_acts_for_this[ti].item()
                if act_val > 0:
                    top_tokens_info.append(f"'{token_str}'({act_val:.2f})")

            tokens_display = ", ".join(top_tokens_info) if top_tokens_info else "(无激活)"
            print(f"  #{rank+1:2d}  Feature {feat_idx:5d}  max={max_val:.2f}  "
                  f"top tokens: {tokens_display}")

    # 额外分析：找 JSON 结构字符的专属特征
    print(f"\n{'─' * 50}")
    print(f"  JSON 结构特征分析")
    print(f"{'─' * 50}")
    print(f"  寻找在 JSON 标点 ({{ }} [ ] \" : ,) 上特别活跃的特征...\n")

    json_punct_chars = {'{', '}', '[', ']', '"', ':', ','}

    # 获取 JSON 文本的特征激活和 token
    json_feature_acts = []
    json_token_strs = []
    json_is_punct = []

    for text in json_texts:
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(tokens, names_filter=[HOOK_NAME])
        activations = cache[HOOK_NAME].squeeze(0).to(torch.float32)

        with torch.no_grad():
            feature_acts = sae.encode(activations)

        token_ids = tokens.squeeze(0)
        token_strs = [model.tokenizer.decode([t.item()]) for t in token_ids]

        json_feature_acts.append(feature_acts)
        json_token_strs.extend(token_strs)
        json_is_punct.extend([
            any(c in t for c in json_punct_chars) for t in token_strs
        ])

    json_feature_acts = torch.cat(json_feature_acts, dim=0)  # [n_tokens, d_sae]
    json_is_punct = torch.tensor(json_is_punct, dtype=torch.bool)

    if json_is_punct.any():
        # 计算每个特征在 JSON 标点 token vs 非标点 token 上的平均激活比
        punct_acts = json_feature_acts[json_is_punct].mean(dim=0)    # [d_sae]
        nonpunct_acts = json_feature_acts[~json_is_punct].mean(dim=0)  # [d_sae]

        # 避免除零
        ratio = punct_acts / (nonpunct_acts + 1e-8)

        # 只看那些在标点上确实有明显激活的特征
        active_mask = punct_acts > 0.1
        ratio_masked = ratio * active_mask.float()

        top_json_k = 10
        top_json_features = torch.topk(ratio_masked, top_json_k).indices

        print(f"  Top-{top_json_k} JSON 标点偏好特征（标点/非标点激活比）:\n")
        for rank, feat_idx in enumerate(top_json_features):
            feat_idx = feat_idx.item()
            r = ratio[feat_idx].item()
            p_act = punct_acts[feat_idx].item()
            np_act = nonpunct_acts[feat_idx].item()

            # 找该特征在哪些标点 token 上最活跃
            punct_indices = torch.where(json_is_punct)[0]
            punct_feature_vals = json_feature_acts[punct_indices, feat_idx]
            top3_idx = torch.topk(punct_feature_vals, min(3, len(punct_indices))).indices
            top_punct_tokens = [json_token_strs[punct_indices[i].item()] for i in top3_idx]
            top_punct_display = ", ".join([f"'{t}'" for t in top_punct_tokens])

            print(f"  #{rank+1:2d}  Feature {feat_idx:5d}  "
                  f"ratio={r:.2f}  punct_mean={p_act:.3f}  other_mean={np_act:.3f}  "
                  f"top_punct: {top_punct_display}")
    else:
        print("  (未检测到 JSON 标点 token)")

    print()


# ===== 主流程 =====

def main():
    print("\n" + "=" * 60)
    print("  Mistral-7B Layer 16 SAE 验证")
    print("  SAE 路径: " + SAE_PATH)
    print("=" * 60 + "\n")

    # 环境配置
    setup_environment()

    # 加载模型和 SAE
    model = load_model()
    sae = load_sae()

    # 收集所有文本的激活（验证 1 和 2 共用）
    print("收集激活数据...")
    all_texts = JSON_TEXTS + NATURAL_TEXTS
    all_activations, all_tokens = get_activations(model, all_texts)
    print(f"共 {sum(a.shape[0] for a in all_activations)} 个 token\n")

    # 验证 1：重建质量
    ev, mse = validate_reconstruction(sae, all_activations)

    # 验证 2：稀疏度
    l0 = validate_sparsity(sae, all_activations)

    # 验证 3：特征可解释性
    validate_interpretability(model, sae, JSON_TEXTS, NATURAL_TEXTS)

    # 总结
    print("=" * 60)
    print("  验证总结")
    print("=" * 60)
    print(f"\n  Explained Variance:  {ev:.4f}  {'[PASS]' if ev > 0.85 else '[FAIL]'}")
    print(f"  MSE:                 {mse:.6f}")
    print(f"  L0 (avg features):   {l0:.1f}    {'[PASS]' if 50 <= l0 <= 500 else '[WARN]'}")
    print(f"\n  请检查上方特征可解释性输出，确认是否存在 JSON 结构特征。")
    print()


if __name__ == "__main__":
    main()
