"""
Mistral-7B SAE 特征干预实验
在 Layer 16 hook 点关掉 JSON 结构特征，观察模型生成 JSON 的能力是否崩溃，
同时检查语义是否保留。

证明：LEACE 切不开的结构/语义纠缠，SAE 升维后能精确切开。

用法：
  python intervene_sae.py

硬件：DGX Spark GB10, 128GB 统一内存
"""

import os

# 强制离线模式，容器里没网
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import re
import torch
from functools import partial

# ===== 路径 =====
MODEL_PATH = "/workspace/models/Mistral-7B-Instruct-v0.1"
SAE_PATH = "/workspace/mistral-7b-sae-train/output/"
HOOK_NAME = "blocks.16.hook_resid_post"


# ===== 环境配置（复用 validate_sae.py） =====

def setup_environment():
    """设置离线环境：rope_theta monkey-patch + HF cache 软链"""
    import transformers

    _orig_from_pretrained = transformers.AutoConfig.from_pretrained

    @staticmethod
    def _patched_from_pretrained(*args, **kwargs):
        result = _orig_from_pretrained(*args, **kwargs)
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


# ===== 测试 Prompt =====

SEMANTIC_PROMPTS = [
    {
        "prompt": "Generate a JSON object representing a car with fields: make, model, and year. Respond with ONLY the JSON, no explanation.",
        "expected_keys": ["make", "model", "year"],
        "content_words": ["toyota", "honda", "ford", "bmw", "tesla", "car",
                          "sedan", "suv", "2020", "2021", "2022", "2023", "2024"],
    },
    {
        "prompt": "Generate a JSON object representing a person with fields: name, age, and city. Respond with ONLY the JSON, no explanation.",
        "expected_keys": ["name", "age", "city"],
        "content_words": ["john", "jane", "alice", "bob", "new york", "london",
                          "tokyo", "paris", "san francisco"],
    },
    {
        "prompt": "Generate a JSON object for a book with fields: title, author, pages, and published. Respond with ONLY the JSON, no explanation.",
        "expected_keys": ["title", "author", "pages", "published"],
        "content_words": ["book", "novel", "story", "chapter", "page",
                          "author", "write", "publish"],
    },
    {
        "prompt": "Generate a JSON object for a restaurant with fields: name, cuisine, rating, and address. Respond with ONLY the JSON, no explanation.",
        "expected_keys": ["name", "cuisine", "rating", "address"],
        "content_words": ["restaurant", "food", "italian", "chinese", "french",
                          "japanese", "mexican", "street", "avenue"],
    },
    {
        "prompt": "Generate a JSON object for a school with fields: name, address, founded, and a teachers array of 3 objects each with name and subject. Respond with ONLY the JSON, no explanation.",
        "expected_keys": ["name", "address", "founded", "teachers"],
        "content_words": ["school", "academy", "university", "math", "science",
                          "english", "history", "teacher", "professor"],
    },
]

NONSEMANTIC_PROMPTS = [
    {
        "prompt": "Generate a JSON object representing a blonf with fields: zrelk, grimbat, and quav. Respond with ONLY the JSON, no explanation.",
        "expected_keys": ["zrelk", "grimbat", "quav"],
        "content_words": [],  # 无意义字段，不检查语义
    },
    {
        "prompt": "Generate a JSON object representing a trelm with fields: plovk, dranq, and blixt. Respond with ONLY the JSON, no explanation.",
        "expected_keys": ["plovk", "dranq", "blixt"],
        "content_words": [],
    },
]

ALL_PROMPTS = SEMANTIC_PROMPTS + NONSEMANTIC_PROMPTS

# ===== 干预特征配置 =====

# 从 validate_sae.py 的 top-10 JSON 标点偏好特征
INTERVENTION_LEVELS = {
    "baseline": [],
    "light": [2122],                          # 只关 ": 键值分隔符
    "medium": [2122, 9353, 1000],             # + {" 和 [
    "heavy": [14474, 15086, 442, 1978, 8666,  # 全部 top-10 JSON 标点偏好特征
              5243, 2122, 9353, 16094, 9582],
}


# ===== Hook 函数 =====

def make_intervention_hook(sae, features_to_ablate):
    """
    创建干预 hook：在 Layer 16 residual stream 上做 SAE 特征消融。

    流程：
    1. 拦截 blocks.16.hook_resid_post 的激活
    2. SAE encode -> 16384 维特征
    3. 将目标特征 clamp 到 0
    4. SAE decode -> 重建回 4096 维
    5. 用重建激活替换原始激活
    """
    features_to_ablate = list(features_to_ablate)

    def hook_fn(activation, hook):
        # activation: [batch, seq_len, d_model]
        orig_dtype = activation.dtype
        act_float = activation.to(torch.float32)

        # 记录原始形状，展平处理
        batch, seq_len, d_model = act_float.shape
        act_flat = act_float.reshape(-1, d_model)  # [batch*seq, d_model]

        with torch.no_grad():
            # SAE encode
            feature_acts = sae.encode(act_flat)  # [batch*seq, d_sae]

            # 消融目标特征
            if features_to_ablate:
                feature_acts[:, features_to_ablate] = 0.0

            # SAE decode
            reconstructed = sae.decode(feature_acts)  # [batch*seq, d_model]

        # 恢复形状和精度
        reconstructed = reconstructed.reshape(batch, seq_len, d_model)
        return reconstructed.to(orig_dtype)

    return hook_fn


# ===== 生成函数 =====

def generate_with_intervention(model, sae, prompt, features_to_ablate,
                               max_new_tokens=200):
    """
    带 SAE 特征干预的文本生成。

    使用 model.generate()，hook 会在每个生成步骤的前向传播中被调用。
    temperature=0 (贪婪解码) 确保可复现。
    """
    # 构造 Mistral instruct 格式
    formatted = f"[INST] {prompt} [/INST]"
    tokens = model.to_tokens(formatted)

    if not features_to_ablate:
        # baseline：无干预直接生成
        output_tokens = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
        )
    else:
        # 有干预：挂 hook 后生成
        hook_fn = make_intervention_hook(sae, features_to_ablate)
        with model.hooks(fwd_hooks=[(HOOK_NAME, hook_fn)]):
            output_tokens = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
            )

    # 解码生成的 token（跳过 prompt 部分）
    prompt_len = tokens.shape[1]
    generated_tokens = output_tokens[0, prompt_len:]
    generated_text = model.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text.strip()


# ===== 评估函数 =====

def strip_markdown_codeblock(text):
    """去除 markdown code block 包裹：```json ... ``` 或 ``` ... ```"""
    stripped = text.strip()
    # 匹配 ```json\n...\n``` 或 ```\n...\n```
    pattern = r'^```(?:json)?\s*\n?(.*?)\n?\s*```$'
    match = re.match(pattern, stripped, re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


def evaluate_structural_compliance(text):
    """SC: json.loads() 能否解析输出"""
    cleaned = strip_markdown_codeblock(text)
    try:
        json.loads(cleaned)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def evaluate_semantic_correctness(text, content_words):
    """
    SemC: 语义正确率。
    对语义类 prompt，检查期望内容词是否出现在输出中。
    返回 0.0 ~ 1.0 的分数。
    """
    if not content_words:
        return None  # 无意义 prompt 不评估语义

    text_lower = text.lower()
    hits = sum(1 for word in content_words if word.lower() in text_lower)

    # 我们不要求所有词都出现，用命中比例做分数
    # 但也需要至少命中一些才算语义保留
    return hits / len(content_words)


def evaluate_collapse(text):
    """
    Collapse 检测：
    - 内容词 < 3 个独立 token
    - 或重复率太高（同一 3-gram 出现超过总 3-gram 数的 50%）
    返回 True 表示检测到崩溃。
    """
    words = text.split()

    # 太短
    if len(words) < 3:
        return True

    # 空内容
    if not text.strip():
        return True

    # 重复率检测：3-gram 分析
    if len(words) >= 3:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        if trigrams:
            from collections import Counter
            counter = Counter(trigrams)
            most_common_count = counter.most_common(1)[0][1]
            if most_common_count > len(trigrams) * 0.5 and most_common_count > 3:
                return True

    return False


# ===== 主实验 =====

def run_experiment(model, sae):
    """运行完整的干预实验"""

    results = []  # 收集所有结果用于汇总

    for prompt_info in ALL_PROMPTS:
        prompt = prompt_info["prompt"]
        expected_keys = prompt_info["expected_keys"]
        content_words = prompt_info["content_words"]
        is_semantic = len(content_words) > 0

        # 截断 prompt 显示
        display_prompt = prompt[:80] + "..." if len(prompt) > 80 else prompt

        print(f"\n{'=' * 80}")
        print(f"Prompt: \"{display_prompt}\"")
        print(f"  Type: {'Semantic' if is_semantic else 'Nonsemantic'}")
        print(f"  Expected keys: {expected_keys}")
        print(f"{'─' * 80}")

        for level_name, features in INTERVENTION_LEVELS.items():
            # 生成
            output = generate_with_intervention(
                model, sae, prompt, features, max_new_tokens=200
            )

            # 评估
            sc = evaluate_structural_compliance(output)
            semc = evaluate_semantic_correctness(output, content_words)
            collapsed = evaluate_collapse(output)

            # 格式化显示
            sc_str = "SC=\u2705" if sc else "SC=\u274c"
            semc_str = f"SemC={semc:.2f}" if semc is not None else "SemC=N/A"
            collapse_str = " [COLLAPSED]" if collapsed else ""
            feat_count = len(features)
            level_display = f"{level_name} ({feat_count} feat)" if features else "baseline"

            # 截断输出显示
            output_display = output[:100].replace("\n", "\\n")
            if len(output) > 100:
                output_display += "..."

            print(f"  {level_display:20s}  {sc_str}  {semc_str:12s}"
                  f"{collapse_str}")
            print(f"    Output: {output_display}")

            # 记录结果
            results.append({
                "prompt": prompt,
                "prompt_type": "semantic" if is_semantic else "nonsemantic",
                "level": level_name,
                "n_features": feat_count,
                "output": output,
                "sc": sc,
                "semc": semc,
                "collapsed": collapsed,
            })

    return results


def print_summary(results):
    """打印汇总表"""

    print("\n\n" + "=" * 80)
    print("  汇总表 (Summary)")
    print("=" * 80)

    # 按干预级别汇总
    levels = ["baseline", "light", "medium", "heavy"]

    # --- 总体汇总 ---
    print(f"\n{'─' * 60}")
    print(f"  总体 (All Prompts)")
    print(f"{'─' * 60}")
    print(f"  {'Level':<20s} {'SC Rate':>10s} {'Avg SemC':>10s} {'Collapse':>10s} {'N':>5s}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*5}")

    for level in levels:
        level_results = [r for r in results if r["level"] == level]
        n = len(level_results)
        sc_rate = sum(1 for r in level_results if r["sc"]) / n if n else 0
        semc_vals = [r["semc"] for r in level_results if r["semc"] is not None]
        avg_semc = sum(semc_vals) / len(semc_vals) if semc_vals else float("nan")
        collapse_rate = sum(1 for r in level_results if r["collapsed"]) / n if n else 0

        feat_n = level_results[0]["n_features"] if level_results else 0
        label = f"{level} ({feat_n}feat)"
        semc_str = f"{avg_semc:.2f}" if semc_vals else "N/A"
        print(f"  {label:<20s} {sc_rate:>10.1%} {semc_str:>10s} {collapse_rate:>10.1%} {n:>5d}")

    # --- 按 prompt 类型分开 ---
    for ptype, ptype_label in [("semantic", "Semantic Prompts"),
                                ("nonsemantic", "Nonsemantic Prompts")]:
        print(f"\n{'─' * 60}")
        print(f"  {ptype_label}")
        print(f"{'─' * 60}")
        print(f"  {'Level':<20s} {'SC Rate':>10s} {'Avg SemC':>10s} {'Collapse':>10s} {'N':>5s}")
        print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*5}")

        for level in levels:
            level_results = [r for r in results
                             if r["level"] == level and r["prompt_type"] == ptype]
            n = len(level_results)
            if n == 0:
                continue
            sc_rate = sum(1 for r in level_results if r["sc"]) / n
            semc_vals = [r["semc"] for r in level_results if r["semc"] is not None]
            avg_semc = sum(semc_vals) / len(semc_vals) if semc_vals else float("nan")
            collapse_rate = sum(1 for r in level_results if r["collapsed"]) / n

            feat_n = level_results[0]["n_features"]
            label = f"{level} ({feat_n}feat)"
            semc_str = f"{avg_semc:.2f}" if semc_vals else "N/A"
            print(f"  {label:<20s} {sc_rate:>10.1%} {semc_str:>10s} {collapse_rate:>10.1%} {n:>5d}")

    # --- 期望结论 ---
    print(f"\n{'─' * 60}")
    print(f"  期望结论:")
    print(f"  - baseline: SC=100%, SemC 高")
    print(f"  - light/medium: SC 下降, SemC 基本保留 -> 结构/语义可分离")
    print(f"  - heavy: SC 严重崩溃, SemC 仍有残留 -> SAE 精确切开纠缠")
    print(f"  - 如果 SemC 也随 SC 同步崩溃 -> 说明切不开（与 LEACE 同）")
    print(f"{'─' * 60}")
    print()


# ===== 主流程 =====

def main():
    print("\n" + "=" * 80)
    print("  Mistral-7B SAE 特征干预实验")
    print("  Layer 16 JumpReLU SAE (16384 features)")
    print("  目标：关掉 JSON 结构特征，验证结构/语义分离")
    print("=" * 80 + "\n")

    # 环境配置
    setup_environment()

    # 加载模型和 SAE
    model = load_model()
    sae = load_sae()

    # 运行实验
    print("\n开始干预实验...\n")
    results = run_experiment(model, sae)

    # 汇总
    print_summary(results)

    # 保存原始结果到 JSON
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "output", "intervention_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 序列化友好版本
    serializable = []
    for r in results:
        serializable.append({
            "prompt": r["prompt"],
            "prompt_type": r["prompt_type"],
            "level": r["level"],
            "n_features": r["n_features"],
            "output": r["output"],
            "sc": r["sc"],
            "semc": r["semc"],
            "collapsed": r["collapsed"],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"原始结果已保存: {output_path}")


if __name__ == "__main__":
    main()
