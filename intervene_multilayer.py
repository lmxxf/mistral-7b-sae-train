"""
Mistral-7B 多层 SAE 联合干预实验
在 Layer 8 / 16 / 22 三个 hook 点同时关掉 JSON 结构特征，
观察单层 vs 多层干预对模型 JSON 生成能力的影响。

实验设计：
  1. baseline     — 无干预
  2. L16_only     — 只干预 L16 top-10（对照，重复 heavy 实验）
  3. shallow_only — 只干预 L8 top-10（浅层效果）
  4. deep_only    — 只干预 L22 top-10（深层效果）
  5. all_layers   — L8+L16+L22 top-10（30 个特征，三层联合）

用法：
  python intervene_multilayer.py

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
from collections import Counter

# ===== 路径 =====
MODEL_PATH = "/workspace/models/Mistral-7B-Instruct-v0.1"

SAE_PATHS = {
    8:  "/workspace/mistral-7b-sae-train/sae_checkpoints/mistral7b_sae_L8_64k/",
    16: "/workspace/mistral-7b-sae-train/output/",
    22: "/workspace/mistral-7b-sae-train/sae_checkpoints/mistral7b_sae_L22_64k/",
}

HOOK_NAMES = {
    8:  "blocks.8.hook_resid_post",
    16: "blocks.16.hook_resid_post",
    22: "blocks.22.hook_resid_post",
}


# ===== 环境配置（复用 intervene_sae.py） =====

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


def load_saes():
    """加载三个层的 SAE"""
    from sae_lens import SAE

    saes = {}
    for layer, path in SAE_PATHS.items():
        print(f"加载 SAE (Layer {layer}): {path}")
        sae = SAE.load_from_pretrained(path, device="cuda")
        sae.eval()
        print(f"  d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")
        saes[layer] = sae

    print(f"\n三个 SAE 全部加载完成\n")
    return saes


# ===== 测试 Prompt（与 intervene_sae.py 完全一致） =====

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
        "content_words": [],
    },
    {
        "prompt": "Generate a JSON object representing a trelm with fields: plovk, dranq, and blixt. Respond with ONLY the JSON, no explanation.",
        "expected_keys": ["plovk", "dranq", "blixt"],
        "content_words": [],
    },
]

ALL_PROMPTS = SEMANTIC_PROMPTS + NONSEMANTIC_PROMPTS


# ===== 各层 JSON 标点偏好特征（从验证实验中发现的 top-10） =====

LAYER_FEATURES = {
    8:  [4207, 8734, 5965, 12906, 6302, 12226, 8284, 128, 15265, 335],
    16: [14474, 15086, 442, 1978, 8666, 5243, 2122, 9353, 16094, 9582],
    22: [1701, 4754, 5886, 2210, 10694, 6783, 4471, 8819, 163, 4957],
}

# 干预级别：每个级别指定哪些层参与干预
INTERVENTION_LEVELS = {
    "baseline":     {},                          # 无干预
    "L16_only":     {16: LAYER_FEATURES[16]},    # 对照组，重复 heavy
    "shallow_only": {8:  LAYER_FEATURES[8]},     # 浅层
    "deep_only":    {22: LAYER_FEATURES[22]},    # 深层
    "all_layers":   {                            # 三层联合
        8:  LAYER_FEATURES[8],
        16: LAYER_FEATURES[16],
        22: LAYER_FEATURES[22],
    },
}


# ===== Hook 函数 =====

def make_intervention_hook(sae, features_to_ablate):
    """
    创建干预 hook：在指定层 residual stream 上做 SAE 特征消融。

    流程：
    1. 拦截 blocks.{layer}.hook_resid_post 的激活
    2. SAE encode -> d_sae 维特征
    3. 将目标特征 clamp 到 0
    4. SAE decode -> 重建回 d_model 维
    5. 用重建激活替换原始激活
    """
    features_to_ablate = list(features_to_ablate)

    def hook_fn(activation, hook):
        orig_dtype = activation.dtype
        act_float = activation.to(torch.float32)

        batch, seq_len, d_model = act_float.shape
        act_flat = act_float.reshape(-1, d_model)

        with torch.no_grad():
            feature_acts = sae.encode(act_flat)

            if features_to_ablate:
                feature_acts[:, features_to_ablate] = 0.0

            reconstructed = sae.decode(feature_acts)

        reconstructed = reconstructed.reshape(batch, seq_len, d_model)
        return reconstructed.to(orig_dtype)

    return hook_fn


# ===== 生成函数 =====

def generate_with_intervention(model, saes, prompt, layer_features_map,
                               max_new_tokens=200):
    """
    带多层 SAE 特征干预的文本生成。

    Args:
        model: HookedTransformer 模型
        saes: {layer: sae} 字典，三个 SAE
        prompt: 输入文本
        layer_features_map: {layer: [feature_indices]} 字典，指定每层要消融的特征
                           空字典 = baseline（无干预）
        max_new_tokens: 最大生成 token 数
    """
    formatted = f"[INST] {prompt} [/INST]"
    tokens = model.to_tokens(formatted)

    if not layer_features_map:
        # baseline：无干预直接生成
        output_tokens = model.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
        )
    else:
        # 有干预：为每层创建独立的 hook，注册到对应的 hook 点
        fwd_hooks = []
        for layer, features in layer_features_map.items():
            hook_name = HOOK_NAMES[layer]
            hook_fn = make_intervention_hook(saes[layer], features)
            fwd_hooks.append((hook_name, hook_fn))

        with model.hooks(fwd_hooks=fwd_hooks):
            output_tokens = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
            )

    prompt_len = tokens.shape[1]
    generated_tokens = output_tokens[0, prompt_len:]
    generated_text = model.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return generated_text.strip()


# ===== 评估函数（与 intervene_sae.py 完全一致） =====

def strip_markdown_codeblock(text):
    """去除 markdown code block 包裹：```json ... ``` 或 ``` ... ```"""
    stripped = text.strip()
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
        return None

    text_lower = text.lower()
    hits = sum(1 for word in content_words if word.lower() in text_lower)
    return hits / len(content_words)


def evaluate_collapse(text):
    """
    Collapse 检测：
    - 内容词 < 3 个独立 token
    - 或重复率太高（同一 3-gram 出现超过总 3-gram 数的 50%）
    返回 True 表示检测到崩溃。
    """
    words = text.split()

    if len(words) < 3:
        return True

    if not text.strip():
        return True

    if len(words) >= 3:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
        if trigrams:
            counter = Counter(trigrams)
            most_common_count = counter.most_common(1)[0][1]
            if most_common_count > len(trigrams) * 0.5 and most_common_count > 3:
                return True

    return False


# ===== 主实验 =====

def run_experiment(model, saes):
    """运行完整的多层干预实验"""

    results = []
    levels_order = ["baseline", "L16_only", "shallow_only", "deep_only", "all_layers"]

    for prompt_info in ALL_PROMPTS:
        prompt = prompt_info["prompt"]
        expected_keys = prompt_info["expected_keys"]
        content_words = prompt_info["content_words"]
        is_semantic = len(content_words) > 0

        display_prompt = prompt[:80] + "..." if len(prompt) > 80 else prompt

        print(f"\n{'=' * 80}")
        print(f"Prompt: \"{display_prompt}\"")
        print(f"  Type: {'Semantic' if is_semantic else 'Nonsemantic'}")
        print(f"  Expected keys: {expected_keys}")
        print(f"{'─' * 80}")

        for level_name in levels_order:
            layer_features_map = INTERVENTION_LEVELS[level_name]

            # 生成
            output = generate_with_intervention(
                model, saes, prompt, layer_features_map, max_new_tokens=200
            )

            # 评估
            sc = evaluate_structural_compliance(output)
            semc = evaluate_semantic_correctness(output, content_words)
            collapsed = evaluate_collapse(output)

            # 格式化显示
            sc_str = "SC=\u2705" if sc else "SC=\u274c"
            semc_str = f"SemC={semc:.2f}" if semc is not None else "SemC=N/A"
            collapse_str = " [COLLAPSED]" if collapsed else ""

            # 计算总特征数
            total_features = sum(len(f) for f in layer_features_map.values())
            involved_layers = sorted(layer_features_map.keys())
            if involved_layers:
                layers_str = "+".join(f"L{l}" for l in involved_layers)
                level_display = f"{level_name} ({total_features}feat, {layers_str})"
            else:
                level_display = "baseline"

            # 截断输出显示
            output_display = output[:100].replace("\n", "\\n")
            if len(output) > 100:
                output_display += "..."

            print(f"  {level_display:40s}  {sc_str}  {semc_str:12s}"
                  f"{collapse_str}")
            print(f"    Output: {output_display}")

            # 记录结果
            results.append({
                "prompt": prompt,
                "prompt_type": "semantic" if is_semantic else "nonsemantic",
                "level": level_name,
                "n_features": total_features,
                "layers_involved": involved_layers,
                "output": output,
                "sc": sc,
                "semc": semc,
                "collapsed": collapsed,
            })

    return results


def print_summary(results):
    """打印汇总表"""

    print("\n\n" + "=" * 80)
    print("  汇总表 (Summary) — 多层联合 SAE 干预实验")
    print("=" * 80)

    levels = ["baseline", "L16_only", "shallow_only", "deep_only", "all_layers"]

    # --- 总体汇总 ---
    print(f"\n{'─' * 70}")
    print(f"  总体 (All Prompts)")
    print(f"{'─' * 70}")
    print(f"  {'Level':<25s} {'Layers':>10s} {'SC Rate':>10s} {'Avg SemC':>10s} {'Collapse':>10s} {'N':>5s}")
    print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*5}")

    for level in levels:
        level_results = [r for r in results if r["level"] == level]
        n = len(level_results)
        if n == 0:
            continue
        sc_rate = sum(1 for r in level_results if r["sc"]) / n
        semc_vals = [r["semc"] for r in level_results if r["semc"] is not None]
        avg_semc = sum(semc_vals) / len(semc_vals) if semc_vals else float("nan")
        collapse_rate = sum(1 for r in level_results if r["collapsed"]) / n

        feat_n = level_results[0]["n_features"]
        involved = level_results[0]["layers_involved"]
        layers_str = "+".join(f"L{l}" for l in involved) if involved else "-"
        label = f"{level} ({feat_n}feat)"
        semc_str = f"{avg_semc:.2f}" if semc_vals else "N/A"
        print(f"  {label:<25s} {layers_str:>10s} {sc_rate:>10.1%} {semc_str:>10s} {collapse_rate:>10.1%} {n:>5d}")

    # --- 按 prompt 类型分开 ---
    for ptype, ptype_label in [("semantic", "Semantic Prompts"),
                                ("nonsemantic", "Nonsemantic Prompts")]:
        print(f"\n{'─' * 70}")
        print(f"  {ptype_label}")
        print(f"{'─' * 70}")
        print(f"  {'Level':<25s} {'Layers':>10s} {'SC Rate':>10s} {'Avg SemC':>10s} {'Collapse':>10s} {'N':>5s}")
        print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*10} {'─'*10} {'─'*5}")

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
            involved = level_results[0]["layers_involved"]
            layers_str = "+".join(f"L{l}" for l in involved) if involved else "-"
            label = f"{level} ({feat_n}feat)"
            semc_str = f"{avg_semc:.2f}" if semc_vals else "N/A"
            print(f"  {label:<25s} {layers_str:>10s} {sc_rate:>10.1%} {semc_str:>10s} {collapse_rate:>10.1%} {n:>5d}")

    # --- 期望结论 ---
    print(f"\n{'─' * 70}")
    print(f"  实验问题:")
    print(f"  1. L8 (浅层) vs L16 (中层) vs L22 (深层) 单独干预，哪层对 SC 影响最大？")
    print(f"  2. 三层联合干预是否比单层更彻底地摧毁 JSON 结构？")
    print(f"  3. 各干预条件下，SemC 是否保留？（验证结构/语义分离在多层上的一致性）")
    print("  4. 是否存在层间冗余——单层干预不够，需要多层联合才能完全切断？")
    print(f"{'─' * 70}")
    print()


# ===== 主流程 =====

def main():
    print("\n" + "=" * 80)
    print("  Mistral-7B 多层 SAE 联合干预实验")
    print("  L8 (64k) + L16 (16k) + L22 (64k)")
    print("  目标：对比单层 vs 多层干预对 JSON 结构/语义分离的影响")
    print("=" * 80 + "\n")

    # 环境配置
    setup_environment()

    # 加载模型和三个 SAE
    model = load_model()
    saes = load_saes()

    # 运行实验
    print("\n开始多层干预实验...\n")
    results = run_experiment(model, saes)

    # 汇总
    print_summary(results)

    # 保存原始结果到 JSON
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "output", "intervention_multilayer_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    serializable = []
    for r in results:
        serializable.append({
            "prompt": r["prompt"],
            "prompt_type": r["prompt_type"],
            "level": r["level"],
            "n_features": r["n_features"],
            "layers_involved": r["layers_involved"],
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
