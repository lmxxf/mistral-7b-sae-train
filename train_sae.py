"""
Mistral-7B SAE 训练脚本
目标：对 Mistral-7B-Instruct-v0.3 的残差流训练稀疏自编码器
用途：赵磊的 SC/SemC 双分离实验，替代 LEACE（线性刀切不开纠缠特征）

用法：
  # 训单层（默认 layer 16）
  python train_sae.py

  # 指定层
  python train_sae.py --layer 10

  # 批量训 L10-L25
  python train_sae.py --layer-range 10 25

硬件：DGX Spark GB10, 128GB 统一内存
"""

import argparse
import json
import os

# 强制离线模式，容器里没网
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoConfig
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    LoggingConfig,
)
from sae_lens.saes.jumprelu_sae import JumpReLUTrainingSAEConfig


# ===== 路径 =====
MODEL_PATH = "/workspace/models/Mistral-7B-Instruct-v0.1"
DATASET_PATH = "/workspace/datasets/openwebtext"
OUTPUT_DIR = "/workspace/mistral-7b-sae-train/sae_checkpoints"

# ===== 超参 =====
D_IN = 4096            # Mistral-7B hidden dim
D_SAE = 16384          # 字典大小 16K (4x expansion, 省内存且特征更容易检查)
CONTEXT_SIZE = 256     # token 窗口
BATCH_SIZE = 4096      # tokens per batch
TOTAL_STEPS = 12_000   # 训练步数（快速验证版）
TOTAL_TOKENS = TOTAL_STEPS * BATCH_SIZE  # ~50M tokens
LR = 5e-5
LR_WARMUP = 1000
LR_DECAY = TOTAL_STEPS // 5  # 最后 20% 衰减


def train_single_layer(layer: int):
    """训练单层 SAE"""
    print(f"\n{'='*60}")
    print(f"  训练 SAE: Mistral-7B Layer {layer}")
    print(f"  字典大小: {D_SAE}, 训练 tokens: {TOTAL_TOKENS:,}")
    print(f"{'='*60}\n")

    hook_name = f"blocks.{layer}.hook_resid_post"
    run_name = f"mistral7b_sae_L{layer}_64k"

    # 从本地路径加载 HF 模型，再传给 TransformerLens（绕过容器网络问题）
    print(f"加载模型: {MODEL_PATH}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float16
    )

    # 修补 transformers 5.x 兼容性：新版删了 rope_theta，TransformerLens 还在用
    import transformers
    _orig_from_pretrained = transformers.AutoConfig.from_pretrained
    @staticmethod
    def _patched_from_pretrained(*args, **kwargs):
        config = _orig_from_pretrained(*args, **kwargs)
        if not hasattr(config, "rope_theta"):
            # 从原始 JSON 里拿，默认 10000.0
            import json
            config_path = os.path.join(MODEL_PATH, "config.json")
            with open(config_path) as f:
                raw = json.load(f)
            config.rope_theta = raw.get("rope_theta", 10000.0)
        return config
    transformers.AutoConfig.from_pretrained = _patched_from_pretrained

    # 把本地模型路径注入 HF cache，让 TransformerLens 的 AutoConfig 能离线找到
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    link_name = os.path.join(cache_dir, "models--mistralai--Mistral-7B-Instruct-v0.1")
    if not os.path.exists(link_name):
        os.makedirs(cache_dir, exist_ok=True)
        # 创建目录结构模拟 HF cache
        os.makedirs(link_name, exist_ok=True)
        snapshot_dir = os.path.join(link_name, "snapshots", "local")
        os.makedirs(snapshot_dir, exist_ok=True)
        # 软链所有文件
        for f in os.listdir(MODEL_PATH):
            src = os.path.join(MODEL_PATH, f)
            dst = os.path.join(snapshot_dir, f)
            if not os.path.exists(dst) and os.path.isfile(src):
                os.symlink(src, dst)
        # refs/main 指向 snapshot
        refs_dir = os.path.join(link_name, "refs")
        os.makedirs(refs_dir, exist_ok=True)
        with open(os.path.join(refs_dir, "main"), "w") as fp:
            fp.write("local")
        print(f"已创建 HF cache 软链: {link_name}")

    cfg = LanguageModelSAERunnerConfig(
        # --- 模型 ---
        model_name="mistralai/Mistral-7B-Instruct-v0.1",
        model_class_name="HookedTransformer",
        hook_name=hook_name,
        model_from_pretrained_kwargs={
            "center_writing_weights": False,
            "hf_model": hf_model,
        },

        # --- 数据 ---
        dataset_path=DATASET_PATH,
        streaming=False,
        is_dataset_tokenized=False,
        context_size=CONTEXT_SIZE,

        # --- SAE 架构 (JumpReLU, Anthropic 同款) ---
        sae=JumpReLUTrainingSAEConfig(
            d_in=D_IN,
            d_sae=D_SAE,
            normalize_activations="expected_average_only_in",
            jumprelu_sparsity_loss_mode="tanh",
            l0_coefficient=5.0,
            l0_warm_up_steps=5000,
            pre_act_loss_coefficient=3e-6,
            jumprelu_init_threshold=0.01,
            jumprelu_bandwidth=0.05,
        ),

        # --- 训练 ---
        lr=LR,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr_scheduler_name="constant",
        lr_warm_up_steps=LR_WARMUP,
        lr_decay_steps=LR_DECAY,
        train_batch_size_tokens=BATCH_SIZE,
        training_tokens=TOTAL_TOKENS,

        # --- 数据加载 ---
        n_batches_in_buffer=16,    # 64→16, SAELens要求buffer≥batch，8太小会报错
        store_batch_size_prompts=4, # 16→4, 每次喂给模型的prompt数减少

        # --- 设备 ---
        device="cuda",
        dtype="float32",
        autocast_lm=True,  # 模型用 float16 省显存，SAE 用 float32 保精度

        # --- 输出 ---
        seed=42,
        checkpoint_path=os.path.join(OUTPUT_DIR, run_name),
        logger=LoggingConfig(
            log_to_wandb=False,  # Spark 上不用 wandb
        ),
    )

    runner = LanguageModelSAETrainingRunner(cfg)
    try:
        sae = runner.run()
    except TypeError as e:
        if "not JSON serializable" in str(e):
            # SAELens bug: 保存 cfg.json 时试图序列化 hf_model 对象
            # 训练已完成，只是保存 cfg 时崩了，手动补救
            print(f"\n⚠️  SAELens 保存 cfg.json 时报错（已知 bug），手动保存...")
            sae = runner.sae

            # 手动保存权重和 cfg
            save_dir = os.path.join(OUTPUT_DIR, run_name)
            os.makedirs(save_dir, exist_ok=True)

            # 保存权重（转换为推理态：log_threshold → threshold）
            from safetensors.torch import save_file
            state_dict = sae.state_dict()
            if "log_threshold" in state_dict and "threshold" not in state_dict:
                state_dict["threshold"] = torch.exp(state_dict.pop("log_threshold"))
            save_file(state_dict, os.path.join(save_dir, "sae_weights.safetensors"))

            # 手动写干净的 cfg.json（不含 hf_model）
            clean_cfg = {
                "device": "cuda",
                "d_sae": D_SAE,
                "apply_b_dec_to_input": True,
                "reshape_activations": "none",
                "d_in": D_IN,
                "dtype": "float32",
                "metadata": {
                    "sae_lens_version": "6.39.0",
                    "sae_lens_training_version": "6.39.0",
                    "dataset_path": DATASET_PATH,
                    "hook_name": hook_name,
                    "model_name": "mistralai/Mistral-7B-Instruct-v0.1",
                    "model_class_name": "HookedTransformer",
                    "hook_head_index": None,
                    "context_size": CONTEXT_SIZE,
                    "seqpos_slice": [None],
                    "model_from_pretrained_kwargs": {
                        "center_writing_weights": False,
                    },
                },
                "architecture": "jumprelu",
                "model_name": "mistralai/Mistral-7B-Instruct-v0.1",
                "hook_name": hook_name,
                "hook_layer": layer,
                "hook_head_index": None,
                "context_size": CONTEXT_SIZE,
                "prepend_bos": True,
                "normalize_activations": "expected_average_only_in",
            }
            with open(os.path.join(save_dir, "cfg.json"), "w") as f:
                json.dump(clean_cfg, f, indent=2)

            print(f"✅ 手动保存完成: {save_dir}")
        else:
            raise
    print(f"\n✅ Layer {layer} SAE 训练完成，保存在 {OUTPUT_DIR}/{run_name}")
    return sae


def main():
    parser = argparse.ArgumentParser(description="Train SAE for Mistral-7B")
    parser.add_argument("--layer", type=int, default=16, help="单层训练，默认 16")
    parser.add_argument("--layer-range", type=int, nargs=2, metavar=("START", "END"),
                        help="批量训练，如 --layer-range 10 25")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.layer_range:
        start, end = args.layer_range
        for layer in range(start, end + 1):
            train_single_layer(layer)
    else:
        train_single_layer(args.layer)


if __name__ == "__main__":
    main()
