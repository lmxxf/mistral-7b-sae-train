"""
Qwen3-8B SAE 训练脚本
和 Mistral/Llama 配置一致，用于三模型跨模型复现

层选择基于 DAS 层扫描：
  L9 (25%) - 浅层
  L22 (61%) - DAS peak，最重要
  L27 (75%) - 深层，DAS 次峰

用法：
  python train_sae_qwen3.py --layer 9
  python train_sae_qwen3.py --layer 22
  python train_sae_qwen3.py --layer 27

硬件：DGX Spark GB10, 128GB 统一内存
"""

import argparse
import json
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoModelForCausalLM
from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    LoggingConfig,
)
from sae_lens.saes.jumprelu_sae import JumpReLUTrainingSAEConfig


# ===== 路径 =====
MODEL_PATH = "/workspace/models/Qwen3-8B/Qwen/Qwen3-8B"
DATASET_PATH = "/workspace/datasets/lmsys_chat_qwen3"
OUTPUT_DIR = "/workspace/mistral-7b-sae-train/sae_checkpoints"
TRANSFORMERLENS_MODEL_NAME = "Qwen/Qwen3-8B"

# ===== 超参（和 Mistral/Llama 一致）=====
D_IN = 4096
D_SAE = 16384
CONTEXT_SIZE = 256
BATCH_SIZE = 4096
TOTAL_STEPS = 12_000
TOTAL_TOKENS = TOTAL_STEPS * BATCH_SIZE
LR = 5e-5
LR_WARMUP = 1000
LR_DECAY = TOTAL_STEPS // 5


def train_single_layer(layer: int):
    print(f"\n{'='*60}")
    print(f"  训练 SAE: Qwen3-8B Layer {layer}")
    print(f"  字典大小: {D_SAE}, 训练 tokens: {TOTAL_TOKENS:,}")
    print(f"{'='*60}\n")

    hook_name = f"blocks.{layer}.hook_resid_post"
    run_name = f"qwen3_8b_sae_L{layer}_16k"

    print(f"加载模型: {MODEL_PATH}")
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True
    )

    # 修补 transformers 5.x：Qwen3Config 没有 rope_theta，TransformerLens 需要
    import transformers
    _orig_from_pretrained = transformers.AutoConfig.from_pretrained
    @staticmethod
    def _patched_from_pretrained(*args, **kwargs):
        config = _orig_from_pretrained(*args, **kwargs)
        if not hasattr(config, "rope_theta"):
            import json as _json
            config_path = os.path.join(MODEL_PATH, "config.json")
            with open(config_path) as f:
                raw = _json.load(f)
            config.rope_theta = raw.get("rope_theta", 1000000)
        return config
    transformers.AutoConfig.from_pretrained = _patched_from_pretrained

    # 把本地模型路径注入 HF cache
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    link_name = os.path.join(cache_dir, "models--Qwen--Qwen3-8B")
    if os.path.exists(link_name):
        import shutil
        shutil.rmtree(link_name)
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
    print(f"已创建 HF cache 软链: {link_name} -> {MODEL_PATH}")

    cfg = LanguageModelSAERunnerConfig(
        # --- 模型 ---
        model_name=TRANSFORMERLENS_MODEL_NAME,
        model_class_name="HookedTransformer",
        hook_name=hook_name,
        model_from_pretrained_kwargs={
            "center_writing_weights": False,
            "hf_model": hf_model,
            "trust_remote_code": True,
        },

        # --- 数据 ---
        dataset_path=DATASET_PATH,
        streaming=False,
        is_dataset_tokenized=False,
        context_size=CONTEXT_SIZE,

        # --- SAE 架构 (JumpReLU, 和 Mistral/Llama 一致) ---
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
        n_batches_in_buffer=16,
        store_batch_size_prompts=4,

        # --- 设备 ---
        device="cuda",
        dtype="float32",
        autocast_lm=True,

        # --- 输出 ---
        seed=42,
        checkpoint_path=os.path.join(OUTPUT_DIR, run_name),
        logger=LoggingConfig(
            log_to_wandb=False,
        ),
    )

    runner = LanguageModelSAETrainingRunner(cfg)
    try:
        sae = runner.run()
    except TypeError as e:
        if "not JSON serializable" in str(e):
            print(f"\n⚠️  SAELens 保存 cfg.json 时报错（已知 bug），手动保存...")
            sae = runner.sae

            save_dir = os.path.join(OUTPUT_DIR, run_name)
            os.makedirs(save_dir, exist_ok=True)

            from safetensors.torch import save_file
            state_dict = sae.state_dict()
            if "log_threshold" in state_dict and "threshold" not in state_dict:
                state_dict["threshold"] = torch.exp(state_dict.pop("log_threshold"))
            save_file(state_dict, os.path.join(save_dir, "sae_weights.safetensors"))

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
                    "model_name": "Qwen/Qwen3-8B",
                    "model_class_name": "HookedTransformer",
                    "hook_head_index": None,
                    "context_size": CONTEXT_SIZE,
                    "seqpos_slice": [None],
                    "model_from_pretrained_kwargs": {
                        "center_writing_weights": False,
                    },
                },
                "architecture": "jumprelu",
                "model_name": "Qwen/Qwen3-8B",
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
    parser = argparse.ArgumentParser(description="Train SAE for Qwen3-8B")
    parser.add_argument("--layer", type=int, default=22, help="单层训练，默认 22（DAS peak）")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_single_layer(args.layer)


if __name__ == "__main__":
    main()
