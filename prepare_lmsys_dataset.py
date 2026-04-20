"""
预处理 lmsys-chat-1m-chat-formatted 数据集
将多轮对话用 Mistral v0.3 的 chat template 渲染成纯文本，存成 Arrow 格式
SAELens 直接读 text 字段，和原来 OpenWebText 一样

用法（在容器 d2l_exp 里跑）：
  python prepare_lmsys_dataset.py
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

MODEL_PATH = "/workspace/models/Mistral-7B-Instruct-v0.3"
INPUT_DIR = "/workspace/datasets/lmsys-chat-1m-chat-formatted"
OUTPUT_DIR = "/workspace/datasets/lmsys_chat_mistral"

def main():
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("加载 lmsys 数据集...")
    ds = load_dataset(
        "parquet",
        data_files=f"{INPUT_DIR}/data/train-*.parquet",
        split="train",
    )
    print(f"原始数据: {len(ds)} 条对话")

    def render_chat(example):
        """用 Mistral v0.3 chat template 渲染对话"""
        conv = example["conversation"]
        try:
            text = tokenizer.apply_chat_template(conv, tokenize=False)
        except Exception:
            text = ""
        return {"text": text}

    print("渲染 chat template...")
    ds_rendered = ds.map(render_chat, remove_columns=ds.column_names, num_proc=8)

    # 过滤空文本
    before = len(ds_rendered)
    ds_rendered = ds_rendered.filter(lambda x: len(x["text"]) > 50)
    after = len(ds_rendered)
    print(f"过滤: {before} -> {after} ({before - after} 条丢弃)")

    print(f"保存到 {OUTPUT_DIR}...")
    data_dir = os.path.join(OUTPUT_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    # 存 parquet，目录结构让 load_dataset 自动识别
    ds_rendered.to_parquet(os.path.join(data_dir, "train-00000-of-00001.parquet"))

    # 看几条样本
    print("\n=== 样本 ===")
    for i in range(3):
        text = ds_rendered[i]["text"]
        print(f"\n[{i}] ({len(text)} chars): {text[:200]}...")

    print(f"\n✅ 完成: {len(ds_rendered)} 条, 保存在 {data_dir}/")


if __name__ == "__main__":
    main()
