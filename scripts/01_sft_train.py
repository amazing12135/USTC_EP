#!/usr/bin/env python
"""Phase 1：SFT 预热训练脚本。

功能：
1. 加载 Qwen2.5-7B-Instruct + QLoRA 4-bit
2. 用 L1 SFT 数据（GSM8K 合成 + 手写模板）微调
3. 让模型学会 <tool_call> / <answer> 格式
4. 保存 LoRA adapter checkpoint

用法（AutoDL 云端）：
  python scripts/01_sft_train.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.parser import ConfigParser
from utils.logger import get_logger
from utils.helpers import set_seed, Timer

logger = get_logger(__name__)


def main() -> None:
    logger.info("=== Phase 1: SFT 训练 ===")

    # ---- 1. 配置和数据 ----
    config = ConfigParser()
    set_seed(config.get("training.seed", 42))

    # ---- 2. 加载 SFT 数据 ----
    sft_path = Path("data/sft/sft_data.jsonl")
    records = []
    with open(sft_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    logger.info(f"SFT 数据: {len(records)} 条")

    _print_data_stats(records)

    # ---- 3. 加载模型 (QLoRA) ----
    logger.info(f"加载模型: {config.get('model.name')}")
    from model.loader import ModelLoader

    loader = ModelLoader(
        model_name=config.get("model.name", "Qwen/Qwen2.5-7B-Instruct"),
    )
    tokenizer = loader.load_tokenizer()
    model = loader.load_for_training(lora_config=config.lora)

    # ---- 4. 构建 Dataset ----
    from datasets import Dataset
    dataset = Dataset.from_list(records)

    # ---- 5. 训练参数 ----
    from transformers import TrainingArguments

    output_dir = Path(config.get("training.output_dir", "./outputs")) / "sft"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        eval_strategy="no",
        dataloader_num_workers=2,
        remove_unused_columns=True,
        report_to=["none"],
        seed=config.get("training.seed", 42),
    )

    # ---- 6. SFT 训练 ----
    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        formatting_func=lambda ex: formatting_func(ex, tokenizer),
        max_seq_length=config.get("data.max_seq_length", 2048),
        packing=False,
    )

    logger.info("开始训练 (3 epochs, batch=16, lr=2e-4)...")
    with Timer("SFT Training"):
        train_result = trainer.train()

    logger.info(f"训练完成: {trainer.state.global_step} steps, "
                 f"loss={train_result.metrics.get('train_loss', 0):.4f}")

    # ---- 7. 保存 checkpoint ----
    checkpoint_dir = Path(config.get("training.checkpoint_dir", "./checkpoints")) / "sft"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(checkpoint_dir))
    tokenizer.save_pretrained(str(checkpoint_dir))
    logger.info(f"Checkpoint 已保存: {checkpoint_dir}")


# ================================================================
# 辅助函数
# ================================================================

def formatting_func(example: Dict[str, Any], tokenizer: Any) -> str:
    """将 ChatML messages 转为 Qwen2.5 训练文本。

    tokenizer.apply_chat_template 自动处理：
    - ChatML 格式拼接 (<|im_start|>...<|im_end|>)
    - assistant 部分的 label masking
    """
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


def _print_data_stats(records: List[Dict[str, Any]]) -> None:
    """打印数据集统计信息。"""
    total = len(records)
    tool_call_count = sum(
        1 for r in records
        if any("<tool_call>" in m["content"] for m in r["messages"])
    )
    answer_count = sum(
        1 for r in records
        if any("<answer>" in m["content"] for m in r["messages"])
    )
    tool_names: Dict[str, int] = {}
    for r in records:
        for m in r["messages"]:
            if "<tool_call>" in m["content"]:
                names = re.findall(r'"name":\s*"(\w+)"', m["content"])
                for n in names:
                    tool_names[n] = tool_names.get(n, 0) + 1

    logger.info(f"  tool_call: {tool_call_count}/{total} ({100*tool_call_count//total}%)")
    logger.info(f"  answer:    {answer_count}/{total} ({100*answer_count//total}%)")
    logger.info(f"  工具分布: {tool_names}")


if __name__ == "__main__":
    main()
