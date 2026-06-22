#!/usr/bin/env python
"""Phase 2：RFT Baseline 训练脚本。

流程：
1. 加载 SFT checkpoint (Phase 1 产物)
2. 用当前模型对 L2 (MATH) 训练集采样
3. 筛选正确轨迹
4. 用正确轨迹做 SFT 微调
5. 评估并记录轨迹到 MD 记忆库
6. 可选：多轮迭代

用法（AutoDL 云端）：
  python scripts/02_rft_train.py

用法（本地测试采样逻辑）：
  python scripts/02_rft_train.py --mock-run
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.parser import ConfigParser
from utils.logger import get_logger
from utils.helpers import set_seed, Timer

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RFT Baseline 训练")
    parser.add_argument(
        "--sft-checkpoint", type=str, default="checkpoints/sft",
        help="Phase 1 SFT checkpoint 路径",
    )
    parser.add_argument(
        "--mock-run", action="store_true",
        help="Mock 模式：只验证数据流和逻辑，不实际训练",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ConfigParser()
    set_seed(config.get("training.seed", 42))
    rl_config = config.rl

    logger.info("=== Phase 2: RFT Baseline 训练 ===")
    logger.info(f"  SFT checkpoint: {args.sft_checkpoint}")
    logger.info(f"  RFT rounds: {rl_config.get('rft_rounds', 2)}")
    logger.info(f"  Temperature: {rl_config.get('rft_temperature', 0.8)}")
    logger.info(f"  Epochs/round: {rl_config.get('rft_epochs', 2)}")

    # ---- 1. 加载数据 ----
    from data.dataset import DatasetManager

    dm = DatasetManager(data_config=config.data)
    train_data = dm.load_l2_data()
    val_data = dm.load_l2_data()
    val_size = config.get("data.l2_val_size", 1000)
    train_size = min(len(train_data), config.get("data.l2_train_size", 9000))
    train_data = train_data[:train_size]
    val_data = val_data[-val_size:]

    logger.info(f"  训练集: {len(train_data)} 题")
    logger.info(f"  验证集: {len(val_data)} 题")

    if args.mock_run:
        logger.info("--- Mock Run: 验证数据流 ---")
        verify_data(train_data[:10])
        verify_data(val_data[:5])
        logger.info("  数据流验证通过")
        return

    # ---- 2. 加载模型 ----
    from model.loader import ModelLoader
    from model.inference import BatchInference
    from agent.react_agent import ReActAgent

    loader = ModelLoader(model_name=config.get("model.name"))
    tokenizer = loader.load_tokenizer()

    inference = BatchInference(
        model_name_or_path=args.sft_checkpoint,
        max_model_len=config.get("data.max_seq_length", 2048),
    )
    policy_model = loader.load_for_training(lora_config=config.lora)
    agent = ReActAgent(model=inference, max_turns=config.get("agent.max_turns", 10))

    # ---- 3. RFT 训练 ----
    from rl.rft_trainer import RFTTrainer

    trainer = RFTTrainer(
        agent=agent,
        model=policy_model,
        tokenizer=tokenizer,
        temperature=rl_config.get("rft_temperature", 0.8),
        filter_ratio=rl_config.get("rft_filter_ratio", 0.5),
        epochs=rl_config.get("rft_epochs", 2),
        rounds=rl_config.get("rft_rounds", 2),
        output_dir=config.get("training.checkpoint_dir", "./checkpoints") + "/rft",
        memory_dir="data/memory_store",
    )

    with Timer("RFT Training"):
        history = trainer.iterate(
            dataset=train_data,
            n_rounds=rl_config.get("rft_rounds", 2),
            val_dataset=val_data,
        )

    # ---- 4. 输出结果 ----
    logger.info(f"\n{'='*50}")
    logger.info("RFT 训练完成")
    logger.info(f"{'='*50}")
    for i in range(len(history["rounds"])):
        r = history["rounds"][i]
        c = history["train_correct"][i]
        t = history["train_total"][i]
        a = history["val_accuracy"][i]
        logger.info(f"  Round {r}: train {c}/{t} ({c/t*100:.1f}%)"
                     f", val acc={a*100:.1f}%" if a else "")
        logger.info(f"    checkpoint: {history['checkpoint_dirs'][i]}")

    # 保存历史记录
    log_path = Path("outputs") / "rft_history.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"训练历史: {log_path}")

    # ---- 5. 自动生成图表 ----
    _generate_rft_plots(history)


def _generate_rft_plots(history: dict) -> None:
    """RFT 训练结束后自动生成轮次准确率图表。"""
    rounds = history.get("rounds", [])
    if not rounds:
        return

    import matplotlib.pyplot as plt

    train_pct = []
    for c, t in zip(history.get("train_correct", []), history.get("train_total", [])):
        train_pct.append(c / t * 100 if t > 0 else 0)

    val_acc = [a * 100 for a in history.get("val_accuracy", []) if a is not None]

    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 子图：训练正确率 + 验证准确率
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    round_labels = [f"Round {r}" for r in rounds]

    ax1.bar(round_labels, train_pct, color="#1f77b4")
    ax1.set_ylabel("Training Correct Rate (%)")
    ax1.set_title("RFT Training: Sample Correct Rate per Round")
    for bar, v in zip(ax1.patches, train_pct):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{v:.1f}%", ha="center", va="bottom")

    if val_acc:
        ax2.bar(round_labels, val_acc, color="#2ca02c")
        ax2.set_ylabel("Validation Accuracy (%)")
        ax2.set_title("RFT Validation Accuracy per Round")
        for bar, v in zip(ax2.patches, val_acc):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{v:.1f}%", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(fig_dir / "rft_round_accuracy.png")
    plt.close()
    logger.info(f"RFT 准确率图已保存: {fig_dir}/rft_round_accuracy.png")


def verify_data(problems):
    """验证数据流格式。"""
    for i, p in enumerate(problems):
        logger.info(f"  [{i}] category={p.category}, "
                     f"question={p.question[:60]}..., "
                     f"answer={p.answer[:30]}...")


if __name__ == "__main__":
    main()
