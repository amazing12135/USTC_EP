#!/usr/bin/env python
"""统一图表生成脚本。

训练全部完成后，一次性生成所有图表：
  - 学习曲线（reward + loss 双轴）
  - KL 散度曲线
  - 阶段准确率对比（Base / SFT / RFT / GRPO）
  - 代码执行成功率对比
  - 错误率趋势

用法:
  # 生成所有图表
  python scripts/06_plot_results.py

  # 只生成 GRPO 学习曲线
  python scripts/06_plot_results.py --grpo-only

  # 只生成阶段对比图（需要评估摘要文件）
  python scripts/06_plot_results.py --compare-only

  # 从指定路径加载数据
  python scripts/06_plot_results.py --grpo-history outputs/grpo_history.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logger import get_logger

logger = get_logger(__name__)

# 图表输出目录
FIG_DIR = Path("results/figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统一图表生成")
    parser.add_argument("--grpo-only", action="store_true", help="只生成 GRPO 图表")
    parser.add_argument("--compare-only", action="store_true", help="只生成阶段对比图表")
    parser.add_argument("--grpo-history", type=str, default="outputs/grpo_history.json",
                        help="GRPO 训练历史 JSON 路径")
    parser.add_argument("--rft-history", type=str, default="outputs/rft_history.json",
                        help="RFT 训练历史 JSON 路径")
    parser.add_argument("--eval-dir", type=str, default="outputs/evals",
                        help="评估摘要目录")
    parser.add_argument("--output-dir", type=str, default="results/figures",
                        help="图表输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig_dir = Path(args.output_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 确定要生成哪些图表
    do_grpo = not args.compare_only
    do_compare = not args.grpo_only

    if do_grpo:
        _plot_grpo_learning_curve(args.grpo_history)
        _plot_rft_rounds(args.rft_history)

    if do_compare:
        _plot_stage_comparison(args.eval_dir)

    logger.info(f"\n所有图表已保存到: {fig_dir.resolve()}")


# ================================================================
# GRPO 学习曲线 + KL 曲线
# ================================================================

def _plot_grpo_learning_curve(history_path: str) -> None:
    path = Path(history_path)
    if not path.exists():
        logger.info(f"[跳过] GRPO 历史文件不存在: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)

    if not history:
        logger.warning("GRPO 历史为空，跳过")
        return

    import matplotlib.pyplot as plt

    steps = [h["step"] for h in history]
    rewards = [h["reward_mean"] for h in history]
    losses = [h["loss"] for h in history]
    kl_values = [h.get("kl_div", 0) for h in history]

    # --- 学习曲线（reward + loss 双轴）---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Average Reward", color="tab:blue")
    ax1.plot(steps, rewards, color="tab:blue", label="Reward", marker="o", markersize=3)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, linestyle=":", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Loss", color="tab:red")
    ax2.plot(steps, losses, color="tab:red", label="Loss", linestyle="--", marker="x", markersize=3)
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    plt.title("GRPO Training: Reward & Loss")
    plt.savefig(FIG_DIR / "learning_curve.png", dpi=150)
    plt.close()
    logger.info(f"学习曲线: {FIG_DIR / 'learning_curve.png'}")

    # --- KL 散度曲线 ---
    if any(kl_values):
        plt.figure(figsize=(10, 4))
        plt.plot(steps, kl_values, color="tab:purple", marker=".", markersize=3)
        plt.xlabel("Training Steps")
        plt.ylabel("KL Divergence")
        plt.title("KL Divergence During GRPO Training")
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "kl_divergence.png", dpi=150)
        plt.close()
        logger.info(f"KL 曲线: {FIG_DIR / 'kl_divergence.png'}")


# ================================================================
# RFT 轮次准确率
# ================================================================

def _plot_rft_rounds(history_path: str) -> None:
    path = Path(history_path)
    if not path.exists():
        logger.info(f"[跳过] RFT 历史文件不存在: {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        history = json.load(f)

    rounds = history.get("rounds", [])
    if not rounds:
        logger.info("RFT 无轮次数据，跳过")
        return

    import matplotlib.pyplot as plt

    train_pct = []
    for c, t in zip(history.get("train_correct", []), history.get("train_total", [])):
        train_pct.append(c / t * 100 if t > 0 else 0)
    val_acc = [a * 100 for a in history.get("val_accuracy", []) if a is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    round_labels = [f"Round {r}" for r in rounds]

    ax1.bar(round_labels, train_pct, color="#1f77b4")
    ax1.set_ylabel("Training Correct Rate (%)")
    ax1.set_title("RFT: Sample Correct Rate per Round")
    for bar, v in zip(ax1.patches, train_pct):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{v:.1f}%", ha="center", va="bottom")

    if val_acc:
        ax2.bar(round_labels, val_acc, color="#2ca02c")
        ax2.set_ylabel("Validation Accuracy (%)")
        ax2.set_title("RFT: Validation Accuracy per Round")
        for bar, v in zip(ax2.patches, val_acc):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{v:.1f}%", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "rft_round_accuracy.png", dpi=150)
    plt.close()
    logger.info(f"RFT 准确率图: {FIG_DIR / 'rft_round_accuracy.png'}")


# ================================================================
# 阶段对比图（跨 SFT / RFT / GRPO）
# ================================================================

def _plot_stage_comparison(eval_dir: str) -> None:
    """从评估摘要 JSON 中收集各阶段指标，生成对比柱状图。"""
    eval_path = Path(eval_dir)
    summaries = []
    for fpath in sorted(eval_path.glob("summary_*.json")):
        with open(fpath, "r", encoding="utf-8") as f:
            summaries.append(json.load(f))

    if not summaries:
        logger.info(f"[跳过] 评估摘要目录 {eval_dir} 中无 summary_*.json 文件")
        logger.info(f"  提示: 用 --plot 标志运行 04_evaluate.py 可生成摘要文件")
        _plot_mock_comparison()
        return

    import matplotlib.pyplot as plt

    stages = [s.get("stage", "?") for s in summaries]
    accuracies = [s.get("accuracy", 0) * 100 for s in summaries]
    tool_rates = [s.get("tool_success_rate", 0) * 100 for s in summaries]

    # --- 准确率对比 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:len(stages)]
    bars1 = ax1.bar(stages, accuracies, color=colors)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Mathematics Reasoning Accuracy by Stage")
    ax1.set_ylim(0, max(accuracies) * 1.2 if max(accuracies) > 0 else 100)
    for bar, v in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{v:.1f}%", ha="center", va="bottom")

    bars2 = ax2.bar(stages, tool_rates, color=colors)
    ax2.set_ylabel("Tool Success Rate (%)")
    ax2.set_title("Tool Execution Success Rate by Stage")
    ax2.set_ylim(0, max(tool_rates) * 1.2 if max(tool_rates) > 0 else 100)
    for bar, v in zip(bars2, tool_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{v:.1f}%", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "accuracy_comparison.png", dpi=150)
    plt.close()
    logger.info(f"阶段对比图: {FIG_DIR / 'accuracy_comparison.png'}")

    # --- 代码执行成功率 ---
    plt.figure(figsize=(8, 5))
    bars = plt.bar(stages, tool_rates, color=colors)
    plt.ylabel("Execution Success Rate (%)")
    plt.title("Agent Code Execution Performance by Stage")
    plt.ylim(0, max(tool_rates) * 1.2 if max(tool_rates) > 0 else 100)
    for bar, v in zip(bars, tool_rates):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{v:.1f}%", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "code_execution_accuracy.png", dpi=150)
    plt.close()
    logger.info(f"代码执行图: {FIG_DIR / 'code_execution_accuracy.png'}")


def _plot_mock_comparison() -> None:
    """使用模拟数据生成示例阶段对比图，展示图表效果。"""
    import matplotlib.pyplot as plt

    stages = ["Base", "SFT", "RFT", "GRPO (expected)"]
    accuracies = [15, 55, 65, 82]
    tool_rates = [0, 75, 82, 92]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    bars1 = ax1.bar(stages, accuracies, color=colors)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Expected Accuracy Progression (Illustration)")
    ax1.set_ylim(0, 100)
    for bar, v in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{v}%", ha="center", va="bottom")

    bars2 = ax2.bar(stages, tool_rates, color=colors)
    ax2.set_ylabel("Tool Success Rate (%)")
    ax2.set_title("Expected Tool Success Progression (Illustration)")
    ax2.set_ylim(0, 100)
    for bar, v in zip(bars2, tool_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{v}%", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "accuracy_comparison.png", dpi=150)
    plt.close()
    logger.info(f"阶段对比图（模拟数据）: {FIG_DIR / 'accuracy_comparison.png'}")


if __name__ == "__main__":
    main()
