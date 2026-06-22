#!/usr/bin/env python
"""Phase 3：GRPO 训练脚本。

完整流程：
1. 加载 RFT checkpoint → policy_model + ref_model
2. vLLM 推理引擎（用于 rollout 采样）
3. GRPO 训练循环：采样 → reward → 计算 loss → 更新策略
4. Wandb 监控 + 定期 checkpoint
5. 训练完自动在验证集评估

用法:
  python scripts/03_grpo_train.py                    # 正式训练
  python scripts/03_grpo_train.py --small-scale      # 小规模调参（1K, 100步）
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.parser import ConfigParser
from utils.logger import get_logger
from utils.helpers import set_seed, Timer

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO 训练")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/rft/round_2",
        help="RFT/SFT checkpoint 路径（作为 GRPO 起点）",
    )
    parser.add_argument(
        "--small-scale", action="store_true",
        help="小规模调参模式（1K 数据, 100 步）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ConfigParser()
    rl_config = config.rl
    seed = config.get("training.seed", 42)
    set_seed(seed)

    # ---- 小规模调参参数覆盖 ----
    if args.small_scale:
        max_steps = 100
        batch_size = 8
        group_size = 4
        eval_every = 25
        save_every = 50
        wandb_mode = "tuning"
        logger.info("=== Phase 3: GRPO 训练（小规模调参） ===")
    else:
        max_steps = rl_config.get("max_steps", 500)
        batch_size = rl_config.get("batch_size", 16)
        group_size = rl_config.get("group_size", 8)
        eval_every = rl_config.get("eval_every", 50)
        save_every = rl_config.get("save_every", 100)
        wandb_mode = "formal"
        logger.info("=== Phase 3: GRPO 训练 ===")

    logger.info(f"  Checkpoint:     {args.checkpoint}")
    logger.info(f"  Max steps:      {max_steps}")
    logger.info(f"  Batch size:     {batch_size}")
    logger.info(f"  Group size:     {group_size}")
    logger.info(f"  KL coef:        {rl_config.get('kl_coef', 0.04)}")
    logger.info(f"  Clip eps:       {rl_config.get('clip_eps', 0.2)}")
    logger.info(f"  Learning rate:  {rl_config.get('learning_rate', 2e-6)}")
    logger.info(f"  Mode:           {wandb_mode}")

    # ================================================================
    # 1. 初始化 Wandb
    # ================================================================
    try:
        import wandb
        wandb.init(
            project=config.get("training.wandb_project", "llm-math-agent"),
            name=f"grpo-{wandb_mode}",
            config={
                "model": config.get("model.name"),
                "checkpoint": args.checkpoint,
                "batch_size": batch_size,
                "group_size": group_size,
                "kl_coef": rl_config.get("kl_coef", 0.04),
                "clip_eps": rl_config.get("clip_eps", 0.2),
                "learning_rate": rl_config.get("learning_rate", 2e-6),
                "max_steps": max_steps,
                "mode": wandb_mode,
            },
        )
    except ImportError:
        logger.warning("Wandb 未安装，跳过监控。安装: pip install wandb")
        wandb = None

    # ================================================================
    # 2. 加载模型
    # ================================================================
    logger.info("加载模型...")
    from model.loader import ModelLoader
    from model.inference import BatchInference

    loader = ModelLoader(model_name=config.get("model.name"))
    tokenizer = loader.load_tokenizer()

    # vLLM 推理引擎（从 checkpoint 加载，用于 rollout）
    inference = BatchInference(
        model_name_or_path=args.checkpoint,
        max_model_len=config.get("data.max_seq_length", 2048),
    )

    # policy_model: 可训练的 LoRA
    policy_model = loader.load_for_training(lora_config=config.lora)
    # ref_model: 冻结的 LoRA（用于 KL 约束）
    ref_model = loader.load_ref_model(lora_config=config.lora)

    # ================================================================
    # 3. 初始化 Agent + Sampler
    # ================================================================
    from agent.react_agent import ReActAgent
    from rl.sampler import TrajectorySampler
    from rl.reward import RewardFunction
    from rl.grpo_trainer import GRPOTrainer

    agent = ReActAgent(
        model=inference,
        max_turns=config.get("agent.max_turns", 10),
    )

    sampler = TrajectorySampler(
        agent=agent,
        batch_size=batch_size,
        group_size=group_size,
        temperature=rl_config.get("temperature", 1.0),
    )

    # ================================================================
    # 4. GRPO Trainer
    # ================================================================
    trainer = GRPOTrainer(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        sampler=sampler,
        reward_fn=RewardFunction(),
        kl_coef=rl_config.get("kl_coef", 0.04),
        clip_eps=rl_config.get("clip_eps", 0.2),
        max_grad_norm=rl_config.get("max_grad_norm", 1.0),
        learning_rate=rl_config.get("learning_rate", 2e-6),
        checkpoint_dir=config.get("training.checkpoint_dir", "./checkpoints") + "/grpo",
    )

    # ================================================================
    # 5. 加载数据
    # ================================================================
    from data.dataset import DatasetManager

    dm = DatasetManager(data_config=config.data)
    train_data = dm.load_l2_data()
    train_size = min(len(train_data), config.get("data.l2_train_size", 9000))
    train_data = train_data[:train_size]
    logger.info(f"训练集: {len(train_data)} 题")

    val_data = train_data[-200:] if len(train_data) > 200 else train_data
    logger.info(f"验证集: {len(val_data)} 题（取训练集后 200 题）")

    # ================================================================
    # 6. 训练循环
    # ================================================================
    logger.info(f"\n开始 GRPO 训练 ({max_steps} steps)...")
    best_reward = -float("inf")
    best_ckpt_path = None

    with Timer("GRPO Training"):
        for step in range(max_steps):
            # 随机采样 batch
            batch = random.sample(train_data, min(batch_size, len(train_data)))

            # 一步训练
            metrics = trainer.train_step(batch)

            # 终端日志（每 10 步）
            if step % 10 == 0:
                logger.info(
                    f"  Step {step:4d}/{max_steps} | "
                    f"loss={metrics.get('loss', 0):.4f} | "
                    f"reward={metrics.get('reward_mean', 0):.3f}±{metrics.get('reward_std', 0):.3f} | "
                    f"kl={metrics.get('kl_div', 0):.4f}"
                )

            # Wandb 上报
            trainer.maybe_log_wandb(metrics)

            # ---- 定期评估 ----
            if (step + 1) % eval_every == 0:
                logger.info(f"\n--- Eval @ Step {step+1} ---")
                eval_metrics = _eval_on_subset(
                    agent, sampler, val_data[:50], trainer
                )
                logger.info(
                    f"  Val reward: {eval_metrics['reward_mean']:.3f}±{eval_metrics['reward_std']:.3f} | "
                    f"accuracy: {eval_metrics.get('accuracy', 0)*100:.1f}%"
                )
                if wandb:
                    wandb.log({"eval/" + k: v for k, v in eval_metrics.items()},
                              step=step + 1)

            # ---- 保存 checkpoint ----
            if (step + 1) % save_every == 0:
                ckpt_path = trainer.save_checkpoint()
                current_reward = metrics.get("reward_mean", 0)
                if current_reward > best_reward:
                    best_reward = current_reward
                    best_ckpt_path = trainer.save_checkpoint(
                        trainer.checkpoint_dir / "best"
                    )
                    logger.info(f"  >> Best checkpoint (reward={best_reward:.3f})")

    # ================================================================
    # 7. 最终保存
    # ================================================================
    final_ckpt = trainer.save_checkpoint()
    logger.info(f"\n{'='*50}")
    logger.info("GRPO 训练完成")
    logger.info(f"  Best reward: {best_reward:.3f}")
    logger.info(f"  Best checkpoint: {best_ckpt_path}")
    logger.info(f"  Final checkpoint: {final_ckpt}")
    logger.info(f"{'='*50}")

    # 训练历史
    log_path = Path("outputs") / "grpo_history.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(trainer.metrics_history, f, indent=2, default=str)
    logger.info(f"训练历史: {log_path}")

    # ---- 8. 自动生成图表 ----
    _generate_training_plots(trainer.metrics_history)

    if wandb:
        wandb.finish()


def _generate_training_plots(metrics_history: list) -> None:
    """训练结束后自动生成学习曲线和错误趋势图表。"""
    if not metrics_history:
        logger.warning("无训练指标，跳过图表生成")
        return

    from evaluation.visualizer import PerformanceVisualizer
    from evaluation.code_visualizer import CodePerformanceVisualizer

    steps = [m["step"] for m in metrics_history]
    rewards = [m["reward_mean"] for m in metrics_history]
    losses = [m["loss"] for m in metrics_history]

    # 学习曲线（reward + loss 双轴）
    perf_viz = PerformanceVisualizer()
    perf_viz.plot_learning_curve(steps, rewards, losses)
    logger.info(f"学习曲线已保存: {perf_viz.save_dir}/learning_curve.png")

    # 错误趋势（如果有相关指标）
    code_viz = CodePerformanceVisualizer()
    syntax_errors = [m.get("syntax_error_rate", 0) for m in metrics_history]
    logic_errors = [m.get("logic_error_rate", 0) for m in metrics_history]
    if any(syntax_errors) or any(logic_errors):
        code_viz.plot_error_rates_trend(steps, syntax_errors, logic_errors)
        logger.info(f"错误趋势图已保存: {code_viz.save_dir}/code_error_trend.png")
    else:
        logger.info("（无 syntax/logic error 指标，跳过错误趋势图）")

    # KL 散度单独曲线
    _plot_kl_curve(steps, metrics_history)

    logger.info(f"所有图表已保存到 results/figures/")


def _plot_kl_curve(steps: list, metrics_history: list) -> None:
    """绘制 KL 散度随训练步数的变化曲线。"""
    import matplotlib.pyplot as plt

    kl_values = [m.get("kl_div", 0) for m in metrics_history]
    if not any(kl_values):
        return

    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(steps, kl_values, color="tab:purple", marker=".")
    plt.xlabel("Training Steps")
    plt.ylabel("KL Divergence")
    plt.title("KL Divergence During GRPO Training")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.savefig(fig_dir / "kl_divergence.png")
    plt.close()
    logger.info(f"KL 散度曲线已保存: {fig_dir}/kl_divergence.png")


def _eval_on_subset(agent, sampler, val_problems, trainer) -> dict:
    """在验证子集上评估当前模型。"""
    import numpy as np
    from evaluation.metrics import MetricsCalculator

    trajectories = sampler.sample_batch(val_problems)
    metrics_calc = MetricsCalculator()
    result = metrics_calc.compute_all(trajectories)

    rewards = trainer.reward_fn.compute_batch(trajectories)
    return {
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "accuracy": result.accuracy,
        "avg_turns": result.avg_turns,
        "tool_success_rate": result.tool_success_rate,
    }


if __name__ == "__main__":
    main()
