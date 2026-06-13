"""GRPO 训练器：实现 Group Relative Policy Optimization。"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """GRPO (Group Relative Policy Optimization) 训练器。

    GRPO Loss 公式（对于每个 group G，同一道题的 group_size 条轨迹）：

      A_i = (R_i - mean(R_group)) / std(R_group)    ← Group Relative Advantage

      L_GRPO = - mean_i [ min( ratio_i * A_i,          ← PPO-style clip
                              clip(ratio_i, 1-ε, 1+ε) * A_i )
                          - β * KL(π_θ(·|s_i) || π_ref(·|s_i)) ]   ← KL 惩罚

      ratio_i = exp( log π_θ(a_i|s_i) - log π_old(a_i|s_i) )
    """

    def __init__(
        self,
        policy_model: Any = None,  # PeftModel (当前策略，带 LoRA)
        ref_model: Any = None,  # PeftModel (参考策略，冻结)
        sampler: Any = None,  # TrajectorySampler
        reward_fn: Any = None,  # RewardFunction
        optimizer: Any = None,  # torch.optim.Optimizer
        kl_coef: float = 0.04,
        clip_eps: float = 0.2,
        gamma: float = 1.0,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str | Path = "checkpoints/grpo",
    ) -> None:
        """初始化 GRPO 训练器。

        Args:
            policy_model: 当前策略模型（带 LoRA 权重）
            ref_model: 参考模型（冻结，用于 KL 约束）
            sampler: 轨迹采样器
            reward_fn: 奖励函数
            optimizer: PyTorch 优化器
            kl_coef: KL 散度系数 β
            clip_eps: PPO clip 范围 ε
            gamma: 折扣因子（数学题不需要折扣，默认 1.0）
            max_grad_norm: 梯度裁剪阈值
            checkpoint_dir: checkpoint 保存目录
        """
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.sampler = sampler
        self.reward_fn = reward_fn
        self.optimizer = optimizer
        self.kl_coef = kl_coef
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 训练状态
        self.global_step: int = 0
        self.metrics_history: List[Dict[str, float]] = []

    def train_step(
        self,
        batch_data: List[Any],  # List[Problem]
    ) -> Dict[str, float]:
        """执行一步 GRPO 训练。

        Args:
            batch_data: 一批问题数据

        Returns:
            训练指标字典
        """
        if self.sampler is None:
            return {}

        # 采样轨迹
        trajectories, group_ids, rewards = self.sampler.sample_grpo_batch(batch_data)

        # 计算损失
        loss = self.compute_loss(trajectories, group_ids, rewards)

        # 梯度更新
        if loss is not None and self.optimizer is not None:
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(...)
            # self.optimizer.step()
            # self.optimizer.zero_grad()
            pass

        self.global_step += 1

        # 记录指标
        metrics = {
            "step": self.global_step,
            "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
            "reward_std": float(np.std(rewards)) if rewards else 0.0,
            "loss": float(loss) if loss is not None else 0.0,
        }
        self.metrics_history.append(metrics)

        return metrics

    def compute_loss(
        self,
        trajectories: List[Any],
        group_ids: List[int],
        rewards: List[float],
    ) -> Optional[Any]:
        """计算 GRPO 损失。

        Args:
            trajectories: 推理轨迹列表
            group_ids: 轨迹对应的 group ID
            rewards: 轨迹奖励列表

        Returns:
            标量损失值（PyTorch Tensor）
        """
        # 计算 Group Relative Advantage
        from .advantage import AdvantageEstimator

        adv_estimator = AdvantageEstimator()
        advantages = adv_estimator.compute_group_relative(rewards, group_ids)

        # TODO: 完整的 GRPO loss 计算需要模型 logprobs
        # ratio_i = exp(log_pi_theta - log_pi_old)
        # loss = - mean( min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)
        #              - kl_coef * KL(pi_theta || pi_ref) )
        return None

    def _compute_kl(
        self,
        pi_logprobs: Any,
        ref_logprobs: Any,
    ) -> Any:
        """计算当前策略与参考策略之间的 KL 散度。

        Args:
            pi_logprobs: 当前策略的对数概率
            ref_logprobs: 参考策略的对数概率

        Returns:
            KL 散度值
        """
        # KL(pi || ref) = exp(log_ref_logprob - log_pi_logprob)
        #               - (log_ref_logprob - log_pi_logprob) + 1
        # 近似: KL = ref_logprobs - pi_logprobs
        return 0.0  # TODO: 实现 KL 计算

    def save_checkpoint(self, path: Optional[str | Path] = None) -> None:
        """保存训练 checkpoint。

        Args:
            path: 保存路径
        """
        save_path = Path(path) if path else self.checkpoint_dir / f"step_{self.global_step}"
        save_path.mkdir(parents=True, exist_ok=True)

        # TODO: 保存 LoRA 权重
        # self.policy_model.save_pretrained(save_path)

        logger.info(f"Checkpoint saved to {save_path}")

    def get_metrics(self) -> Dict[str, Any]:
        """获取当前训练指标。"""
        if not self.metrics_history:
            return {}
        return self.metrics_history[-1]
