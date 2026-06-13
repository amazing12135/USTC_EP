"""优势估计模块：GRPO 的 Group Relative Advantage 计算。"""

from typing import Any, Dict, List

import numpy as np


class AdvantageEstimator:
    """优势估计器。

    GRPO 采用 Group Relative Advantage：
    同一道题的多条轨迹构成一个 group，
    advantage = (reward - mean(group_rewards)) / std(group_rewards)
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """初始化优势估计器。

        Args:
            eps: 防止除零的小量
        """
        self.eps = eps

    def compute_group_relative(
        self,
        rewards: List[float],
        group_ids: List[int],
    ) -> List[float]:
        """计算 Group Relative Advantage。

        Args:
            rewards: 每条轨迹的奖励值
            group_ids: 每条轨迹所属的 group ID（同一题 = 同一 group_id）

        Returns:
            每条轨迹的 advantage 值
        """
        if len(rewards) != len(group_ids):
            raise ValueError(
                f"rewards and group_ids must have same length: {len(rewards)} vs {len(group_ids)}"
            )

        advantages = [0.0] * len(rewards)

        # 按组聚合
        groups: Dict[int, List[int]] = {}
        for i, gid in enumerate(group_ids):
            groups.setdefault(gid, []).append(i)

        for gid, indices in groups.items():
            group_rewards = [rewards[i] for i in indices]
            mean_r = np.mean(group_rewards)
            std_r = np.std(group_rewards) + self.eps

            for i in indices:
                advantages[i] = (rewards[i] - mean_r) / std_r

        return advantages

    def compute_stats(
        self,
        rewards: List[float],
        advantages: List[float],
    ) -> Dict[str, float]:
        """计算优势统计量。

        Args:
            rewards: 奖励列表
            advantages: 优势值列表

        Returns:
            统计量字典
        """
        return {
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "advantage_mean": float(np.mean(advantages)),
            "advantage_std": float(np.std(advantages)),
            "advantage_max": float(np.max(np.abs(advantages))),
        }
