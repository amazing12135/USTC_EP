"""轨迹采样器：批量/group 采样 Agent 推理轨迹。"""

import random
from typing import Any, Dict, List, Optional

from .reward import RewardFunction


class TrajectorySampler:
    """轨迹采样器。

    负责对数学问题进行批量或分组采样，生成 Agent 推理轨迹。

    GRPO 采样流程：
    1. 随机抽取 batch_size=16 道题
    2. 对每道题，用不同温度采样 group_size=8 条轨迹
    3. 共获得 16×8=128 条轨迹
    """

    def __init__(
        self,
        agent: Any,  # ReActAgent
        batch_size: int = 16,
        group_size: int = 8,
        temperature: float = 1.0,
        reward_fn: Optional[RewardFunction] = None,
    ) -> None:
        """初始化轨迹采样器。

        Args:
            agent: ReActAgent 实例
            batch_size: 每步采样的题目数
            group_size: 每题采样的轨迹数（GRPO 每组大小）
            temperature: 采样温度
            reward_fn: 奖励函数
        """
        self.agent = agent
        self.batch_size = batch_size
        self.group_size = group_size
        self.temperature = temperature
        self.reward_fn = reward_fn or RewardFunction()

    def sample_batch(
        self,
        problems: List[Any],
    ) -> List[Any]:
        """对一批问题进行采样（每题 1 条轨迹）。

        Args:
            problems: 问题列表

        Returns:
            AgentTrajectory 列表
        """
        trajectories = []
        for problem in problems:
            traj = self._run_single(problem)
            trajectories.append(traj)
        return trajectories

    def sample_group(
        self,
        problem: Any,
    ) -> List[Any]:
        """对单个问题采样 group_size 条轨迹。

        用于 GRPO：同一题采样多条轨迹以计算组内相对优势。

        Args:
            problem: 单道问题

        Returns:
            group_size 条轨迹列表
        """
        trajectories = []
        for _ in range(self.group_size):
            # 使用不同随机种子确保多样性
            traj = self._run_single(problem)
            trajectories.append(traj)
        return trajectories

    def sample_grpo_batch(
        self,
        problems: List[Any],
    ) -> tuple:
        """GRPO 采样：对每个题采样 group_size 条轨迹。

        Args:
            problems: 问题列表

        Returns:
            (trajectories, group_ids, rewards)
            - trajectories: 所有轨迹 (batch_size * group_size 条)
            - group_ids: 每个轨迹对应的题目编号
            - rewards: 每个轨迹的奖励
        """
        all_trajectories = []
        group_ids = []

        for i, problem in enumerate(problems):
            group_trajs = self.sample_group(problem)
            all_trajectories.extend(group_trajs)
            group_ids.extend([i] * len(group_trajs))

        rewards = self.reward_fn.compute_batch(all_trajectories)

        return all_trajectories, group_ids, rewards

    def _run_single(self, problem: Any) -> Any:
        """对单个问题运行一次完整的 Agent 推理。

        Args:
            problem: Problem 实例

        Returns:
            AgentTrajectory 推理轨迹
        """
        return self.agent.run(
            question=problem.question,
            ground_truth=problem.answer,
        )
