"""奖励函数设计：结果奖励 + 格式奖励 + 效率惩罚。"""

from dataclasses import dataclass, field
from typing import Any, Optional

from evaluation.math_judge import MathJudge


@dataclass
class RewardConfig:
    """奖励函数配置。"""

    outcome_weight: float = 1.0  # 结果奖励权重
    format_weight: float = 0.1  # 格式奖励权重
    step_penalty: float = 0.0  # 步数惩罚（0=不启用）
    penalty_per_step: float = 0.01  # 每步惩罚
    tool_json_bonus: float = 0.01  # 工具调用格式正确加分
    answer_tag_bonus: float = 0.05  # 包含 <answer> 标签加分
    parse_error_penalty: float = 0.1  # JSON 解析失败扣分
    consecutive_invalid_penalty: float = 0.1  # 连续无效输出扣分


class RewardFunction:
    """奖励函数。

    总奖励 = outcome_reward * outcome_weight
           + format_reward * format_weight
           - step_penalty * total_steps

    outcome_reward:
      - 答案正确: +1.0
      - 答案错误:  0.0

    format_reward (范围约 -0.2 ~ +0.1):
      - 每步工具调用JSON格式正确: +0.01
      - 输出包含 <answer> 标签: +0.05
      - 工具调用JSON解析失败: -0.1 / 次
      - 连续无效输出: -0.1 / 次
    """

    def __init__(self, config: Optional[RewardConfig] = None) -> None:
        """初始化奖励函数。

        Args:
            config: 奖励配置
        """
        self.config = config or RewardConfig()
        self._judge = MathJudge()

    def compute(self, trajectory: Any) -> float:
        """计算完整轨迹的奖励。

        Args:
            trajectory: AgentTrajectory 推理轨迹

        Returns:
            标量奖励值
        """
        outcome = self._outcome_reward(trajectory.is_correct)
        fmt = self._format_reward(trajectory)
        penalty = self._efficiency_penalty(trajectory.total_turns)

        total = (
            outcome * self.config.outcome_weight
            + fmt * self.config.format_weight
            - penalty
        )
        return total

    def compute_batch(self, trajectories: list) -> list[float]:
        """批量计算轨迹奖励。

        Args:
            trajectories: AgentTrajectory 列表

        Returns:
            奖励值列表
        """
        return [self.compute(t) for t in trajectories]

    def _outcome_reward(self, is_correct: bool) -> float:
        """结果奖励：正确=1.0, 错误=0.0。

        Args:
            is_correct: 答案是否正确

        Returns:
            结果奖励值
        """
        return 1.0 if is_correct else 0.0

    def _format_reward(self, trajectory: Any) -> float:
        """格式奖励：检查输出格式的合规性。

        Args:
            trajectory: AgentTrajectory

        Returns:
            格式奖励值
        """
        reward = 0.0
        steps = getattr(trajectory, "steps", [])

        has_answer_tag = False
        parse_errors = 0
        tool_json_ok = 0

        for step in steps:
            thought = getattr(step, "thought", "")

            # 检查 <answer> 标签
            if "<answer>" in thought and "</answer>" in thought:
                has_answer_tag = True

            # 检查工具调用格式
            if "<tool_call>" in thought and "</tool_call>" in thought:
                # JSON 格式基本检查（更细粒度的解析由 ActionParser 完成）
                try:
                    import json
                    import re
                    match = re.search(
                        r"<tool_call>(.*?)</tool_call>", thought, re.DOTALL
                    )
                    if match:
                        json.loads(match.group(1).strip())
                        tool_json_ok += 1
                    else:
                        parse_errors += 1
                except Exception:
                    parse_errors += 1

        if has_answer_tag:
            reward += self.config.answer_tag_bonus
        reward += tool_json_ok * self.config.tool_json_bonus
        reward -= parse_errors * self.config.parse_error_penalty

        return reward

    def _efficiency_penalty(self, total_turns: int) -> float:
        """步数效率惩罚。

        Args:
            total_turns: 总推理轮次

        Returns:
            惩罚值
        """
        if self.config.step_penalty <= 0:
            return 0.0
        return total_turns * self.config.penalty_per_step
