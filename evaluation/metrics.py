"""评估指标计算：正确率、工具调用统计、RL 训练监控指标。"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MetricsResult:
    """评估指标汇总。"""

    accuracy: float = 0.0
    avg_turns: float = 0.0
    tool_call_rate: float = 0.0
    tool_success_rate: float = 0.0
    format_compliance: float = 0.0
    parse_error_rate: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)


class MetricsCalculator:
    """评估指标计算器。

    主要指标：
    - accuracy: 答案正确率（最重要）
    - avg_turns: 平均推理轮次
    - tool_call_rate: 工具调用使用率
    - tool_success_rate: 工具调用成功率
    - format_compliance: 输出格式合规率
    - parse_error_rate: 解析错误率

    RL 训练监控指标：
    - reward_mean / reward_std: 奖励均值和标准差
    - kl_divergence: KL散度
    - policy_ratio: 策略比率
    - advantage_mean: 优势均值
    """

    def compute_all(
        self,
        trajectories: List[Any],
        rewards: Optional[List[float]] = None,
    ) -> MetricsResult:
        """计算全部评估指标。

        Args:
            trajectories: Agent 推理轨迹列表
            rewards: 每条轨迹对应的奖励值（RL训练时使用）

        Returns:
            MetricsResult 所有指标汇总
        """
        if not trajectories:
            return MetricsResult()

        n = len(trajectories)

        # accuracy
        correct_count = sum(1 for t in trajectories if getattr(t, "is_correct", False))
        accuracy = correct_count / n if n > 0 else 0.0

        # avg_turns
        total_turns = sum(getattr(t, "total_turns", 0) for t in trajectories)
        avg_turns = total_turns / n if n > 0 else 0.0

        # tool_call_rate
        has_tool = sum(
            1 for t in trajectories
            if any(s.action is not None for s in getattr(t, "steps", []))
        )
        tool_call_rate = has_tool / n if n > 0 else 0.0

        # tool_success_rate: 成功工具调用数 / 总工具调用数
        total_tool_calls = 0
        successful_tool_calls = 0
        for t in trajectories:
            for step in getattr(t, "steps", []):
                if step.action is not None and step.observation is not None:
                    total_tool_calls += 1
                    if not step.observation.startswith("Error:"):
                        successful_tool_calls += 1
        tool_success_rate = (
            successful_tool_calls / total_tool_calls
            if total_tool_calls > 0 else 1.0
        )

        # format_compliance
        format_ok = sum(
            1 for t in trajectories
            if getattr(t, "final_answer", "")
        )
        format_compliance = format_ok / n if n > 0 else 0.0

        # parse_error_rate (approximate)
        parse_error_rate = 0.0

        extra: Dict[str, float] = {}
        if rewards is not None and rewards:
            extra["reward_mean"] = sum(rewards) / len(rewards)
            extra["reward_std"] = (
                (sum((r - extra["reward_mean"]) ** 2 for r in rewards) / len(rewards)) ** 0.5
            )
            extra["reward_min"] = min(rewards)
            extra["reward_max"] = max(rewards)

        return MetricsResult(
            accuracy=accuracy,
            avg_turns=avg_turns,
            tool_call_rate=tool_call_rate,
            tool_success_rate=tool_success_rate,
            format_compliance=format_compliance,
            parse_error_rate=parse_error_rate,
            extra=extra,
        )

    def compute_rl_metrics(
        self,
        rewards: List[float],
        kl_divergence: Optional[float] = None,
        policy_ratio: Optional[float] = None,
        advantage_mean: Optional[float] = None,
    ) -> Dict[str, float]:
        """计算 RL 训练监控指标。

        Args:
            rewards: 奖励列表
            kl_divergence: KL 散度
            policy_ratio: 策略比率
            advantage_mean: 优势均值

        Returns:
            指标字典
        """
        metrics: Dict[str, float] = {}

        if rewards:
            metrics["reward_mean"] = sum(rewards) / len(rewards)
            mean = metrics["reward_mean"]
            metrics["reward_std"] = (
                sum((r - mean) ** 2 for r in rewards) / len(rewards)
            ) ** 0.5 if len(rewards) > 1 else 0.0
            metrics["reward_min"] = min(rewards)
            metrics["reward_max"] = max(rewards)

        if kl_divergence is not None:
            metrics["kl_divergence"] = kl_divergence
        if policy_ratio is not None:
            metrics["policy_ratio"] = policy_ratio
        if advantage_mean is not None:
            metrics["advantage_mean"] = advantage_mean

        return metrics
