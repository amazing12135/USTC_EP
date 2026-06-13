"""环境抽象基类：定义 RL 环境的标准接口。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from environment.tools.registry import Tool


@dataclass
class Observation:
    """环境观察。"""

    content: str  # 观察内容
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """单步执行结果。"""

    observation: Observation  # 新观察
    reward: float = 0.0  # 即时奖励
    done: bool = False  # 是否结束
    info: Dict[str, Any] = field(default_factory=dict)  # 额外信息


class Environment(ABC):
    """RL 环境抽象基类。

    定义了 Agent 与环境交互的标准接口。
    """

    @abstractmethod
    def reset(self, problem: str) -> Observation:
        """初始化环境，返回初始观察。

        Args:
            problem: 问题描述

        Returns:
            初始观察
        """
        ...

    @abstractmethod
    def step(self, action: "Any") -> StepResult:
        """执行一步动作。

        Args:
            action: 要执行的动作（ToolCall 或其他格式）

        Returns:
            包含观察、奖励、终止标志和额外信息的 StepResult
        """
        ...

    @abstractmethod
    def get_tools(self) -> List[Tool]:
        """返回当前环境可用的工具列表。"""
        ...

    @abstractmethod
    def get_ground_truth(self) -> str:
        """返回标准答案（训练时用于奖励计算）。"""
        ...

    def render(self) -> str:
        """环境状态可视化（可选）。"""
        return ""

    def close(self) -> None:
        """清理资源（可选）。"""
        pass
