"""数学问题求解环境：将数学问题包装为标准的 RL 环境。"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import Environment, Observation, StepResult
from .tools.registry import Tool, ToolRegistry
from .tools.calculator import Calculator
from .tools.python_executor import PythonExecutor
from .tools.sympy_tool import SymPyTool


@dataclass
class ToolCall:
    """工具调用请求。"""

    name: str
    args: Dict[str, Any]


class MathEnvironment(Environment):
    """数学问题求解环境。

    将数学问题包装为 RL 环境，管理工具调用和状态追踪。
    """

    def __init__(self, max_steps: int = 10, registry: Optional[ToolRegistry] = None) -> None:
        """初始化数学环境。

        Args:
            max_steps: 最大步数
            registry: 工具注册中心，如果为 None 则使用默认工具集
        """
        self.max_steps = max_steps
        self.registry = registry or self._default_registry()
        self._problem: str = ""
        self._ground_truth: str = ""
        self._current_step: int = 0

    @staticmethod
    def _default_registry() -> ToolRegistry:
        """创建包含默认工具集的注册中心。"""
        registry = ToolRegistry()
        registry.register(Calculator.get_tool())
        registry.register(PythonExecutor.get_tool())
        registry.register(SymPyTool.get_tool())
        return registry

    def reset(self, problem: str, ground_truth: str = "") -> Observation:
        """重置环境，设置新问题。

        Args:
            problem: 数学问题文本
            ground_truth: 标准答案（可选，评估/训练时使用）

        Returns:
            初始观察
        """
        self._problem = problem
        self._ground_truth = ground_truth
        self._current_step = 0

        return Observation(
            content=problem,
            metadata={
                "tools_available": self.registry.get_names(),
                "max_steps": self.max_steps,
            },
        )

    def step(self, action: ToolCall) -> StepResult:
        """执行一步工具调用。

        Args:
            action: 工具调用请求

        Returns:
            StepResult 包含工具执行结果
        """
        self._current_step += 1
        done = self._current_step >= self.max_steps

        tool = self.registry.get(action.name)
        if tool is None:
            return StepResult(
                observation=Observation(
                    content=f"Error: tool '{action.name}' not found. Available: {self.registry.get_names()}",
                    metadata={"error": "tool_not_found"},
                ),
                reward=0.0,
                done=done,
                info={"tool_name": action.name, "success": False},
            )

        try:
            output = tool.func(**action.args)
            success = not output.startswith("Error:")
            return StepResult(
                observation=Observation(
                    content=output,
                    metadata={"tool_name": action.name, "success": success},
                ),
                reward=0.0,  # 中间步骤 reward 为 0，最终奖励由 RewardFunction 统一计算
                done=done,
                info={"tool_name": action.name, "success": success, "output": output},
            )
        except Exception as e:
            return StepResult(
                observation=Observation(
                    content=f"Error: {str(e)}",
                    metadata={"tool_name": action.name, "success": False},
                ),
                reward=0.0,
                done=done,
                info={"tool_name": action.name, "success": False, "error": str(e)},
            )

    def get_tools(self) -> List[Tool]:
        """返回可用工具列表。"""
        return self.registry.list_all()

    def get_ground_truth(self) -> str:
        """返回标准答案。"""
        return self._ground_truth

    @property
    def problem(self) -> str:
        """当前问题文本。"""
        return self._problem

    @property
    def current_step(self) -> int:
        """当前步数。"""
        return self._current_step
