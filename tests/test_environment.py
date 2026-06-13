"""环境层单元测试。"""

import pytest

from environment.base import Environment
from environment.math_env import MathEnvironment, ToolCall


class TestMathEnvironment:
    """数学环境测试。"""

    def test_reset(self):
        env = MathEnvironment()
        obs = env.reset("What is 1+1?", "2")
        assert "What is 1+1?" == obs.content
        assert env.get_ground_truth() == "2"
        assert env.current_step == 0

    def test_step_calculator(self):
        env = MathEnvironment()
        env.reset("What is 12 + 34?", "46")
        action = ToolCall(name="calculator", args={"expression": "12 + 34"})
        result = env.step(action)
        assert "46" in result.observation.content
        assert result.info["success"] is True

    def test_step_unknown_tool(self):
        env = MathEnvironment()
        env.reset("test", "")
        action = ToolCall(name="nonexistent", args={})
        result = env.step(action)
        assert "not found" in result.observation.content.lower()
        assert result.info["success"] is False

    def test_max_steps(self):
        env = MathEnvironment(max_steps=3)
        env.reset("test", "")
        action = ToolCall(name="calculator", args={"expression": "1+1"})

        for _ in range(3):
            result = env.step(action)
        assert result.done is True

    def test_get_tools(self):
        env = MathEnvironment()
        tools = env.get_tools()
        tool_names = [t.name for t in tools]
        assert "calculator" in tool_names
        assert "python_exec" in tool_names
        assert "sympy" in tool_names
