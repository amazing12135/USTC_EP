"""工具集单元测试。"""

import pytest

from environment.tools.calculator import Calculator
from environment.tools.registry import ToolRegistry, Tool
from environment.tools.python_executor import PythonExecutor
from environment.tools.sympy_tool import SymPyTool


class TestToolRegistry:
    """工具注册中心测试。"""

    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = Tool(
            name="test",
            description="A test tool",
            parameters={},
            func=lambda: "ok",
        )
        registry.register(tool)
        assert registry.get("test") is tool
        assert registry.get("nonexistent") is None

    def test_list_all(self):
        registry = ToolRegistry()
        tool1 = Tool(name="t1", description="", parameters={}, func=lambda: "")
        tool2 = Tool(name="t2", description="", parameters={}, func=lambda: "")
        registry.register(tool1)
        registry.register(tool2)
        assert len(registry.list_all()) == 2

    def test_get_schema(self):
        registry = ToolRegistry()
        tool = Tool(
            name="calc",
            description="Calculate stuff",
            parameters={"expr": "string"},
            func=lambda: "",
        )
        registry.register(tool)
        schema = registry.get_schema()
        assert "calc" in schema
        assert "Calculate stuff" in schema

    def test_contains(self):
        registry = ToolRegistry()
        tool = Tool(name="test", description="", parameters={}, func=lambda: "")
        registry.register(tool)
        assert "test" in registry
        assert "other" not in registry


class TestCalculator:
    """计算器工具测试。"""

    def test_basic_arithmetic(self):
        assert Calculator.calculate("2 + 3") == "5"
        assert Calculator.calculate("10 - 7") == "3"
        assert Calculator.calculate("6 * 7") == "42"
        assert Calculator.calculate("15 / 3") == "5.0"

    def test_sqrt(self):
        result = Calculator.calculate("sqrt(144)")
        assert "12" in result

    def test_power(self):
        result = Calculator.calculate("pow(2, 10)")
        assert "1024" in result

    def test_constants(self):
        assert "3.14" in Calculator.calculate("pi")[:4]

    def test_get_tool(self):
        tool = Calculator.get_tool()
        assert tool.name == "calculator"
        assert "expression" in str(tool.parameters)


class TestPythonExecutor:
    """Python 执行器测试。"""

    def test_basic_exec(self):
        result = PythonExecutor.execute("print(1 + 1)")
        assert "2" in result

    def test_math_import(self):
        result = PythonExecutor.execute("print(math.sqrt(16))")
        assert "4.0" in result

    def test_banned_keywords(self):
        result = PythonExecutor.execute("import os; print(os.getcwd())")
        assert "Error" in result
        assert "banned" in result.lower()

    def test_get_tool(self):
        tool = PythonExecutor.get_tool()
        assert tool.name == "python_exec"
        assert tool.timeout == 5


class TestSymPyTool:
    """SymPy 工具测试。"""

    def test_basic_exec(self):
        result = SymPyTool.execute("print(solve(x**2 - 4, x))")
        assert "2" in result or "-2" in result

    def test_banned_keywords(self):
        result = SymPyTool.execute("import os; os.system('ls')")
        assert "Error" in result
        assert "banned" in result.lower()

    def test_get_tool(self):
        tool = SymPyTool.get_tool()
        assert tool.name == "sympy"
