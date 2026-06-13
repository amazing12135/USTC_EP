"""基础计算器工具：安全的数值表达式求值。"""

import math
import re
from typing import Any, Dict

from .registry import Tool


class Calculator:
    """安全的数值表达式计算器。

    支持的运算: +, -, *, /, **, %, sqrt, abs, round, sin, cos, log, exp
    """

    TOOL_NAME = "calculator"
    TOOL_DESCRIPTION = "计算数学表达式，返回数值结果"

    # 允许的表达式字符白名单
    _ALLOWED_PATTERN = re.compile(r"^[\d\s+\-*/().,%^a-zA-Z_]+$")

    # 预注入的安全函数
    _SAFE_FUNCTIONS: Dict[str, Any] = {
        "sqrt": math.sqrt,
        "abs": abs,
        "round": round,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
        "pow": pow,
    }

    @classmethod
    def get_tool(cls) -> Tool:
        """创建 Calculator 工具实例。"""
        return Tool(
            name=cls.TOOL_NAME,
            description=cls.TOOL_DESCRIPTION,
            parameters={"expression": {"type": "string", "description": "要计算的数学表达式"}},
            func=cls.calculate,
        )

    @staticmethod
    def calculate(expression: str) -> str:
        """计算数学表达式并返回结果。

        Args:
            expression: 数学表达式字符串

        Returns:
            计算结果字符串
        """
        # 简单表达式验证
        expression = expression.strip()
        if not Calculator._ALLOWED_PATTERN.match(expression):
            return f"Error: invalid expression '{expression}'"

        try:
            # 使用受限的 eval 环境
            result = eval(expression, {"__builtins__": {}}, Calculator._SAFE_FUNCTIONS)
            if isinstance(result, float):
                # 四舍五入到合理精度
                result = round(result, 10)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"
