"""SymPy 符号计算工具：代数化简、方程求解、微积分、矩阵运算。"""

import subprocess
import sys
from typing import Optional

from .registry import Tool

# SymPy 预导入
_SYMPY_PRE_IMPORTS = """
import sympy
from sympy import symbols, solve, solveset, simplify, expand, factor, diff, integrate, limit, series
from sympy import Matrix, Rational, Symbol, oo, pi, E, I, sin, cos, tan, log, exp, sqrt
from sympy import Eq, Ne, Lt, Gt, Le, Ge
from sympy import latex, pprint
from sympy.abc import x, y, z, a, b, c, n, m, t, k, u, v, w
"""

_BANNED_KEYWORDS = ["os.", "subprocess", "__import__", "eval(", "exec(", "open(", "file("]


class SymPyTool:
    """SymPy 符号计算工具。

    功能：
    - 代数化简 (simplify, expand, factor)
    - 方程求解 (solve, solveset)
    - 微积分 (diff, integrate, limit)
    - 矩阵运算
    """

    TOOL_NAME = "sympy"
    TOOL_DESCRIPTION = "执行 SymPy 符号计算，支持代数化简、方程求解、微积分、矩阵运算等"
    TIMEOUT = 5
    MAX_OUTPUT_LENGTH = 2000

    @classmethod
    def get_tool(cls) -> Tool:
        """创建 SymPyTool 工具实例。"""
        return Tool(
            name=cls.TOOL_NAME,
            description=cls.TOOL_DESCRIPTION,
            parameters={"code": {"type": "string", "description": "要执行的 SymPy 代码"}},
            func=cls.execute,
            timeout=cls.TIMEOUT,
        )

    @staticmethod
    def execute(code: str) -> str:
        """在沙箱中执行 SymPy 代码。

        Args:
            code: 要执行的 SymPy 代码字符串

        Returns:
            执行结果字符串
        """
        code_lower = code.lower()
        for keyword in _BANNED_KEYWORDS:
            if keyword.lower() in code_lower:
                return f"Error: banned keyword '{keyword}' detected in code"

        full_code = _SYMPY_PRE_IMPORTS + "\n# --- user code ---\n" + code

        try:
            result = subprocess.run(
                [sys.executable, "-c", full_code],
                capture_output=True,
                text=True,
                timeout=SymPyTool.TIMEOUT,
            )

            output = result.stdout
            if result.stderr:
                output += "\n[stderr]\n" + result.stderr

            output = output.strip()
            if len(output) > SymPyTool.MAX_OUTPUT_LENGTH:
                output = output[: SymPyTool.MAX_OUTPUT_LENGTH] + "\n... (output truncated)"

            return output if output else "(no output)"

        except subprocess.TimeoutExpired:
            return f"Error: execution timed out after {SymPyTool.TIMEOUT}s"
        except Exception as e:
            return f"Error: {str(e)}"
