"""Python 代码沙箱执行器：在受限环境中执行 Python 代码。"""

import subprocess
import sys
from pathlib import Path
from typing import Optional

from .registry import Tool

# 预导入的模块列表（在沙箱中自动可用）
_PRE_IMPORTS = """
import math
import sympy
from sympy import symbols, solve, simplify, expand, factor, diff, integrate, limit, Matrix, Rational
from fractions import Fraction
from decimal import Decimal
import itertools
import collections
import statistics
import random
"""

# 禁用的危险模块/函数
_BANNED_KEYWORDS = [
    "os.", "subprocess", "__import__", "eval(", "exec(",
    "open(", "file(", "compile(", "input(",
    "sys.", "shutil", "pathlib", "socket",
]


class PythonExecutor:
    """Python 代码沙箱执行器。

    安全措施：
    - 使用 subprocess 隔离执行，带超时限制
    - 预设可用模块列表
    - 输出截断至安全长度
    """

    TOOL_NAME = "python_exec"
    TOOL_DESCRIPTION = "执行 Python 代码（支持 math, sympy, fractions 等），返回输出结果"
    TIMEOUT = 5  # 秒
    MAX_OUTPUT_LENGTH = 2000  # 字符

    @classmethod
    def get_tool(cls) -> Tool:
        """创建 PythonExecutor 工具实例。"""
        return Tool(
            name=cls.TOOL_NAME,
            description=cls.TOOL_DESCRIPTION,
            parameters={"code": {"type": "string", "description": "要执行的 Python 代码"}},
            func=cls.execute,
            timeout=cls.TIMEOUT,
        )

    @staticmethod
    def execute(code: str) -> str:
        """在沙箱中执行 Python 代码。

        Args:
            code: 要执行的 Python 代码字符串

        Returns:
            执行结果（stdout + stderr，被截断）
        """
        # 安全检查：检查危险关键字
        code_lower = code.lower()
        for keyword in _BANNED_KEYWORDS:
            if keyword.lower() in code_lower:
                return f"Error: banned keyword '{keyword}' detected in code"

        # 构造完整脚本（预导入 + 用户代码）
        full_code = _PRE_IMPORTS + "\n" + code

        try:
            result = subprocess.run(
                [sys.executable, "-c", full_code],
                capture_output=True,
                text=True,
                timeout=PythonExecutor.TIMEOUT,
            )

            output = result.stdout
            if result.stderr:
                output += "\n[stderr]\n" + result.stderr

            output = output.strip()
            if len(output) > PythonExecutor.MAX_OUTPUT_LENGTH:
                output = output[: PythonExecutor.MAX_OUTPUT_LENGTH] + "\n... (output truncated)"

            return output if output else "(no output)"

        except subprocess.TimeoutExpired:
            return f"Error: execution timed out after {PythonExecutor.TIMEOUT}s"
        except Exception as e:
            return f"Error: {str(e)}"
