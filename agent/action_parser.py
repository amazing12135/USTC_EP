"""动作解析器：解析 LLM 原始输出，提取工具调用或最终答案。"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional


class ActionType(Enum):
    """动作类型枚举。"""

    TOOL_CALL = auto()  # 工具调用
    FINAL_ANSWER = auto()  # 最终答案
    THOUGHT = auto()  # 纯推理文本
    PARSE_ERROR = auto()  # 解析错误


@dataclass
class ParsedAction:
    """解析后的动作。"""

    action_type: ActionType
    content: str = ""  # 原文本
    tool_name: Optional[str] = None  # 工具名称（TOOL_CALL 时）
    tool_args: Dict[str, Any] = field(default_factory=dict)  # 工具参数（TOOL_CALL 时）
    answer: Optional[str] = None  # 最终答案（FINAL_ANSWER 时）
    error_reason: Optional[str] = None  # 错误原因（PARSE_ERROR 时）


class ActionParser:
    """解析 LLM 输出中的动作描述。

    支持的格式：
    - <tool_call>{"name": "...", "args": {...}}</tool_call> — 工具调用
    - <answer>...</answer> — 最终答案
    - 普通文本 — 视为推理文本（THOUGHT）
    """

    # 正则表达式
    TOOL_CALL_PATTERN = re.compile(
        r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL
    )
    ANSWER_PATTERN = re.compile(
        r"<answer>\s*(.*?)\s*</answer>", re.DOTALL
    )

    def parse(self, llm_output: str) -> ParsedAction:
        """解析 LLM 的原始输出。

        Args:
            llm_output: LLM 生成的原始文本

        Returns:
            ParsedAction 解析结果
        """
        if not llm_output or not llm_output.strip():
            return ParsedAction(
                action_type=ActionType.PARSE_ERROR,
                content=llm_output,
                error_reason="empty output",
            )

        # 1. 检查 <answer> 标签（优先级最高，因为是最终输出）
        answer_match = self.ANSWER_PATTERN.search(llm_output)
        if answer_match:
            return ParsedAction(
                action_type=ActionType.FINAL_ANSWER,
                content=llm_output,
                answer=answer_match.group(1).strip(),
            )

        # 2. 检查 <tool_call> 标签
        tool_match = self.TOOL_CALL_PATTERN.search(llm_output)
        if tool_match:
            return self._parse_tool_call(llm_output, tool_match.group(1))

        # 3. 视为纯推理文本
        return ParsedAction(
            action_type=ActionType.THOUGHT,
            content=llm_output,
        )

    def _parse_tool_call(self, raw_output: str, json_str: str) -> ParsedAction:
        """解析工具调用 JSON。"""
        try:
            data = json.loads(json_str.strip())
            name = data.get("name", "")
            args = data.get("args", {})
            if not name:
                return ParsedAction(
                    action_type=ActionType.PARSE_ERROR,
                    content=raw_output,
                    error_reason="tool_call missing 'name' field",
                )
            return ParsedAction(
                action_type=ActionType.TOOL_CALL,
                content=raw_output,
                tool_name=name,
                tool_args=args,
            )
        except json.JSONDecodeError as e:
            return ParsedAction(
                action_type=ActionType.PARSE_ERROR,
                content=raw_output,
                error_reason=f"JSON parse error: {str(e)}",
            )

    def is_final_answer(self, llm_output: str) -> bool:
        """判断 LLM 输出是否包含最终答案标记。

        Args:
            llm_output: LLM 生成的原始文本

        Returns:
            是否包含最终答案
        """
        return bool(self.ANSWER_PATTERN.search(llm_output))

    def extract_answer(self, llm_output: str) -> Optional[str]:
        """从 LLM 输出中提取最终答案。

        Args:
            llm_output: LLM 生成的原始文本

        Returns:
            提取的答案字符串，如果未找到则返回 None
        """
        match = self.ANSWER_PATTERN.search(llm_output)
        if match:
            return match.group(1).strip()
        return None

    def has_tool_call(self, llm_output: str) -> bool:
        """判断 LLM 输出是否包含工具调用。

        Args:
            llm_output: LLM 生成的原始文本

        Returns:
            是否包含工具调用
        """
        return bool(self.TOOL_CALL_PATTERN.search(llm_output))
