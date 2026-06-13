"""数据预处理：答案标准化、数值提取、Prompt 格式化。"""

import re
from fractions import Fraction
from typing import Any, Dict, List, Optional


class DataPreprocessor:
    """数据预处理器。"""

    @staticmethod
    def normalize_answer(answer: str) -> str:
        """标准化答案格式。

        操作：去空格、统一分数格式、去除多余的标点。

        Args:
            answer: 原始答案字符串

        Returns:
            标准化后的答案
        """
        if not answer:
            return ""

        ans = answer.strip()
        # 去多余空格
        ans = re.sub(r"\s+", " ", ans)
        # 移除末尾的句号
        ans = ans.rstrip(".")
        # 统一 LaTeX 中的空格
        ans = ans.replace("\\,", "").replace("\\;", "").replace("\\ ", "")

        return ans

    @staticmethod
    def extract_numeric(answer: str) -> Optional[float]:
        """从答案文本中提取数值。

        支持分数 "3/4"、小数 "0.75"、科学计数法 "1.5e-3"。

        Args:
            answer: 答案文本

        Returns:
            提取的数值，无法提取时返回 None
        """
        if not answer:
            return None

        ans = answer.strip()

        # 尝试分数格式
        frac_match = re.match(r"^(-?\d+)\s*/\s*(-?\d+)$", ans)
        if frac_match:
            num, den = int(frac_match.group(1)), int(frac_match.group(2))
            if den != 0:
                return num / den

        # 尝试小数/整数
        num_match = re.search(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", ans)
        if num_match:
            try:
                return float(num_match.group(0))
            except ValueError:
                pass

        return None

    @staticmethod
    def format_prompt(
        question: str,
        tools_schema: str,
        system_prompt: Optional[str] = None,
        few_shot_examples: Optional[List[str]] = None,
    ) -> str:
        """将问题格式化为带系统提示的完整 prompt。

        Args:
            question: 原始问题文本
            tools_schema: 工具的描述 schema
            system_prompt: 系统提示词（可选）
            few_shot_examples: few-shot 示例列表（可选）

        Returns:
            完整格式化的 prompt 字符串
        """
        parts: List[str] = []

        if system_prompt:
            parts.append(f"<|system|>\n{system_prompt}\n{tools_schema}")

        if few_shot_examples:
            for example in few_shot_examples:
                parts.append(example)

        parts.append(f"<|user|>\n问题：{question}\n<|assistant|>")

        return "\n\n".join(parts)

    @staticmethod
    def format_chatml(
        messages: List[Dict[str, str]],
    ) -> str:
        """将对话历史格式化为 ChatML 格式。

        Args:
            messages: 消息列表 [{"role": "...", "content": "..."}]

        Returns:
            ChatML 格式字符串
        """
        parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)
