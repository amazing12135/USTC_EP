"""SFT 训练数据构造：生成工具调用格式的监督微调数据。

两种策略：
- 策略A: 用 GPT-4 生成高质量示例（需要 API）
- 策略B: 手写 few-shot 模板 + GSM8K 自动合成（零成本，当前默认）
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .dataset import Problem
from .sft_templates import get_templates
from environment.tools.calculator import Calculator


class SFTDataGenerator:
    """SFT 训练数据生成器。

    目的：构造 SFT 训练数据，让模型学会工具调用格式。
    Phase 1 策略B（零成本）：
    1. 使用 20 条手写 few-shot 模板作为高质量训练样本
    2. 对 GSM8K 问题，从答案文本中提取算术表达式，合成为 calculator 工具调用示例
    """

    SYSTEM_PROMPT_PATH = Path("config/prompts/system.txt")

    def __init__(self, output_dir: str | Path = "data/sft") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """加载系统提示词文件。"""
        if self.SYSTEM_PROMPT_PATH.exists():
            return self.SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
        return self._default_system_prompt()

    def generate_from_templates(
        self,
        problems: List[Problem],
        templates: Optional[List[Dict[str, Any]]] = None,
        output_path: Optional[str | Path] = None,
    ) -> List[Dict[str, Any]]:
        """策略B：用 Few-shot 模板 + GSM8K 自动合成构造 SFT 数据。

        流程：
        1. 将 20 条手写 few-shot 模板直接纳入训练集
        2. 对每条 GSM8K 问题，自动从答案文本中提取算术运算，
           生成为带 calculator 工具调用的 assistant 回复

        Args:
            problems: GSM8K L1 问题列表
            templates: 额外的手写模板
            output_path: 输出 JSONL 文件路径

        Returns:
            ChatML 格式的训练数据
        """
        data: List[Dict[str, Any]] = []

        # ---- Part 1: 手写 few-shot 模板作为高质量训练样本 ----
        shot_templates = templates or get_templates()
        for tpl in shot_templates:
            entry = {
                "messages": [
                    {"role": "system", "content": self._system_prompt},
                ]
                + tpl["messages"]  # user + assistant
            }
            data.append(entry)

        # ---- Part 2: GSM8K 自动合成 ----
        for problem in problems:
            assistant_response = self._synthesize_response(problem)
            if not assistant_response:
                continue

            entry = {
                "messages": [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": f"问题：{problem.question}"},
                    {"role": "assistant", "content": assistant_response},
                ]
            }
            data.append(entry)

        # ---- 保存 ----
        output_path = output_path or self.output_dir / "sft_data.jsonl"
        output_path = Path(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return data

    def _synthesize_response(self, problem: Problem) -> str:
        """为 GSM8K 问题自动合成带工具调用的 assistant 回复。

        策略：从 GSM8K 的 answer 字段（含完整推理过程）中提取
        算术表达式（如 "3+4=7"），将其转换为 calculator 工具调用，
        并用工具实际执行结果替换原来的等式。

        Args:
            problem: GSM8K 问题

        Returns:
            ReAct 格式的 assistant 回复，如果无法合成则返回空字符串
        """
        answer_text = problem.answer.strip()

        # GSM8K 答案格式: 推理文本 ... #### 最终答案
        parts = answer_text.split("####")
        reasoning = parts[0].strip() if len(parts) > 1 else answer_text
        final_answer = parts[-1].strip() if len(parts) > 1 else ""

        if not final_answer:
            final_answer = reasoning

        # GSM8K 格式: <<A op B = C>>  先提取这种
        gsm_pattern = re.compile(r"<<(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)>>")
        gsm_matches = gsm_pattern.findall(reasoning)
        if gsm_matches:
            matches = gsm_matches
        else:
            # 回退: 纯文本格式 "A op B = C"
            expr_pattern = re.compile(
                r"(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)"
            )
            matches = expr_pattern.findall(reasoning)

        if not matches:
            # 无显式算术表达式 → 使用终值直接构造
            return self._build_simple_response(reasoning, final_answer)

        # 将推理文本中的算术表达式替换为工具调用 + Observed
        lines = reasoning.split("\n")
        tool_calls_block: List[str] = []
        used_expressions: set = set()

        for match in matches:
            a, op, b, c = match
            expr_str = f"{a} {op} {b}"
            # 去重：同一个表达式只调用一次
            if expr_str in used_expressions:
                continue
            used_expressions.add(expr_str)

            # 用 calculator 实际执行
            tool_result = Calculator.calculate(expr_str)
            tool_calls_block.append(
                f'<tool_call>{{"name": "calculator", "args": {{"expression": "{expr_str}"}}}}</tool_call>'
            )
            tool_calls_block.append(f"Observed: {tool_result}")

        if not tool_calls_block:
            return self._build_simple_response(reasoning, final_answer)

        # 清理 GSM8K <<...>> 标记，保留纯文本
        clean_reasoning = re.sub(r"<<[^>]+>>", "", reasoning)
        clean_reasoning = re.sub(r"\s+", " ", clean_reasoning).strip()

        steps = []
        sentences = [s.strip() for s in clean_reasoning.split(".") if s.strip()]
        if sentences:
            steps.append(f"Thought: {sentences[0]}.")

        steps.extend(tool_calls_block)

        if len(sentences) > 1:
            mid_thought = ". ".join(sentences[1:-1]) if len(sentences) > 2 else sentences[-1]
            if mid_thought and mid_thought not in str(steps):
                steps.append(f"Thought: {mid_thought}.")

        steps.append("Thought: 计算完成，得出最终答案。")
        steps.append(f"<answer>{final_answer}</answer>")

        return "\n\n".join(steps)

    def _build_simple_response(
        self, reasoning: str, final_answer: str
    ) -> str:
        """构建简化版回复（无法提取算术表达式时的回退方案）。

        直接使用 calculator 对 final_answer 做一次"恒等计算"，
        确保工具调用格式出现但不扭曲逻辑。
        """
        # 用 final_answer 构造一个无害计算
        clean_ans = final_answer.strip().replace(",", "").replace(" ", "")
        if re.match(r"^-?\d+\.?\d*$", clean_ans):
            expr = f"{clean_ans} + 0"
            tool_result = Calculator.calculate(expr)
        else:
            # 非纯数值答案，跳过
            return ""

        thought_summary = reasoning[:150].strip() if len(reasoning) > 150 else reasoning
        return (
            f"Thought: {thought_summary}\n\n"
            f'<tool_call>{{"name": "calculator", "args": {{"expression": "{expr}"}}}}</tool_call>\n\n'
            f"Observed: {tool_result}\n\n"
            f"Thought: 确认计算结果。\n\n"
            f"<answer>{final_answer}</answer>"
        )

    @staticmethod
    def _default_system_prompt() -> str:
        """回退：硬编码 system prompt（当 system.txt 不可用时）。"""
        return """你是一个数学解题助手。你可以使用以下工具：

1. calculator - 计算数学表达式
   参数: {"expression": "数学表达式"}

请使用ReAct格式：
1. Thought: 分析问题
2. <tool_call>{"name": "calculator", "args": {...}}</tool_call> 调用工具
3. Observed: 结果
4. <answer>答案</answer> 输出最终结果"""
