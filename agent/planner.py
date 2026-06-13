"""任务规划器：在推理开始前对问题进行初步分析和步骤规划。"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol


class LLMInterface(Protocol):
    """LLM 推理接口协议，用于类型标注。"""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        ...


@dataclass
class Plan:
    """推理计划。"""

    original_question: str
    steps: List[str] = field(default_factory=list)  # 规划的子步骤
    suggested_tools: List[str] = field(default_factory=list)  # 建议使用的工具
    raw_output: str = ""  # LLM 原始输出


class TaskPlanner:
    """任务规划器。在 Agent 推理前对问题进行预处理和步骤规划。

    Planner 为可选模块。对于简单题目（GSM8K 级别），一步规划就足够；
    对于竞赛题（MATH/AMC），可开启多步规划获取更好的推理结构。
    """

    def __init__(self, enabled: bool = True) -> None:
        """初始化规划器。

        Args:
            enabled: 是否启用规划功能
        """
        self.enabled = enabled

    def plan(self, question: str, llm: Optional[LLMInterface] = None) -> Plan:
        """生成初始推理计划。

        Args:
            question: 数学问题文本
            llm: LLM 推理接口（可选），如果不提供则返回空计划

        Returns:
            推理计划
        """
        if not self.enabled or llm is None:
            return Plan(original_question=question)

        prompt = self._build_planning_prompt(question)
        raw_output = llm.generate(prompt)
        return self._parse_plan(question, raw_output)

    def revise(
        self,
        plan: Plan,
        observation: str,
        llm: Optional[LLMInterface] = None,
    ) -> Plan:
        """根据新的观察调整计划。

        Args:
            plan: 当前计划
            observation: 新观察内容
            llm: LLM 推理接口

        Returns:
            调整后的计划
        """
        if not self.enabled or llm is None:
            return plan

        prompt = self._build_revision_prompt(plan, observation)
        raw_output = llm.generate(prompt)
        return self._parse_plan(plan.original_question, raw_output)

    def _build_planning_prompt(self, question: str) -> str:
        """构建规划用的 prompt。"""
        return (
            f"问题：{question}\n\n"
            "请分析这个问题，列出解决步骤和可能需要的工具。\n"
            "格式：\n"
            "步骤1: ...\n"
            "步骤2: ...\n"
            "建议工具: ..."
        )

    def _build_revision_prompt(self, plan: Plan, observation: str) -> str:
        """构建计划修订 prompt。"""
        steps_text = "\n".join(f"- {s}" for s in plan.steps)
        return (
            f"原始计划：\n{steps_text}\n\n"
            f"新观察：{observation}\n\n"
            "根据新观察，修订计划。"
        )

    def _parse_plan(self, question: str, raw_output: str) -> Plan:
        """解析 LLM 输出的计划文本。"""
        steps: List[str] = []
        suggested_tools: List[str] = []

        for line in raw_output.strip().split("\n"):
            line = line.strip()
            if line.lower().startswith(("步骤", "step")):
                steps.append(line)
            elif "工具" in line:
                suggested_tools.append(line)

        return Plan(
            original_question=question,
            steps=steps,
            suggested_tools=suggested_tools,
            raw_output=raw_output,
        )
