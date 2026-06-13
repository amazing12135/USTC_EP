"""ReAct Agent：实现 Thought → Action → Observation 循环的核心执行引擎。"""

from typing import Any, List, Optional, Protocol

from .action_parser import ActionParser, ActionType, ParsedAction
from .memory import AgentMemory, AgentTrajectory, Step
from .planner import Plan, TaskPlanner
from environment.math_env import MathEnvironment, ToolCall


class LLMInterface(Protocol):
    """LLM 推理接口协议。"""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        ...


class ReActAgent:
    """ReAct 推理循环 Agent。

    实现 Thought → Action → Observation 循环。
    支持工具调用、最终答案输出、超时停止。

    Attributes:
        model: LLM 推理接口
        env: 数学环境
        memory: Agent 记忆管理器
        planner: 任务规划器
        action_parser: 动作解析器
        max_turns: 最大推理轮次
        system_prompt: 系统提示词
        max_consecutive_errors: 连续错误最大容忍次数
    """

    def __init__(
        self,
        model: Optional[LLMInterface] = None,
        max_turns: int = 10,
        system_prompt: Optional[str] = None,
        env: Optional[MathEnvironment] = None,
        memory: Optional[AgentMemory] = None,
        planner: Optional[TaskPlanner] = None,
    ) -> None:
        """初始化 ReAct Agent。

        Args:
            model: LLM 推理接口（本地测试时可使用 MockLLM）
            max_turns: 最大推理轮次
            system_prompt: 系统提示词
            env: 数学环境，不提供则创建默认环境
            memory: 记忆管理器
            planner: 任务规划器
        """
        self.model = model
        self.max_turns = max_turns
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.env = env or MathEnvironment(max_steps=max_turns)
        self.memory = memory or AgentMemory()
        self.planner = planner or TaskPlanner()
        self.action_parser = ActionParser()
        self.max_consecutive_errors = 3

    def run(self, question: str, ground_truth: str = "") -> AgentTrajectory:
        """主入口：执行完整推理并返回轨迹。

        Args:
            question: 数学问题文本
            ground_truth: 标准答案（可选，用于结果判断）

        Returns:
            AgentTrajectory 完整推理轨迹
        """
        # 初始化环境和记忆
        self.env.reset(question, ground_truth)
        self.memory.clear()
        self.memory.add_message("system", self.system_prompt)
        self.memory.add_message("user", f"问题：{question}")

        steps: List[Step] = []
        final_answer = ""
        consecutive_errors = 0

        # 可选：预规划
        if self.planner.enabled and self.model:
            plan = self.planner.plan(question, self.model)
            self.memory.add_message("system", f"计划：{plan.steps}")

        # ReAct 循环
        for turn in range(1, self.max_turns + 1):
            # THINK: 生成推理/动作
            context = self.memory.get_context()
            llm_output = self._think(context)

            if llm_output is None:
                break

            # 解析动作
            action = self.action_parser.parse(llm_output)

            # OBSERVE: 执行动作或停止
            if action.action_type == ActionType.FINAL_ANSWER:
                final_answer = action.answer or ""
                steps.append(
                    Step(
                        turn=turn,
                        thought=llm_output,
                        action=None,
                        observation=None,
                    )
                )
                break

            elif action.action_type == ActionType.TOOL_CALL:
                tool_call = ToolCall(
                    name=action.tool_name or "",
                    args=action.tool_args,
                )
                step_result = self._act(tool_call)
                obs = step_result.observation.content
                steps.append(
                    Step(
                        turn=turn,
                        thought=llm_output,
                        action=tool_call,
                        observation=obs,
                    )
                )
                self.memory.add_message("assistant", llm_output)
                self.memory.add_message("tool", obs)

                if step_result.info.get("success"):
                    consecutive_errors = 0
                else:
                    consecutive_errors += 1

            elif action.action_type == ActionType.THOUGHT:
                steps.append(
                    Step(
                        turn=turn,
                        thought=llm_output,
                        action=None,
                        observation=None,
                    )
                )
                self.memory.add_message("assistant", llm_output)
                consecutive_errors = 0

            else:  # PARSE_ERROR
                consecutive_errors += 1
                steps.append(
                    Step(
                        turn=turn,
                        thought=llm_output,
                        action=None,
                        observation=f"Parse error: {action.error_reason}",
                    )
                )

            # 停止条件检查
            if self._should_stop(turn, consecutive_errors):
                break

        # 如果没有显式输出答案，尝试从最后一步提取
        if not final_answer:
            if steps:
                final_answer = self._extract_answer(steps)

        # 构建轨迹
        is_correct = False
        if ground_truth and final_answer:
            from evaluation.math_judge import MathJudge
            judge = MathJudge()
            is_correct = judge.judge(final_answer, ground_truth)

        trajectory = AgentTrajectory(
            question=question,
            ground_truth=ground_truth,
            steps=steps,
            final_answer=final_answer,
            total_turns=len(steps),
            is_correct=is_correct,
        )

        return trajectory

    def _think(self, history: List[Any]) -> Optional[str]:
        """调用 LLM 生成下一步推理。

        Args:
            history: 对话历史

        Returns:
            LLM 生成的文本
        """
        if self.model is None:
            return None

        # 构建 prompt
        if isinstance(history, list):
            prompt = "\n".join(
                f"{m['role']}: {m['content']}" for m in history
                if isinstance(m, dict)
            )
        else:
            prompt = str(history)

        return self.model.generate(prompt)

    def _act(self, action: ToolCall) -> Any:
        """执行工具调用。

        Args:
            action: 工具调用请求

        Returns:
            StepResult 执行结果
        """
        return self.env.step(action)

    def _should_stop(self, turn: int, consecutive_errors: int) -> bool:
        """判断是否应该提前结束推理。

        Args:
            turn: 当前轮次
            consecutive_errors: 连续错误次数

        Returns:
            是否应该停止
        """
        if turn >= self.max_turns:
            return True
        if consecutive_errors >= self.max_consecutive_errors:
            return True
        return False

    def _extract_answer(self, steps: List[Step]) -> str:
        """从轨迹中提取最终答案。

        Args:
            steps: 推理步骤列表

        Returns:
            提取的答案字符串
        """
        # 反向遍历查找最终答案标记
        for step in reversed(steps):
            answer = self.action_parser.extract_answer(step.thought)
            if answer:
                return answer
        # 返回最后一步的推理文本作为答案
        if steps:
            return steps[-1].thought
        return ""

    @staticmethod
    def _default_system_prompt() -> str:
        """默认系统提示词。"""
        return """你是一个数学解题助手，擅长通过推理和工具使用来解决数学问题。

你可以使用以下工具：

1. calculator - 计算数学表达式
   参数: {"expression": "数学表达式"}

2. python_exec - 执行Python代码
   参数: {"code": "Python代码"}

3. sympy - 符号数学计算
   参数: {"code": "SymPy代码"}

请按以下格式进行推理：

1. 分析问题，思考解决方案
2. 如需计算，使用 <tool_call>{"name": "工具名", "args": {...}}</tool_call> 调用工具
3. 根据工具返回的结果继续推理
4. 重复步骤2-3直到得出最终答案
5. 用 <answer>答案</answer> 输出最终结果

注意：
- 最终答案应该放在 <answer></answer> 标签中
- 对于选择题，答案应该只包含选项字母
- 对于数值答案，请使用最简形式（如分数约分、开方化简）
"""
