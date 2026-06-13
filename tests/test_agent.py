"""Agent 层单元测试。"""

import pytest

from agent.action_parser import ActionParser, ActionType, ParsedAction
from agent.planner import TaskPlanner, Plan
from agent.memory import AgentMemory, MemoryRecord, AgentTrajectory, Step


class MockLLM:
    """模拟 LLM 输出，用于本地测试 Agent 逻辑。"""

    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0

    def generate(self, prompt, **kwargs):
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            response = "<answer>42</answer>"
        self.call_count += 1
        return response


class TestActionParser:
    """动作解析器测试。"""

    def setup_method(self):
        self.parser = ActionParser()

    def test_parse_thought(self):
        result = self.parser.parse("Let me think about this problem.")
        assert result.action_type == ActionType.THOUGHT
        assert result.content == "Let me think about this problem."

    def test_parse_tool_call(self):
        output = '<tool_call>{"name": "calculator", "args": {"expression": "2+3"}}</tool_call>'
        result = self.parser.parse(output)
        assert result.action_type == ActionType.TOOL_CALL
        assert result.tool_name == "calculator"
        assert result.tool_args == {"expression": "2+3"}

    def test_parse_tool_call_with_text(self):
        output = 'Thought: I need to calculate.\n<tool_call>{"name": "sympy", "args": {"code": "solve(x-1, x)"}}</tool_call>'
        result = self.parser.parse(output)
        assert result.action_type == ActionType.TOOL_CALL
        assert result.tool_name == "sympy"

    def test_parse_final_answer(self):
        output = "The answer is <answer>42</answer>"
        result = self.parser.parse(output)
        assert result.action_type == ActionType.FINAL_ANSWER
        assert result.answer == "42"

    def test_parse_answer_priority_over_tool_call(self):
        output = '<tool_call>{"name": "calc"}</tool_call> and <answer>yes</answer>'
        result = self.parser.parse(output)
        # answer 优先级高于 tool_call
        assert result.action_type == ActionType.FINAL_ANSWER
        assert result.answer == "yes"

    def test_parse_invalid_json(self):
        output = '<tool_call>not valid json</tool_call>'
        result = self.parser.parse(output)
        assert result.action_type == ActionType.PARSE_ERROR

    def test_parse_empty(self):
        result = self.parser.parse("")
        assert result.action_type == ActionType.PARSE_ERROR

    def test_is_final_answer(self):
        assert self.parser.is_final_answer("<answer>yes</answer>") is True
        assert self.parser.is_final_answer("no answer here") is False

    def test_extract_answer(self):
        assert self.parser.extract_answer("<answer>42</answer>") == "42"
        assert self.parser.extract_answer("nothing") is None

    def test_has_tool_call(self):
        assert self.parser.has_tool_call('<tool_call>{"name":"x"}</tool_call>') is True
        assert self.parser.has_tool_call("no tool call") is False


class TestPlanner:
    """规划器测试。"""

    def test_plan_disabled(self):
        planner = TaskPlanner(enabled=False)
        plan = planner.plan("What is 1+1?")
        assert plan.original_question == "What is 1+1?"
        assert plan.steps == []

    def test_plan_without_llm(self):
        planner = TaskPlanner(enabled=True)
        plan = planner.plan("What is 1+1?", llm=None)
        assert plan.steps == []

    def test_plan_with_llm(self):
        planner = TaskPlanner(enabled=True)
        mock_llm = MockLLM(["步骤1: 分析\n步骤2: 计算\n建议工具: calculator"])
        plan = planner.plan("What is 2+3?", llm=mock_llm)
        assert len(plan.steps) > 0


class TestAgentMemory:
    """Agent 记忆测试。"""

    def test_short_term_memory(self):
        mem = AgentMemory()
        mem.add_message("user", "hi")
        mem.add_message("assistant", "hello")
        ctx = mem.get_context()
        assert len(ctx) == 2

    def test_clear(self):
        mem = AgentMemory()
        mem.add_message("user", "hi")
        mem.clear()
        assert len(mem.get_context()) == 0

    def test_save_trajectory(self, tmp_path):
        mem = AgentMemory(memory_dir=tmp_path / "memory")
        traj = AgentTrajectory(
            question="1+1",
            ground_truth="2",
            steps=[Step(turn=1, thought="Think...")],
            final_answer="2",
            total_turns=1,
            is_correct=True,
        )
        path = mem.save_trajectory(traj, status="success")
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "1+1" in content
        assert "success" in content

    def test_retrieve_without_index(self):
        mem = AgentMemory()
        records = mem.retrieve_similar("test")
        assert records == []

    def test_summarize_old_no_need(self):
        """消息很少时不应压缩。"""
        mem = AgentMemory(max_context_length=2048)
        mem.add_message("user", "hi")
        mem.add_message("assistant", "hello")
        summary = mem.summarize_old()
        assert summary == ""
        assert len(mem.get_context()) == 2  # 未被改动

    def test_summarize_old_compresses(self):
        """消息超窗口一半时应触发压缩。"""
        mem = AgentMemory(max_context_length=40)  # 极小窗口触发压缩
        for i in range(12):
            mem.add_message("assistant", f"msg {i}")
        summary = mem.summarize_old()
        assert summary != ""
        assert len(mem.get_context()) < 12  # 消息数减少了


class TestReActAgent:
    """ReAct Agent 测试（使用 MockLLM）。"""

    def test_run_with_final_answer(self):
        mock_llm = MockLLM([
            "Let me think...<answer>42</answer>"
        ])
        from agent.react_agent import ReActAgent
        from agent.planner import TaskPlanner
        agent = ReActAgent(
            model=mock_llm, max_turns=3, planner=TaskPlanner(enabled=False)
        )
        traj = agent.run("What is 6*7?", "42")
        assert traj.final_answer == "42"
        assert traj.is_correct is True
        assert traj.total_turns == 1

    def test_run_with_tool_call(self):
        mock_llm = MockLLM([
            '<tool_call>{"name": "calculator", "args": {"expression": "6*7"}}</tool_call>',
            'Got it.<answer>42</answer>',
        ])
        from agent.react_agent import ReActAgent
        from agent.planner import TaskPlanner
        agent = ReActAgent(
            model=mock_llm, max_turns=5, planner=TaskPlanner(enabled=False)
        )
        traj = agent.run("What is 6*7?", "42")
        assert traj.final_answer == "42"
        assert traj.total_turns == 2
        # 第一步有 tool call
        assert traj.steps[0].action is not None
        # 第二步有 final answer
        assert traj.steps[1].thought == 'Got it.<answer>42</answer>'

    def test_run_no_model(self):
        from agent.react_agent import ReActAgent
        agent = ReActAgent(model=None, max_turns=3)
        traj = agent.run("test", "")
        assert traj.total_turns == 0
