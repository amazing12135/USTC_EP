"""奖励函数单元测试。"""

import pytest

from rl.reward import RewardFunction, RewardConfig


class DummyTrajectory:
    """用于测试的 Dummy 轨迹对象。"""

    def __init__(self, is_correct, total_turns, steps=None):
        self.is_correct = is_correct
        self.total_turns = total_turns
        self.steps = steps or []


class DummyStep:
    """用于测试的 Dummy 步骤对象。"""

    def __init__(self, thought=""):
        self.thought = thought
        self.action = None
        self.observation = None


class TestRewardFunction:
    """奖励函数测试。"""

    def test_outcome_correct(self):
        rf = RewardFunction()
        traj = DummyTrajectory(is_correct=True, total_turns=3)
        reward = rf.compute(traj)
        assert reward > 0.5  # 正确的轨迹奖励应接近 1.0

    def test_outcome_incorrect(self):
        rf = RewardFunction()
        traj = DummyTrajectory(is_correct=False, total_turns=3)
        reward = rf.compute(traj)
        assert reward < 0.5  # 错误的轨迹奖励应 <= format_bonus

    def test_format_reward_with_answer_tag(self):
        rf = RewardFunction()
        steps = [
            DummyStep(thought="Let me think.\n<answer>42</answer>"),
        ]
        traj = DummyTrajectory(is_correct=True, total_turns=1, steps=steps)
        reward = rf.compute(traj)
        assert reward > 0.5

    def test_format_reward_with_valid_tool_call(self):
        rf = RewardFunction()
        steps = [
            DummyStep(
                thought='<tool_call>{"name": "calc", "args": {"expr": "1+1"}}</tool_call>\n<answer>2</answer>'
            ),
        ]
        traj = DummyTrajectory(is_correct=True, total_turns=1, steps=steps)
        reward = rf.compute(traj)
        assert reward > 0.5

    def test_format_reward_with_invalid_tool_call(self):
        rf = RewardFunction()
        steps = [
            DummyStep(thought="<tool_call>not json</tool_call>"),
        ]
        traj = DummyTrajectory(is_correct=False, total_turns=1, steps=steps)
        reward = rf.compute(traj)
        # 应该有负的 format 分
        assert reward <= 0

    def test_efficiency_penalty_disabled_by_default(self):
        rf = RewardFunction()
        assert rf.config.step_penalty == 0.0
        assert rf._efficiency_penalty(100) == 0.0

    def test_efficiency_penalty_enabled(self):
        config = RewardConfig(step_penalty=1.0, penalty_per_step=0.01)
        rf = RewardFunction(config=config)
        penalty = rf._efficiency_penalty(10)
        assert penalty == 0.1
