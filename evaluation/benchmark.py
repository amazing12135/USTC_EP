"""基准测试运行器：在不同数据集上评估模型性能。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EvalResult:
    """单次评估结果。"""

    accuracy: float = 0.0
    per_category: Dict[str, float] = field(default_factory=dict)
    total_samples: int = 0
    correct_samples: int = 0
    avg_turns: float = 0.0
    trajectories: List[Any] = field(default_factory=list)
    error_analysis: Dict[str, int] = field(default_factory=dict)


class BenchmarkRunner:
    """基准测试运行器。

    在标准数据集上运行模型评估，支持：
    - GSM8K 评估
    - MATH 按类别评估
    - 模型对比
    """

    def __init__(
        self,
        agent: Optional[Any] = None,
        dataset_manager: Optional[Any] = None,
        output_dir: str | Path = "outputs/evals",
    ) -> None:
        """初始化基准测试运行器。

        Args:
            agent: ReActAgent 实例
            dataset_manager: DatasetManager 实例
            output_dir: 评估结果输出目录
        """
        self.agent = agent
        self.dataset_manager = dataset_manager
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_eval(
        self,
        problems: List[Any],
        dataset_name: str = "unknown",
    ) -> EvalResult:
        """在指定问题集上运行评估。

        Args:
            problems: 问题列表
            dataset_name: 数据集名称

        Returns:
            EvalResult 评估结果
        """
        if self.agent is None:
            return EvalResult()

        correct = 0
        total = len(problems)
        total_turns = 0
        trajectories = []
        error_counts: Dict[str, int] = {
            "reasoning_error": 0,
            "tool_error": 0,
            "format_error": 0,
            "answer_extraction_error": 0,
            "capability_boundary": 0,
        }

        for problem in problems:
            traj = self.agent.run(problem.question, problem.answer)
            trajectories.append(traj)
            total_turns += traj.total_turns

            if traj.is_correct:
                correct += 1
            else:
                # 简化的错误分类
                error_type = self._classify_error(traj)
                error_counts[error_type] = error_counts.get(error_type, 0) + 1

        return EvalResult(
            accuracy=correct / total if total > 0 else 0.0,
            total_samples=total,
            correct_samples=correct,
            avg_turns=total_turns / total if total > 0 else 0.0,
            trajectories=trajectories,
            error_analysis=error_counts,
        )

    def run_on_gsm8k(self) -> EvalResult:
        """在 GSM8K 验证集上评估。"""
        if self.dataset_manager is None:
            return EvalResult()
        problems = self.dataset_manager.load_l1_data()
        return self.run_eval(problems, "gsm8k")

    def run_on_math(
        self, categories: Optional[List[str]] = None
    ) -> EvalResult:
        """在 MATH 数据集上评估。

        Args:
            categories: 可选，只评估指定子类

        Returns:
            EvalResult 评估结果
        """
        if self.dataset_manager is None:
            return EvalResult()

        if categories:
            all_problems = []
            for cat in categories:
                all_problems.extend(self.dataset_manager.load_l2_data(category=cat))
        else:
            all_problems = self.dataset_manager.load_l2_data()

        return self.run_eval(all_problems, "math")

    def compare_models(
        self, model_list: List[Any], test_problems: List[Any]
    ) -> Dict[str, EvalResult]:
        """在相同问题上对比多个模型。

        Args:
            model_list: 模型/Agent 列表
            test_problems: 测试问题列表

        Returns:
            模型名 → EvalResult 的映射
        """
        results: Dict[str, EvalResult] = {}
        for i, model in enumerate(model_list):
            self.agent = model
            result = self.run_eval(test_problems, f"model_{i}")
            results[f"model_{i}"] = result
        return results

    def _classify_error(self, trajectory: Any) -> str:
        """对错误的轨迹进行错误分类。

        Returns:
            错误类型: reasoning_error / tool_error / format_error /
                      answer_extraction_error / capability_boundary
        """
        # 简化的分类逻辑
        if not trajectory.final_answer:
            return "format_error"
        if any(s.observation and "Error:" in s.observation for s in trajectory.steps):
            return "tool_error"
        return "reasoning_error"
