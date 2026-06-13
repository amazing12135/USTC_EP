"""RFT (Rejection Fine-Tuning) 训练器：baseline 方法。

流程：
1. 用当前模型对训练集采样（每个问题 1 条轨迹）
2. 筛选出正确的轨迹
3. 用正确轨迹做 SFT 微调
4. 可选：多轮迭代
"""

from pathlib import Path
from typing import Any, Dict, List, Optional


class RFTTrainer:
    """RFT 训练器。

    优势：简单稳定，可作为 GRPO 前的基础
    劣势：只利用正样本，不利用负样本的信息
    """

    def __init__(
        self,
        agent: Any,  # ReActAgent
        model: Any,  # 用于 SFT 微调的模型
        reward_fn: Optional[Any] = None,
        temperature: float = 0.8,
        filter_ratio: float = 0.5,
        epochs: int = 2,
        rounds: int = 2,
        output_dir: str | Path = "checkpoints/rft",
    ) -> None:
        """初始化 RFT 训练器。

        Args:
            agent: ReActAgent 实例
            model: 可微调的模型实例
            reward_fn: 奖励函数
            temperature: 采样温度
            filter_ratio: 保留正确率 top 的比例
            epochs: 每轮 SFT 的 epoch 数
            rounds: RFT 迭代轮数
            output_dir: checkpoint 保存目录
        """
        self.agent = agent
        self.model = model
        self.reward_fn = reward_fn
        self.temperature = temperature
        self.filter_ratio = filter_ratio
        self.epochs = epochs
        self.rounds = rounds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def sample_all(
        self,
        dataset: List[Any],  # List[Problem]
    ) -> List[Any]:
        """用当前模型对全部数据集采样（每个问题 1 条轨迹）。

        Args:
            dataset: 问题列表

        Returns:
            AgentTrajectory 列表
        """
        trajectories = []
        for problem in dataset:
            traj = self.agent.run(
                question=problem.question,
                ground_truth=problem.answer,
            )
            trajectories.append(traj)
        return trajectories

    def filter_correct(
        self,
        trajectories: List[Any],
    ) -> List[Any]:
        """筛选出答案正确的轨迹。

        Args:
            trajectories: AgentTrajectory 列表

        Returns:
            正确轨迹列表
        """
        return [t for t in trajectories if t.is_correct]

    def sft_finetune(
        self,
        correct_trajs: List[Any],
        epochs: Optional[int] = None,
    ) -> None:
        """用正确轨迹做 SFT 微调。

        Args:
            correct_trajs: 正确答案的轨迹列表
            epochs: 训练的 epoch 数
        """
        if not correct_trajs:
            return

        epochs = epochs or self.epochs

        # 将轨迹转换为 SFT 训练格式
        # 格式: (prompt, completion) 或 ChatML messages
        train_data = []
        for traj in correct_trajs:
            prompt = f"问题：{traj.question}"
            # 拼接完整的 assistant 响应（所有步骤的 thought + action）
            completion_parts = []
            for step in traj.steps:
                completion_parts.append(step.thought)
            completion = "\n\n".join(completion_parts)
            train_data.append((prompt, completion))

        # TODO: 调用 SFTTrainer 进行微调
        # 在云端 GPU 环境中，这里会调用:
        # from trl import SFTTrainer
        # trainer = SFTTrainer(
        #     model=self.model,
        #     train_dataset=train_data,
        #     args=...
        # )
        # trainer.train()

    def iterate(
        self,
        dataset: List[Any],
        n_rounds: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """执行多轮 RFT 迭代。

        Args:
            dataset: 训练数据集
            n_rounds: 迭代轮数

        Returns:
            每轮的指标记录
        """
        n_rounds = n_rounds or self.rounds
        history: Dict[str, List[float]] = {
            "round": [],
            "correct_count": [],
            "total_count": [],
            "accuracy": [],
        }

        for r in range(n_rounds):
            trajectories = self.sample_all(dataset)
            correct = self.filter_correct(trajectories)

            acc = len(correct) / len(trajectories) if trajectories else 0.0
            history["round"].append(r + 1)
            history["correct_count"].append(len(correct))
            history["total_count"].append(len(trajectories))
            history["accuracy"].append(acc)

            # 用正确轨迹微调
            self.sft_finetune(correct)

        return history
