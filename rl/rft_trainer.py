"""RFT (Rejection Fine-Tuning) 训练器：baseline 方法。

流程：
1. 用当前模型对 L2 训练集采样（每个问题 1 条轨迹）
2. 筛选出答案正确的轨迹
3. 用正确轨迹做 SFT 微调（ChatML 格式）
4. 可选：多轮迭代
5. 每轮保存轨迹到 MD 记忆库

对比：
- 优势：简单稳定，可作为 GRPO 前的基础
- 劣势：只利用正样本，负样本信息被丢弃
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RFTTrainer:
    """RFT 训练器。

    用法:
        trainer = RFTTrainer(agent=agent, model=peft_model, tokenizer=tokenizer)
        history = trainer.iterate(l2_dataset, n_rounds=2)
    """

    def __init__(
        self,
        agent: Any,          # ReActAgent
        model: Any,          # PeftModel (带 LoRA，可训练)
        tokenizer: Any,      # 模型的 tokenizer
        reward_fn: Optional[Any] = None,
        temperature: float = 0.8,
        filter_ratio: float = 0.5,
        epochs: int = 2,
        rounds: int = 2,
        output_dir: str | Path = "checkpoints/rft",
        memory_dir: str | Path = "data/memory_store",
    ) -> None:
        self.agent = agent
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.temperature = temperature
        self.filter_ratio = filter_ratio
        self.epochs = epochs
        self.rounds = rounds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir = Path(memory_dir)

    # ================================================================
    # 采样 & 筛选
    # ================================================================

    def sample_all(self, dataset: List[Any]) -> List[Any]:
        """对全部数据集采样（每题 1 条轨迹）。

        Args:
            dataset: Problem 列表

        Returns:
            AgentTrajectory 列表
        """
        trajectories = []
        n = len(dataset)
        for i, problem in enumerate(dataset):
            traj = self.agent.run(
                question=problem.question,
                ground_truth=problem.answer,
            )
            trajectories.append(traj)
            if (i + 1) % 50 == 0:
                logger.info(f"  采样进度: {i+1}/{n}")
        return trajectories

    def filter_correct(self, trajectories: List[Any]) -> List[Any]:
        """筛选出答案正确的轨迹。

        Args:
            trajectories: AgentTrajectory 列表

        Returns:
            正确轨迹列表
        """
        return [t for t in trajectories if t.is_correct]

    # ================================================================
    # SFT 微调
    # ================================================================

    def sft_finetune(
        self,
        correct_trajs: List[Any],
        epochs: Optional[int] = None,
    ) -> None:
        """用正确轨迹做 SFT 微调。

        将轨迹转换为 ChatML 格式，调用 TRL SFTTrainer 进行微调。

        Args:
            correct_trajs: 正确答案的轨迹列表
            epochs: 训练的 epoch 数
        """
        if not correct_trajs:
            logger.warning("无正确轨迹，跳过微调")
            return

        epochs = epochs or self.epochs
        logger.info(f"RFT 微调: {len(correct_trajs)} 条正确轨迹, {epochs} epochs")

        # 将轨迹转换为 ChatML messages 格式
        train_records = []
        system_prompt = self._load_system_prompt()
        for traj in correct_trajs:
            messages = self._trajectory_to_messages(traj, system_prompt)
            train_records.append({"messages": messages})

        # 构建 Dataset
        from datasets import Dataset
        dataset = Dataset.from_list(train_records)

        # 训练参数
        from transformers import TrainingArguments

        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "rft_temp"),
            num_train_epochs=epochs,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,       # RFT 比初始 SFT 学习率低
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            weight_decay=0.01,
            bf16=True,
            fp16=False,
            logging_steps=5,
            save_strategy="no",       # iterate 最后统一保存
            eval_strategy="no",
            remove_unused_columns=True,
            report_to=["none"],
        )

        from trl import SFTTrainer

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            formatting_func=lambda ex: self.tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False,
            ),
            max_seq_length=2048,
            packing=False,
        )

        trainer.train()
        logger.info(f"  微调完成: loss={trainer.state.log_history[-1].get('loss', 'N/A')}")

    def _trajectory_to_messages(
        self, traj: Any, system_prompt: str
    ) -> List[Dict[str, str]]:
        """将一条 AgentTrajectory 转换为 ChatML messages。

        Args:
            traj: AgentTrajectory
            system_prompt: 系统提示词

        Returns:
            ChatML 消息列表
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"问题：{traj.question}"},
        ]

        for step in traj.steps:
            content = step.thought
            if step.action is not None:
                # 工具调用
                import json
                action_json = json.dumps(
                    {"name": step.action.name, "args": step.action.args},
                    ensure_ascii=False,
                )
                content = f"{content}\n\n<tool_call>{action_json}</tool_call>"
            messages.append({"role": "assistant", "content": content})

            if step.observation:
                messages.append({
                    "role": "tool",
                    "content": f"Observed: {step.observation}",
                })

        return messages

    # ================================================================
    # 多轮迭代
    # ================================================================

    def iterate(
        self,
        dataset: List[Any],
        n_rounds: Optional[int] = None,
        val_dataset: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """执行多轮 RFT 迭代。

        Args:
            dataset: L2 训练数据集
            n_rounds: 迭代轮数
            val_dataset: 验证集（可选，每轮后评估）

        Returns:
            {
                "rounds": [...],
                "train_correct": [...],
                "train_total": [...],
                "val_accuracy": [...],       # 仅当提供了 val_dataset
                "checkpoint_dirs": [...],
            }
        """
        n_rounds = n_rounds or self.rounds
        history: Dict[str, Any] = {
            "rounds": [],
            "train_correct": [],
            "train_total": [],
            "val_accuracy": [],
            "checkpoint_dirs": [],
        }

        for r in range(1, n_rounds + 1):
            logger.info(f"\n{'='*40}\n  RFT Round {r}/{n_rounds}\n{'='*40}")

            # 1. 采样
            logger.info(f"采样 {len(dataset)} 题 (temperature={self.temperature})...")
            trajectories = self.sample_all(dataset)

            # 2. 筛选
            correct = self.filter_correct(trajectories)
            acc = len(correct) / len(trajectories) if trajectories else 0.0
            logger.info(f"  正确: {len(correct)}/{len(trajectories)} = {acc:.1%}")

            # 3. 保存轨迹到 MD 记忆库
            self._save_trajectories_to_memory(trajectories, r)

            # 4. 微调
            self.sft_finetune(correct, epochs=self.epochs)

            # 5. 验证集评估
            val_acc = None
            if val_dataset:
                val_acc = self._evaluate(val_dataset)
                logger.info(f"  验证集准确率: {val_acc:.1%}")
                self._save_trajectories_to_memory(
                    self.sample_all(val_dataset), r, prefix="val"
                )

            # 6. 保存 checkpoint
            ckpt_dir = self.output_dir / f"round_{r}"
            self.model.save_pretrained(str(ckpt_dir))
            self.tokenizer.save_pretrained(str(ckpt_dir))
            logger.info(f"  Checkpoint: {ckpt_dir}")

            history["rounds"].append(r)
            history["train_correct"].append(len(correct))
            history["train_total"].append(len(trajectories))
            history["val_accuracy"].append(val_acc)
            history["checkpoint_dirs"].append(str(ckpt_dir))

        return history

    # ================================================================
    # 辅助
    # ================================================================

    def _evaluate(self, val_dataset: List[Any]) -> float:
        """在验证集上评估当前模型准确率。"""
        trajectories = self.sample_all(val_dataset)
        correct = self.filter_correct(trajectories)
        return len(correct) / len(trajectories) if trajectories else 0.0

    def _save_trajectories_to_memory(
        self,
        trajectories: List[Any],
        round_num: int,
        prefix: str = "train",
    ) -> None:
        """将轨迹保存为 MD 文件。

        按来源/子类分目录：{memory_dir}/{prefix}/round_{round_num}/{status}/
        """
        import datetime

        base = self.memory_dir / prefix / f"round_{round_num:02d}"
        success_dir = base / "success"
        failure_dir = base / "failure"
        success_dir.mkdir(parents=True, exist_ok=True)
        failure_dir.mkdir(parents=True, exist_ok=True)

        for i, traj in enumerate(trajectories):
            status = "success" if traj.is_correct else "failure"
            target_dir = success_dir if traj.is_correct else failure_dir
            date_str = datetime.date.today().isoformat()

            md_content = self._format_trajectory_md(traj, status, date_str, i)
            file_path = target_dir / f"{i:04d}_{status}.md"
            file_path.write_text(md_content, encoding="utf-8")

        logger.info(f"  轨迹已保存: {base}")

    def _format_trajectory_md(
        self, traj: Any, status: str, date_str: str, idx: int
    ) -> str:
        """将轨迹格式化为 MD 内容。"""
        # 推断 category（从 question ID 或其他元信息）
        lines = [
            f"# Problem ID: rft_{idx:04d}",
            f"# Date: {date_str}",
            f"# Status: {status}",
            "# Tags: []",
            "",
            "## Question",
            traj.question,
            "",
            "## Ground Truth",
            traj.ground_truth,
            "",
            "## Agent Trajectory",
        ]

        for step in traj.steps:
            lines.append(f"### Step {step.turn}")
            lines.append(f"- **Thought**: {step.thought[:300]}")
            if step.action is not None:
                import json
                action_json = json.dumps(
                    {"name": step.action.name, "args": step.action.args},
                    ensure_ascii=False,
                )
                lines.append(f"- **Action**: `<tool_call>{action_json}</tool_call>`")
            if step.observation:
                lines.append(f"- **Observation**: {step.observation[:300]}")
            lines.append("")

        lines.append("## Final Answer")
        lines.append(f"<answer>{traj.final_answer}</answer>")
        lines.append("")
        lines.append("## Performance")
        lines.append(f"- Total Turns: {traj.total_turns}")
        tool_count = sum(1 for s in traj.steps if s.action is not None)
        lines.append(f"- Tool Calls: {tool_count}")
        lines.append(f"- Correct: {traj.is_correct}")
        lines.append("")

        if not traj.is_correct:
            lines.append("## Reflection")
            lines.append("- 错误类型: 待分析")
            lines.append("- 原因分析: 待分析")

        return "\n".join(lines)

    @staticmethod
    def _load_system_prompt() -> str:
        """加载系统提示词。"""
        prompt_path = Path("config/prompts/system.txt")
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        return """你是一个数学解题助手。你可以使用以下工具：

1. calculator - 计算数学表达式
   参数: {"expression": "数学表达式"}

2. python_exec - 执行Python代码
   参数: {"code": "Python代码"}

3. sympy - 符号数学计算
   参数: {"code": "SymPy代码"}

请使用ReAct格式推理，用 <tool_call>...</tool_call> 调用工具，用 <answer>答案</answer> 输出结果。"""
