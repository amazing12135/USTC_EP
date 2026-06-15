"""GRPO 训练器：实现 Group Relative Policy Optimization。

GRPO Loss 公式（对于每个 group G，同一道题的 group_size 条轨迹）：

  A_i = (R_i - mean(R_group)) / std(R_group)    ← Group Relative Advantage

  L = - mean_i [ min( ratio_i * A_i,
                      clip(ratio_i, 1-ε, 1+ε) * A_i )
                  - β * KL(π_θ || π_ref) ]

  ratio_i = exp( log π_θ(a_i|s_i) - log π_old(a_i|s_i) )
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class GRPOTrainer:
    """GRPO 训练器。

    用法:
        trainer = GRPOTrainer(policy_model, ref_model, sampler, reward_fn, tokenizer)
        for step in range(max_steps):
            batch = random.sample(train_data, batch_size)
            metrics = trainer.train_step(batch)
            trainer.maybe_log_wandb(metrics)     # 每 N 步
            trainer.maybe_save_checkpoint(step)  # 每 N 步
    """

    def __init__(
        self,
        policy_model: Any,       # PeftModel (当前策略，trainable LoRA)
        ref_model: Any,          # PeftModel (参考策略，frozen)
        tokenizer: Any,          # tokenizer（用于 tokenize trajectory 文本）
        sampler: Any = None,     # TrajectorySampler
        reward_fn: Any = None,   # RewardFunction
        optimizer: Optional[torch.optim.Optimizer] = None,
        kl_coef: float = 0.04,
        clip_eps: float = 0.2,
        gamma: float = 1.0,
        max_grad_norm: float = 1.0,
        learning_rate: float = 2e-6,
        max_seq_length: int = 2048,
        checkpoint_dir: str | Path = "checkpoints/grpo",
    ) -> None:
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.sampler = sampler
        self.reward_fn = reward_fn
        self.kl_coef = kl_coef
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.max_seq_length = max_seq_length
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 优化器：如果未传入，自动创建
        if optimizer is None and policy_model is not None:
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, policy_model.parameters()),
                lr=learning_rate,
            )
        else:
            self.optimizer = optimizer

        # 训练状态
        self.global_step: int = 0
        self.metrics_history: List[Dict[str, float]] = []

    # ================================================================
    # 单步训练
    # ================================================================

    def train_step(
        self,
        batch_data: List[Any],  # List[Problem]
    ) -> Dict[str, float]:
        """执行一步 GRPO 训练。

        流程: 采样 → reward → advantage → 计算 loss → backward → step
        """
        if self.sampler is None:
            return {}

        # 1. 采样轨迹
        trajectories, group_ids, rewards = self.sampler.sample_grpo_batch(batch_data)

        # 2. 计算 GRPO loss
        loss, aux = self.compute_loss(trajectories, group_ids, rewards)

        # 3. 梯度更新
        if loss is not None and self.optimizer is not None:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.policy_model.parameters()),
                self.max_grad_norm,
            )
            self.optimizer.step()

        self.global_step += 1

        # 4. 记录指标
        metrics: Dict[str, float] = {
            "step": self.global_step,
            "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
            "reward_std": float(np.std(rewards)) if rewards else 0.0,
            "loss": float(loss) if loss is not None else 0.0,
        }
        if aux:
            metrics.update({k: float(v) for k, v in aux.items()})
        self.metrics_history.append(metrics)

        return metrics

    # ================================================================
    # Loss 计算（核心）
    # ================================================================

    def compute_loss(
        self,
        trajectories: List[Any],   # List[AgentTrajectory]
        group_ids: List[int],
        rewards: List[float],
    ) -> tuple:
        """计算 GRPO 损失。

        对于每条轨迹：
        1. 提取 assistant 生成的 token 序列
        2. 用 policy_model 前向计算 π_θ logprobs
        3. 用 ref_model 前向计算 π_ref logprobs
        4. 用 vLLM 采样的 π_old logprobs
        5. ratio = exp(π_θ - π_old)
        6. A = (r - mean(r_group)) / std(r_group)
        7. policy_loss = -min(ratio*A, clip(ratio)*A)
        8. kl = KL(π_θ || π_ref)
        9. total = policy_loss + β * kl

        Args:
            trajectories: 推理轨迹列表
            group_ids: 每条轨迹的 group ID
            rewards: 每条轨迹的标量奖励

        Returns:
            (loss_tensor, aux_metrics_dict)
        """
        import torch.nn.functional as F

        # ---- Step 1: 计算 advantages ----
        from .advantage import AdvantageEstimator
        adv_estimator = AdvantageEstimator()
        advantages_list = adv_estimator.compute_group_relative(rewards, group_ids)

        # ---- Step 2: 对每条轨迹提取 assistant tokens 和 logprobs ----
        all_ratios = []
        all_kls = []
        all_advantages = []

        for traj, advantage in zip(trajectories, advantages_list):
            # 拼接轨迹中所有 assistant 生成的文本
            assistant_text = self._build_trajectory_text(traj)
            if not assistant_text:
                continue

            # Tokenize
            enc = self.tokenizer(
                assistant_text, return_tensors="pt", truncation=True,
                max_length=self.max_seq_length, padding=False,
            )
            input_ids = enc["input_ids"].to(self.policy_model.device)
            attention_mask = enc["attention_mask"].to(self.policy_model.device)

            if input_ids.numel() == 0:
                continue

            # ---- 获取旧策略 logprobs（来自 vLLM 采样） ----
            old_logprobs = self._gather_old_logprobs(traj, input_ids)

            # ---- 新策略 logprobs ----
            with torch.no_grad():
                outputs_policy = self.policy_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            # 当前策略的 logprobs
            logprobs_policy = self._compute_token_logprobs(
                outputs_policy.logits, input_ids
            )

            # ---- 参考策略 logprobs ----
            with torch.no_grad():
                outputs_ref = self.ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            logprobs_ref = self._compute_token_logprobs(
                outputs_ref.logits, input_ids
            )

            # ---- ratio = exp(π_θ - π_old) ----
            if old_logprobs is not None and old_logprobs.numel() > 0:
                # 对齐长度
                min_len = min(logprobs_policy.shape[0],
                              old_logprobs.shape[0],
                              logprobs_ref.shape[0])
                lp = logprobs_policy[:min_len]
                lo = old_logprobs[:min_len]
                lr = logprobs_ref[:min_len]

                ratio = torch.exp(lp - lo)
                kl = self._compute_kl(lp, lr)

                # 按 token 平均
                all_ratios.append(ratio.mean())
                all_kls.append(kl.mean())
                all_advantages.append(advantage)

        if not all_ratios:
            return None, {}

        # ---- Step 3: PPO-style clipped loss ----
        ratios_t = torch.stack(all_ratios)          # (n_valid_trajs,)
        advantages_t = torch.tensor(all_advantages, device=ratios_t.device,
                                    dtype=ratios_t.dtype)
        kls_t = torch.stack(all_kls)

        # clipped surrogate
        clipped_ratios = torch.clamp(ratios_t, 1 - self.clip_eps, 1 + self.clip_eps)
        policy_loss = -torch.min(ratios_t * advantages_t,
                                 clipped_ratios * advantages_t).mean()

        # KL penalty（均值，乘以 β）
        kl_loss = kls_t.mean()
        total_loss = policy_loss + self.kl_coef * kl_loss

        aux = {
            "policy_loss": policy_loss.detach(),
            "kl_div": kl_loss.detach(),
            "policy_ratio_mean": ratios_t.mean().detach(),
            "n_valid_trajs": len(all_ratios),
        }
        return total_loss, aux

    def _build_trajectory_text(self, traj: Any) -> str:
        """拼接轨迹中 assistant 生成的所有文本（不含 system/user 提示）。

        只取 assistant 的 thought 和 action，不包含 tool 返回的 observation。
        """
        parts = []
        for step in traj.steps:
            parts.append(step.thought)
        return "\n".join(parts)

    def _gather_old_logprobs(
        self, traj: Any, input_ids: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """从轨迹的 steps 中收集 vLLM 采样时的旧策略 logprobs。

        如果 trajectory steps 中存有 logprobs，直接使用；
        否则返回 None（e.g. 用于第一次采样时 π_old == π_θ）。
        """
        all_lps = []
        for step in traj.steps:
            step_lps = getattr(step, "logprobs", None)
            if step_lps:
                all_lps.extend(step_lps)

        if all_lps:
            return torch.tensor(all_lps, device=input_ids.device, dtype=torch.float32)
        return None

    # ================================================================
    # Token-level 工具
    # ================================================================

    @staticmethod
    def _compute_token_logprobs(
        logits: torch.Tensor,    # (1, seq_len, vocab_size)
        input_ids: torch.Tensor, # (1, seq_len)
    ) -> torch.Tensor:
        """计算每个 token 的对数概率（shifted，对齐预测位置）。

        logits[t] 预测的是 token[t+1]，所以需要 shift：
        logprob[t] = log_softmax(logits[t])[input_ids[t+1]]
        """
        import torch.nn.functional as F
        # shift: logits[0:T-1] → 预测 input_ids[1:T]
        shift_logits = logits[0, :-1, :]           # (T-1, V)
        shift_labels = input_ids[0, 1:]             # (T-1,)
        logprobs = F.log_softmax(shift_logits, dim=-1)
        token_logprobs = logprobs.gather(
            1, shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        return token_logprobs  # (T-1,)

    @staticmethod
    def _compute_kl(
        pi_logprobs: torch.Tensor,   # π_θ
        ref_logprobs: torch.Tensor,  # π_ref
    ) -> torch.Tensor:
        """Kullback-Leibler 散度 KL(π_θ || π_ref)。

        KL = exp(log_ref - log_pi) - (log_ref - log_pi) + 1
        在正态近似或小差值下近似为: mean(ref_logprobs - pi_logprobs)
        这里使用精确公式。
        """
        log_ratio = ref_logprobs - pi_logprobs  # log(π_ref / π_θ)
        # KL = π_θ * log(π_θ / π_ref) ≈ exp(log π_θ - log π_ref) - (log π_θ - log π_ref) + 1
        kl_per_token = torch.exp(-log_ratio) + log_ratio - 1
        return kl_per_token

    # ================================================================
    # Checkpoint & Logging
    # ================================================================

    def save_checkpoint(self, path: Optional[str | Path] = None) -> Path:
        """保存 LoRA adapter checkpoint。"""
        save_path = Path(path) if path else (
            self.checkpoint_dir / f"step_{self.global_step:05d}"
        )
        save_path.mkdir(parents=True, exist_ok=True)
        self.policy_model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))

        # 保存训练状态
        state = {
            "global_step": self.global_step,
            "metrics_history": self.metrics_history[-100:],  # 最近 100 步
        }
        with open(save_path / "train_state.json", "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.info(f"Checkpoint saved: {save_path}")
        return save_path

    def maybe_log_wandb(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        """将指标上报 Wandb（如果已初始化）。"""
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics, step=step or self.global_step)
        except (ImportError, AttributeError):
            pass

    def get_metrics(self) -> Dict[str, Any]:
        if not self.metrics_history:
            return {}
        return self.metrics_history[-1]
