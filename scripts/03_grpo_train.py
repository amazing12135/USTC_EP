#!/usr/bin/env python
"""Phase 3：GRPO 训练脚本。

功能：
1. 初始化 GRPO 训练器（policy_model + ref_model + vLLM sampler）
2. 运行 GRPO 训练循环（采样 → reward → advantage → 策略更新）
3. Wandb 监控
4. 定期保存 checkpoint 并在验证集上评估
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.parser import ConfigParser
from utils.logger import get_logger
from utils.helpers import set_seed, Timer

logger = get_logger(__name__)


def main():
    """GRPO 训练主流程。"""
    logger.info("=== Phase 3: GRPO 训练开始 ===")

    # 1. 加载配置
    config = ConfigParser()
    set_seed(config.training.get("seed", 42))
    rl_config = config.rl

    # 2. 初始化
    logger.info(f"算法: {rl_config.get('algorithm', 'grpo')}")
    logger.info(f"batch_size: {rl_config.get('batch_size', 16)}")
    logger.info(f"group_size: {rl_config.get('group_size', 8)}")
    logger.info(f"kl_coef: {rl_config.get('kl_coef', 0.04)}")
    logger.info(f"learning_rate: {rl_config.get('learning_rate', 2e-6)}")
    logger.info(f"max_steps: {rl_config.get('max_steps', 500)}")

    # 3. 加载模型
    logger.info(f"加载基座模型: {config.model.get('name')}")
    # TODO: 需要 GPU 环境
    # from model.loader import ModelLoader
    # from model.inference import BatchInference
    # from agent.react_agent import ReActAgent
    # from rl.sampler import TrajectorySampler
    # from rl.reward import RewardFunction
    # from rl.grpo_trainer import GRPOTrainer
    # from data.dataset import DatasetManager
    #
    # loader = ModelLoader(model_name=config.model["name"])
    # policy_model = loader.load_for_training()
    # ref_model = loader.load_for_training()  # 冻结的参考模型
    #
    # # vLLM 推理引擎
    # inference = BatchInference(model_name_or_path=config.model["name"])
    #
    # # Agent + Sampler
    # agent = ReActAgent(model=inference, ...)
    # sampler = TrajectorySampler(
    #     agent=agent,
    #     batch_size=rl_config.get("batch_size", 16),
    #     group_size=rl_config.get("group_size", 8),
    # )
    #
    # # 初始化 Wandb
    # import wandb
    # wandb.init(project=config.training.get("wandb_project", "llm-math-agent"))
    #
    # # GRPO Trainer
    # trainer = GRPOTrainer(
    #     policy_model=policy_model,
    #     ref_model=ref_model,
    #     sampler=sampler,
    #     reward_fn=RewardFunction(),
    #     kl_coef=rl_config.get("kl_coef", 0.04),
    #     clip_eps=rl_config.get("clip_eps", 0.2),
    # )
    #
    # # 训练循环
    # dm = DatasetManager(data_config=config.data)
    # train_data = dm.load_l2_data()
    #
    # for step in range(rl_config.get("max_steps", 500)):
    #     metrics = trainer.train_step(train_data)
    #     if step % rl_config.get("eval_every", 50) == 0:
    #         logger.info(f"Step {step}: reward_mean={metrics['reward_mean']:.3f}")
    #         wandb.log(metrics)
    #     if step % rl_config.get("save_every", 100) == 0:
    #         trainer.save_checkpoint()

    logger.info("=== GRPO 训练完成（当前为骨架代码，需在 GPU 环境运行）===")


if __name__ == "__main__":
    main()
