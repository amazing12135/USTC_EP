#!/usr/bin/env python
"""Phase 2：RFT Baseline 训练脚本。

功能：
1. 用当前模型对训练集采样
2. 筛选正确轨迹
3. 用正确轨迹做 SFT 微调
4. 评估并记录轨迹到 MD 记忆库
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.parser import ConfigParser
from utils.logger import get_logger
from utils.helpers import set_seed, Timer

logger = get_logger(__name__)


def main():
    """RFT 训练主流程。"""
    logger.info("=== Phase 2: RFT Baseline 训练开始 ===")

    # 1. 加载配置
    config = ConfigParser()
    set_seed(config.training.get("seed", 42))
    rl_config = config.rl

    # 2. 加载模型和 Agent
    logger.info(f"加载模型: {config.model.get('name')}")
    # TODO: 需要 GPU 环境
    # from model.loader import ModelLoader
    # from agent.react_agent import ReActAgent
    # from data.dataset import DatasetManager
    # from rl.rft_trainer import RFTTrainer
    #
    # loader = ModelLoader(model_name=config.model["name"])
    # model = loader.load_for_inference()
    # agent = ReActAgent(model=model, ...)
    # dm = DatasetManager(data_config=config.data)
    # train_data = dm.load_l2_data()
    #
    # trainer = RFTTrainer(
    #     agent=agent,
    #     model=model,
    #     temperature=rl_config.get("rft_temperature", 0.8),
    #     filter_ratio=rl_config.get("rft_filter_ratio", 0.5),
    #     epochs=rl_config.get("rft_epochs", 2),
    #     rounds=rl_config.get("rft_rounds", 2),
    # )
    #
    # history = trainer.iterate(train_data)
    # logger.info(f"RFT 完成，最终准确率: {history['accuracy'][-1]:.3f}")

    logger.info("=== RFT 训练完成（当前为骨架代码，需在 GPU 环境运行）===")


if __name__ == "__main__":
    main()
