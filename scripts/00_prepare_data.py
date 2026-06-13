#!/usr/bin/env python
"""Phase 0 / Phase 1 前置：数据准备脚本。

功能：
1. 下载 GSM8K, MATH 数据集
2. 创建 train/val 分割
3. 生成 SFT 训练数据（JSONL 格式）
"""

import sys
from pathlib import Path

# 将项目根目录加入 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.parser import ConfigParser
from data.dataset import DatasetManager
from data.sft_data_gen import SFTDataGenerator
from utils.logger import get_logger
from utils.helpers import Timer

logger = get_logger(__name__)


def main():
    """数据准备主流程。"""
    logger.info("=== Phase 0: 数据准备开始 ===")

    # 1. 加载配置
    config = ConfigParser()
    logger.info(f"配置加载完成")

    # 2. 加载数据集
    dm = DatasetManager(data_config=config.data)

    with Timer("加载数据"):
        train_data, val_data = dm.load_all()
        logger.info(f"训练集: {len(train_data)} 条")
        logger.info(f"验证集: {len(val_data)} 条")

    # 3. 生成 SFT 训练数据
    sft_gen = SFTDataGenerator(output_dir="data/sft")
    l1_data = dm.load_l1_data()

    with Timer("生成 SFT 数据"):
        sft_data = sft_gen.generate_from_templates(
            problems=l1_data[: config.data.get("sft_examples", 500)],
        )
        logger.info(f"SFT 数据生成完成: {len(sft_data)} 条")

    logger.info("=== 数据准备完成 ===")


if __name__ == "__main__":
    main()
