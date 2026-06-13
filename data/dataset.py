"""数据集管理：加载和管理 GSM8K, MATH, AMC 数据集。"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Problem:
    """单条数学问题。"""

    id: str
    question: str
    answer: str  # 标准答案
    level: str  # "L1" or "L2"
    category: str = ""  # algebra / counting / ...
    source: str = ""  # "gsm8k" / "math" / "amc"


class DatasetManager:
    """数据集管理器。

    支持的数据集：
    - L1: GSM8K 子集 (~1K) — 仅用于 SFT 阶段
    - L2: MATH (~7.5K) + AMC (~2-3K) — RL 训练主战场
    """

    def __init__(self, data_config: Optional[Dict] = None) -> None:
        """初始化数据集管理器。

        Args:
            data_config: 数据配置字典（来自 config.yaml data 段）
        """
        self.config = data_config or {}
        self._train_data: List[Problem] = []
        self._val_data: List[Problem] = []

    def load_l1_data(self) -> List[Problem]:
        """加载 L1 数据：GSM8K 子集。

        选取难度偏上的题目（多步推理、含分数/小数），
        来源: HuggingFace datasets "openai/gsm8k"

        Returns:
            Problem 列表
        """
        problems: List[Problem] = []
        try:
            from datasets import load_dataset

            ds = load_dataset(
                self.config.get("gsm8k_path", "openai/gsm8k"),
                "main",
                split="train",
            )

            for i, item in enumerate(ds):  # type: ignore
                l1_size = self.config.get("l1_size", 1000)
                if i >= l1_size:
                    break
                problems.append(
                    Problem(
                        id=f"gsm8k_{i:04d}",
                        question=item["question"],
                        answer=item["answer"],
                        level="L1",
                        category="arithmetic",
                        source="gsm8k",
                    )
                )
        except Exception:
            pass

        return problems

    def load_l2_data(self, category: Optional[str] = None) -> List[Problem]:
        """加载 L2 数据：MATH + AMC。

        加载 MATH 竞赛数学数据集，按子类筛选。
        来源: HuggingFace datasets "hendrycks/math"

        Args:
            category: 可选，只加载指定子类（algebra/counting/number_theory/...）

        Returns:
            Problem 列表
        """
        problems: List[Problem] = []
        try:
            from datasets import load_dataset

            ds = load_dataset(
                self.config.get("math_path", "hendrycks/math"),
                split="train",
            )

            idx = 0
            for item in ds:  # type: ignore
                if category and item.get("type", "") != category:
                    continue
                l2_size = self.config.get("l2_train_size", 9000)
                if idx >= l2_size:
                    break
                problems.append(
                    Problem(
                        id=f"math_{item.get('type', 'unknown')}_{idx:04d}",
                        question=item["problem"],
                        answer=item["solution"],
                        level="L2",
                        category=item.get("type", ""),
                        source="math",
                    )
                )
                idx += 1
        except Exception:
            pass

        return problems

    def load_all(self) -> Tuple[List[Problem], List[Problem]]:
        """加载全部数据并创建 train/val 分割。

        Returns:
            (train_data, val_data)
        """
        l1 = self.load_l1_data()
        l2 = self.load_l2_data()

        all_data = l1 + l2

        train_ratio = self.config.get("train_ratio", 0.9)
        split_idx = int(len(all_data) * train_ratio)

        self._train_data = all_data[:split_idx]
        self._val_data = all_data[split_idx:]

        return self._train_data, self._val_data

    def create_splits(
        self, train_ratio: float = 0.9
    ) -> Tuple[List[Problem], List[Problem]]:
        """创建训练/验证集分割。

        Args:
            train_ratio: 训练集比例

        Returns:
            (train_data, val_data)
        """
        if not self._train_data:
            return self.load_all()
        return self._train_data, self._val_data

    @property
    def train_data(self) -> List[Problem]:
        """训练集数据。"""
        if not self._train_data:
            self.load_all()
        return self._train_data

    @property
    def val_data(self) -> List[Problem]:
        """验证集数据。"""
        if not self._val_data:
            self.load_all()
        return self._val_data
