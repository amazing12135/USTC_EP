"""通用辅助函数。"""

import random
import time
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """设置所有随机种子以确保可复现性。

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except (ImportError, OSError):
        pass


def format_duration(seconds: float) -> str:
    """将秒数格式化为可读的时间字符串。

    Args:
        seconds: 秒数

    Returns:
        格式化的时间字符串，如 "2h 30m 15s"
    """
    if seconds < 0:
        return "0s"

    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")

    return " ".join(parts)


class Timer:
    """简单的计时器上下文管理器。"""

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        self.elapsed = time.perf_counter() - self.start_time
        if self.label:
            print(f"[{self.label}] {format_duration(self.elapsed)}")
