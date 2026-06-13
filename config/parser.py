"""配置解析器：加载和访问 config.yaml 配置。"""

from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigParser:
    """解析并管理项目配置文件。"""

    def __init__(self, config_path: str | Path | None = None) -> None:
        """初始化配置解析器。

        Args:
            config_path: 配置文件路径，默认为本模块同目录下的 config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """从 YAML 文件加载配置。"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """用点号分隔的 key 获取配置值（如 'model.name'）。

        Args:
            key: 点号分隔的配置键
            default: 键不存在时返回的默认值

        Returns:
            配置值
        """
        keys = key.split(".")
        value: Any = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    @property
    def model(self) -> Dict[str, Any]:
        return self._config.get("model", {})

    @property
    def lora(self) -> Dict[str, Any]:
        return self._config.get("lora", {})

    @property
    def data(self) -> Dict[str, Any]:
        return self._config.get("data", {})

    @property
    def agent(self) -> Dict[str, Any]:
        return self._config.get("agent", {})

    @property
    def rl(self) -> Dict[str, Any]:
        return self._config.get("rl", {})

    @property
    def reward(self) -> Dict[str, Any]:
        return self._config.get("reward", {})

    @property
    def training(self) -> Dict[str, Any]:
        return self._config.get("training", {})

    @property
    def inference(self) -> Dict[str, Any]:
        return self._config.get("inference", {})
