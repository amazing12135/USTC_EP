"""工具注册中心：管理可用工具的注册、查找和模式描述。"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Tool:
    """单个工具的定义。"""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema 格式的参数定义
    func: Callable[..., str]
    timeout: int = 5  # 执行超时（秒）

    def get_schema(self) -> Dict[str, Any]:
        """返回工具的 JSON Schema 描述。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """工具注册中心：管理所有可用工具。"""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册一个工具。

        Args:
            tool: 要注册的工具实例
        """
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """按名称获取工具。

        Args:
            name: 工具名称

        Returns:
            工具实例，如果未找到返回 None
        """
        return self._tools.get(name)

    def list_all(self) -> List[Tool]:
        """列出所有注册的工具。

        Returns:
            工具列表
        """
        return list(self._tools.values())

    def get_schema(self) -> str:
        """生成所有工具的 JSON Schema 描述，用于拼接到 system prompt 中。

        Returns:
            格式化的工具描述字符串
        """
        schemas = []
        for tool in self._tools.values():
            schemas.append(
                f"- {tool.name}: {tool.description}\n"
                f"  参数: {tool.parameters}"
            )
        return "\n".join(schemas)

    def get_names(self) -> List[str]:
        """获取所有已注册工具的名称列表。"""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
