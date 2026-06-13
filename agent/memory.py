"""Agent 记忆模块：短期记忆 + 长期记忆（MD 文件持久化 + numpy 检索）。"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MemoryRecord:
    """单条长期记忆记录。"""

    id: str
    question: str
    trajectory_md: str  # 完整 MD 内容
    status: str  # "success" | "failure"
    tags: List[str] = field(default_factory=list)
    similarity_score: float = 0.0
    file_path: Optional[Path] = None


@dataclass
class Step:
    """ReAct 单步记录。"""

    turn: int
    thought: str
    action: Optional[Any] = None  # Optional[ToolCall]
    observation: Optional[str] = None  # 工具返回结果


@dataclass
class AgentTrajectory:
    """Agent 完整推理轨迹。"""

    question: str  # 问题文本
    ground_truth: str  # 标准答案
    steps: List[Step] = field(default_factory=list)  # ReAct 步骤序列
    final_answer: str = ""  # Agent 给出的最终答案
    total_turns: int = 0  # 消耗轮次
    is_correct: bool = False  # 是否正确


class AgentMemory:
    """Agent 记忆管理器。

    短期记忆：对话历史、工具调用结果缓存、草稿本
    长期记忆：MD 文件持久化 + numpy embedding 检索

    设计原则：
    - 零外部依赖：不需要 Chroma/FAISS/数据库
    - 人类可读：所有轨迹以 MD 格式存储
    - 轻量检索：embedding 矩阵存 numpy 内存，cosine similarity 直接 np.dot
    """

    def __init__(
        self,
        memory_dir: str | Path = "data/memory_store",
        max_context_length: int = 2048,
    ) -> None:
        """初始化记忆管理器。

        Args:
            memory_dir: 长期记忆存储目录
            max_context_length: 上下文窗口最大 token 数（估算）
        """
        self.memory_dir = Path(memory_dir)
        self.max_context_length = max_context_length

        # 短期记忆
        self._messages: List[Dict[str, str]] = []
        self._tool_results: Dict[str, Any] = {}
        self._scratchpad: List[str] = []

        # 长期记忆（embedding 检索）
        self._embeddings: Optional[np.ndarray] = None  # shape: (N, 384)
        self._index: Dict[int, Path] = {}  # 行号 → MD 文件路径
        self._embedder: Any = None  # 延迟加载 SentenceTransformer

    # ---- 短期记忆 ----

    def add_message(self, role: str, content: str) -> None:
        """添加一条对话消息。

        Args:
            role: 消息角色（system/user/assistant/tool）
            content: 消息内容
        """
        self._messages.append({"role": role, "content": content})

    def add_tool_result(self, tool_name: str, result: str) -> None:
        """缓存工具调用结果。

        Args:
            tool_name: 工具名称
            result: 工具返回结果
        """
        self._tool_results[tool_name] = result

    def get_context(self) -> List[Dict[str, str]]:
        """返回当前上下文窗口内的消息列表。"""
        return self._messages

    def summarize_old(self) -> str:
        """对超出窗口的历史消息进行摘要压缩。

        当消息列表超过上下文窗口一半时，将较早的消息压缩为简短摘要。
        保留最后 max_context_length//4 条消息不变。

        Returns:
            摘要文本，不需要压缩时返回空字符串
        """
        keep_recent = max(1, self.max_context_length // 8)
        if len(self._messages) <= keep_recent + 4:
            return ""

        old_msgs = self._messages[:-keep_recent]
        tool_calls = []
        observations = []

        for m in old_msgs:
            content = m.get("content", "")
            if "<tool_call>" in content:
                tool_calls.append(content[:100])
            elif m.get("role") == "tool":
                observations.append(content[:80])

        summary_parts = [f"[前 {len(old_msgs)} 条消息摘要]"]
        if tool_calls:
            summary_parts.append(f"使用了 {len(tool_calls)} 次工具调用")
        if observations:
            summary_parts.append("观察结果: " + "; ".join(observations[-3:]))

        summary = " ".join(summary_parts)

        # 用摘要替换旧消息
        self._messages = [
            {"role": "system", "content": summary}
        ] + self._messages[-keep_recent:]

        return summary

    def clear(self) -> None:
        """清空短期记忆。"""
        self._messages.clear()
        self._tool_results.clear()
        self._scratchpad.clear()

    # ---- 长期记忆（MD + embedding） ----

    def _get_embedder(self) -> Any:
        """延迟加载 sentence-transformers 模型（all-MiniLM-L6-v2, ~80MB）。"""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(
                "all-MiniLM-L6-v2", device="cpu"
            )
        return self._embedder

    def _encode(self, text: str) -> np.ndarray:
        """对文本进行 embedding 编码。

        Args:
            text: 输入文本

        Returns:
            embedding 向量 (384,)
        """
        embedder = self._get_embedder()
        emb = embedder.encode([text], show_progress_bar=False)
        return np.asarray(emb[0], dtype=np.float32)

    def build_index(self) -> None:
        """（重新）构建 embedding 索引。

        遍历 memory_store/ 目录下所有 MD 文件，
        提取 question 并计算 embedding，构建 numpy 矩阵。
        """
        if not self.memory_dir.exists():
            self._embeddings = None
            self._index = {}
            return

        md_files = list(self.memory_dir.glob("**/*.md"))
        if not md_files:
            self._embeddings = np.empty((0, 384), dtype=np.float32)
            self._index = {}
            return

        embedder = self._get_embedder()
        questions: List[str] = []
        valid_indices: List[int] = []
        valid_paths: List[Path] = []

        for i, md_path in enumerate(md_files):
            try:
                content = md_path.read_text(encoding="utf-8")
                # 提取 ## Question 块
                q_start = content.find("## Question")
                if q_start == -1:
                    continue
                q_end = content.find("\n## ", q_start + 5)
                if q_end == -1:
                    q_end = len(content)
                question_text = content[q_start:q_end].strip()
                questions.append(question_text)
                valid_indices.append(i)
                valid_paths.append(md_path)
            except Exception:
                continue

        if questions:
            embeddings = embedder.encode(questions, show_progress_bar=False)
            self._embeddings = np.asarray(embeddings, dtype=np.float32)
            self._index = {idx: path for idx, path in enumerate(valid_paths)}
        else:
            self._embeddings = np.empty((0, 384), dtype=np.float32)
            self._index = {}

    def retrieve_similar(
        self, question: str, k: int = 3, status_filter: Optional[str] = None
    ) -> List[MemoryRecord]:
        """检索与 question 最相似的历史轨迹。

        Args:
            question: 查询问题文本
            k: 返回 top-k 结果
            status_filter: 可选过滤（"success" / "failure"）

        Returns:
            按相似度排序的 MemoryRecord 列表
        """
        if self._embeddings is None or len(self._embeddings) == 0:
            return []

        query_emb = self._encode(question)  # (384,)

        # cosine similarity: dot / (norm * norm)
        emb_norms = np.linalg.norm(self._embeddings, axis=1)
        query_norm = np.linalg.norm(query_emb)

        if query_norm == 0:
            return []

        scores = np.dot(self._embeddings, query_emb) / (emb_norms * query_norm + 1e-8)

        # top-k indices
        k = min(k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        records: List[MemoryRecord] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            file_path = self._index.get(int(idx))
            if file_path is None:
                continue

            try:
                content = file_path.read_text(encoding="utf-8")
                record = self._parse_md_content(str(idx), content, file_path, score)
                if status_filter and record.status != status_filter:
                    continue
                records.append(record)
            except Exception:
                continue

        return records

    def get_few_shot_examples(self, question: str, k: int = 2) -> List[str]:
        """检索最相似的成功案例，格式化为 few-shot prompt 片段。

        Args:
            question: 查询问题文本
            k: 返回 top-k 示例

        Returns:
            格式化的 few-shot 示例列表
        """
        records = self.retrieve_similar(question, k=k, status_filter="success")
        examples: List[str] = []
        for rec in records:
            examples.append(f"<example>\n{rec.trajectory_md}\n</example>")
        return examples

    def save_trajectory(
        self,
        trajectory: AgentTrajectory,
        status: str,
        reflection: Optional[Dict[str, str]] = None,
    ) -> Path:
        """将完整轨迹保存为 MD 文件。

        Args:
            trajectory: Agent 推理轨迹
            status: "success" / "failure"
            reflection: 失败时的反思信息

        Returns:
            保存的文件路径
        """
        # 确保目录存在
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # 生成文件名
        import datetime

        date_str = datetime.date.today().isoformat()
        problem_id = f"problem_{date_str}_{hash(trajectory.question) % 10000:04d}"
        filename = f"{problem_id}_{status}.md"
        file_path = self.memory_dir / filename

        # 构建 MD 内容
        md_content = self._format_trajectory_md(
            trajectory, status, date_str, problem_id, reflection
        )

        file_path.write_text(md_content, encoding="utf-8")
        return file_path

    def _format_trajectory_md(
        self,
        trajectory: AgentTrajectory,
        status: str,
        date_str: str,
        problem_id: str,
        reflection: Optional[Dict[str, str]] = None,
    ) -> str:
        """将轨迹格式化为 MD 文件内容。"""
        lines = [
            f"# Problem ID: {problem_id}",
            f"# Date: {date_str}",
            f"# Status: {status}",
            "# Tags: []",
            "",
            "## Question",
            trajectory.question,
            "",
            "## Ground Truth",
            trajectory.ground_truth,
            "",
            "## Agent Trajectory",
        ]

        for step in trajectory.steps:
            lines.append(f"### Step {step.turn}")
            lines.append(f"- **Thought**: {step.thought}")
            if step.action:
                lines.append(f"- **Action**: `{step.action}`")
            if step.observation:
                lines.append(f"- **Observation**: {step.observation}")
            lines.append("")

        lines.append("## Final Answer")
        lines.append(f"<answer>{trajectory.final_answer}</answer>")
        lines.append("")
        lines.append("## Performance")
        lines.append(f"- Total Turns: {trajectory.total_turns}")
        tool_calls_count = sum(1 for s in trajectory.steps if s.action)
        lines.append(f"- Tool Calls: {tool_calls_count}")
        lines.append(f"- Correct: {trajectory.is_correct}")
        lines.append("")

        if reflection:
            lines.append("## Reflection")
            for key, val in reflection.items():
                lines.append(f"- {key}: {val}")

        return "\n".join(lines)

    def _parse_md_content(
        self, record_id: str, content: str, file_path: Path, score: float
    ) -> MemoryRecord:
        """从 MD 文件内容解析 MemoryRecord。"""
        # 提取状态
        status = "success"
        if "# Status: failure" in content:
            status = "failure"

        # 提取标签
        tags: List[str] = []
        tag_line_start = content.find("# Tags:")
        if tag_line_start != -1:
            tag_line_end = content.find("\n", tag_line_start)
            tag_str = content[tag_line_start:tag_line_end]
            tags = [t.strip() for t in tag_str.replace("# Tags:", "").strip("[] ").split(",") if t.strip()]

        # 提取 question
        question = ""
        q_start = content.find("## Question")
        if q_start != -1:
            q_end = content.find("\n## ", q_start + 5)
            if q_end == -1:
                q_end = len(content)
            question = content[q_start + len("## Question\n"): q_end].strip()

        return MemoryRecord(
            id=record_id,
            question=question,
            trajectory_md=content,
            status=status,
            tags=tags,
            similarity_score=score,
            file_path=file_path,
        )
