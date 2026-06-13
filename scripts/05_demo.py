#!/usr/bin/env python
"""Phase 4：交互式 Demo 脚本。

功能：
1. 加载训练好的模型
2. 提供命令行交互界面
3. 展示完整的 ReAct 推理过程
4. 支持 MD 记忆检索展示
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.parser import ConfigParser
from utils.logger import get_logger
from utils.helpers import set_seed

logger = get_logger(__name__)


def print_trajectory(traj):
    """打印推理轨迹（带格式化）。"""
    print(f"\n{'='*60}")
    print(f"Question: {traj.question}")
    print(f"{'='*60}")

    for step in traj.steps:
        print(f"\n--- Step {step.turn} ---")
        print(f"Thought: {step.thought[:200]}...")
        if step.action:
            print(f"Action: {step.action}")
        if step.observation:
            print(f"Observation: {step.observation[:200]}...")

    print(f"\n{'='*60}")
    print(f"Final Answer: {traj.final_answer}")
    print(f"Correct: {traj.is_correct}")
    print(f"Total Turns: {traj.total_turns}")
    print(f"{'='*60}\n")


def main():
    """交互式 Demo 主循环。"""
    config = ConfigParser()
    set_seed(config.training.get("seed", 42))

    logger.info("=== LLM Math Agent Demo ===")
    logger.info("输入 'quit' 退出, 'search <关键词>' 检索记忆")

    # TODO: 加载模型和 Agent
    # from model.loader import ModelLoader
    # from agent.react_agent import ReActAgent
    # from agent.memory import AgentMemory
    #
    # loader = ModelLoader(model_name=config.model["name"])
    # inference = loader.load_for_inference()
    # memory = AgentMemory()
    # memory.build_index()
    # agent = ReActAgent(model=inference, memory=memory)
    #
    # while True:
    #     try:
    #         question = input("\n>>> 请输入数学问题: ").strip()
    #     except (EOFError, KeyboardInterrupt):
    #         break
    #
    #     if not question:
    #         continue
    #     if question.lower() == "quit":
    #         break
    #     if question.lower().startswith("search "):
    #         query = question[7:]
    #         records = memory.retrieve_similar(query, k=3)
    #         for r in records:
    #             print(f"  [{r.status}] {r.question[:80]}... (score={r.similarity_score:.3f})")
    #         continue
    #
    #     traj = agent.run(question)
    #     print_trajectory(traj)
    #
    #     # 保存轨迹
    #     memory.save_trajectory(traj, status="success" if traj.is_correct else "failure")

    logger.info("=== Demo 就绪（当前为骨架代码，需在 GPU 环境运行）===")


if __name__ == "__main__":
    main()
