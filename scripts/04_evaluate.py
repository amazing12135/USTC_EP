#!/usr/bin/env python
"""模型评估脚本。

Phase 1 (SFT 后): 评估格式合规率 — 模型是否正确输出 <tool_call> / <answer>
Phase 2/3: 评估 MATH accuracy + tool usage 等完整指标

用法:
  # SFT 格式评估（需要 GPU / vLLM）
  python scripts/04_evaluate.py --model checkpoints/sft/ --dataset l1 --size 100

  # 从 saved trajectories 评估（无需 GPU）
  python scripts/04_evaluate.py --from-trajectories data/eval_trajs.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.parser import ConfigParser
from utils.logger import get_logger
from utils.helpers import set_seed

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 LLM Math Agent")
    parser.add_argument(
        "--model", type=str, default=None,
        help="模型 checkpoint 路径",
    )
    parser.add_argument(
        "--dataset", type=str, default="l1",
        choices=["l1", "l2", "all"],
        help="评估数据集",
    )
    parser.add_argument(
        "--size", type=int, default=100,
        help="评估样本数量",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="运行完整评估（全部验证集）",
    )
    parser.add_argument(
        "--from-trajectories", type=str, default=None,
        help="从已保存的轨迹文件评估（跳过推理）",
    )
    return parser.parse_args()


# ================================================================
# 格式合规率评估（Phase 1 核心指标）
# ================================================================

def compute_format_compliance(trajectories: List[Any]) -> Dict[str, Any]:
    """计算格式合规率。

    Phase 1 SFT 后需要检查的格式项：
    - tool_call_tag: 至少一次正确的 <tool_call>...</tool_call>
    - answer_tag: 有 <answer>...</answer>
    - valid_json: <tool_call> 内是合法 JSON
    - named_tool: JSON 中有 "name" 字段
    """
    n = len(trajectories)
    stats = {
        "total": n,
        "tool_call_present": 0,
        "answer_tag_present": 0,
        "valid_json": 0,
        "named_tool": 0,
        "all_compliant": 0,  # 四项全过
    }

    for traj in trajectories:
        full_text = ""
        for step in getattr(traj, "steps", []):
            full_text += getattr(step, "thought", "") + "\n"
        full_text += getattr(traj, "final_answer", "")

        has_tool = "<tool_call>" in full_text and "</tool_call>" in full_text
        has_answer = "<answer>" in full_text and "</answer>" in full_text
        valid_json = False
        named = False

        if has_tool:
            stats["tool_call_present"] += 1
            import re
            match = re.search(r"<tool_call>(.*?)</tool_call>", full_text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1).strip())
                    valid_json = True
                    if data.get("name"):
                        named = True
                except json.JSONDecodeError:
                    pass

        if valid_json:
            stats["valid_json"] += 1
        if named:
            stats["named_tool"] += 1
        if has_answer:
            stats["answer_tag_present"] += 1

        if has_tool and has_answer and valid_json and named:
            stats["all_compliant"] += 1

    # 百分比
    pct = {}
    for k in ["tool_call_present", "answer_tag_present", "valid_json", "named_tool", "all_compliant"]:
        pct[k] = stats[k] / n * 100 if n > 0 else 0

    return {"counts": stats, "percentages": pct}


# ================================================================
# 评估主流程
# ================================================================

def evaluate_with_model(
    config: ConfigParser,
    model_path: Optional[str],
    dataset: str,
    size: int,
) -> None:
    """使用模型进行推理评估（需要 GPU）。"""
    from model.loader import ModelLoader
    from model.inference import BatchInference
    from agent.react_agent import ReActAgent
    from data.dataset import DatasetManager
    from evaluation.metrics import MetricsCalculator

    # 加载模型
    loader = ModelLoader(
        model_name=model_path or config.get("model.name"),
    )
    inference = loader.load_for_inference(lora_path=model_path)
    agent = ReActAgent(model=inference)

    # 加载数据
    dm = DatasetManager(data_config=config.data)
    if dataset == "l1":
        problems = dm.load_l1_data()[:size]
    elif dataset == "l2":
        problems = dm.load_l2_data()[:size]
    else:
        problems = dm.load_l1_data()[:size // 2] + dm.load_l2_data()[:size // 2]

    logger.info(f"评估 {len(problems)} 题...")

    # 推理
    trajectories = []
    for i, p in enumerate(problems):
        traj = agent.run(p.question, p.answer)
        trajectories.append(traj)
        if (i + 1) % 20 == 0:
            logger.info(f"  {i+1}/{len(problems)}")

    # 计算指标
    metrics = MetricsCalculator().compute_all(trajectories)
    fmt_stats = compute_format_compliance(trajectories)

    _print_results(metrics, fmt_stats)

    # 保存轨迹
    output_path = Path("outputs/evals") / f"eval_{dataset}_{len(problems)}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for t in trajectories:
            f.write(json.dumps({
                "question": t.question,
                "ground_truth": t.ground_truth,
                "final_answer": t.final_answer,
                "is_correct": t.is_correct,
                "total_turns": t.total_turns,
                "steps": [
                    {
                        "turn": s.turn,
                        "thought": s.thought[:200],
                        "has_action": s.action is not None,
                        "observation": (
                            s.observation[:200] if s.observation else None
                        ),
                    }
                    for s in t.steps
                ],
            }, ensure_ascii=False) + "\n")
    logger.info(f"轨迹已保存: {output_path}")


def evaluate_from_trajectories(traj_path: str) -> None:
    """从已保存的轨迹文件评估（无需 GPU）。"""
    logger.info(f"从轨迹文件评估: {traj_path}")
    # 这只是格式评估，不需要真实 trajectories
    records = []
    with open(traj_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # 构造简化轨迹对象
    class SimpleTraj:
        def __init__(self, d):
            self.final_answer = d.get("final_answer", "")
            self.is_correct = d.get("is_correct", False)
            self.total_turns = d.get("total_turns", 0)
            self.steps = []
            for s in d.get("steps", []):
                class SimpleStep:
                    pass
                step = SimpleStep()
                step.thought = s.get("thought", "")
                step.action = "dummy" if s.get("has_action") else None
                step.observation = s.get("observation")
                self.steps.append(step)

    trajectories = [SimpleTraj(r) for r in records]
    fmt_stats = compute_format_compliance(trajectories)

    logger.info(f"评估 {len(trajectories)} 条轨迹")
    _print_results(None, fmt_stats)


def _print_results(
    metrics: Any,
    fmt_stats: Dict[str, Any],
) -> None:
    """打印评估结果。"""
    if metrics is not None:
        logger.info(f"\n{'='*50}")
        logger.info("综合指标")
        logger.info(f"{'='*50}")
        logger.info(f"  Accuracy:           {metrics.accuracy*100:.1f}%")
        logger.info(f"  Avg Turns:          {metrics.avg_turns:.1f}")
        logger.info(f"  Tool Call Rate:     {metrics.tool_call_rate*100:.1f}%")
        logger.info(f"  Tool Success Rate:  {metrics.tool_success_rate*100:.1f}%")
        logger.info(f"  Format Compliance:  {metrics.format_compliance*100:.1f}%")

    logger.info(f"\n{'='*50}")
    logger.info("格式合规率 (Phase 1 核心指标)")
    logger.info(f"{'='*50}")
    pct = fmt_stats["percentages"]
    logger.info(f"  <tool_call> 存在:   {pct['tool_call_present']:.1f}%")
    logger.info(f"  <answer> 存在:      {pct['answer_tag_present']:.1f}%")
    logger.info(f"  JSON 合法:          {pct['valid_json']:.1f}%")
    logger.info(f"  工具名存在:         {pct['named_tool']:.1f}%")
    logger.info(f"  全部合规:           {pct['all_compliant']:.1f}%")
    logger.info(f"{'='*50}\n")

    # Phase 1 验证标准
    if pct["valid_json"] >= 90:
        logger.info("[PASS] 工具调用格式正确率 > 90%")
    else:
        logger.info("[FAIL] 工具调用格式正确率 < 90%, 需要优化 SFT 数据或 prompt")
    if pct["answer_tag_present"] >= 95:
        logger.info("[PASS] 答案提取成功率 > 95%")
    else:
        logger.info("[FAIL] 答案提取成功率 < 95%")


# ================================================================
# 入口
# ================================================================

def main() -> None:
    args = parse_args()
    config = ConfigParser()
    set_seed(config.get("training.seed", 42))

    logger.info("=== 模型评估 ===")

    if args.from_trajectories:
        evaluate_from_trajectories(args.from_trajectories)
    else:
        evaluate_with_model(config, args.model, args.dataset, args.size)


if __name__ == "__main__":
    main()
