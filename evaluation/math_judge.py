"""数学答案等价性判断器：综合多种策略判定答案正确性。"""

import re
from fractions import Fraction
from typing import Any, List, Optional


class MathJudge:
    """数学答案裁判。

    判断策略（按优先级）：
    1. 精确字符串匹配（归一化后）
    2. 数值比较（提取数值，相对误差 < 1e-6）
    3. SymPy 符号等价性（simplify(e1 - e2) == 0）
    4. 集合比较（处理无序答案）

    特殊处理：
    - 分数: 1/2 vs 0.5 → 数值比较判定
    - 表达式: x^2 vs x**2 → SymPy 标准化
    - 集合: {1,2} vs {2,1} → set 比较
    """

    # 相对误差阈值
    EPSILON = 1e-6

    def judge(self, predicted: str, ground_truth: str) -> bool:
        """判断预测答案是否正确。

        Args:
            predicted: Agent 预测的答案
            ground_truth: 标准答案

        Returns:
            是否正确
        """
        if not predicted or not ground_truth:
            return False

        pred = self._normalize(predicted)
        gt = self._normalize(ground_truth)

        # 1. 精确字符串匹配
        if pred == gt:
            return True

        # 2. 数值比较
        pred_num = self._extract_number(pred)
        gt_num = self._extract_number(gt)
        if pred_num is not None and gt_num is not None:
            if gt_num == 0:
                if abs(pred_num) < self.EPSILON:
                    return True
            else:
                rel_error = abs(pred_num - gt_num) / abs(gt_num)
                if rel_error < self.EPSILON:
                    return True

        # 3. SymPy 符号等价性
        if self._sympy_equivalent(predicted, ground_truth):
            return True

        # 4. 集合比较
        if self._sets_equal(predicted, ground_truth):
            return True

        return False

    def judge_batch(
        self,
        predictions: List[str],
        ground_truths: List[str],
    ) -> List[bool]:
        """批量判断答案正确性。

        Args:
            predictions: 预测答案列表
            ground_truths: 标准答案列表

        Returns:
            布尔值列表
        """
        return [
            self.judge(p, g) for p, g in zip(predictions, ground_truths)
        ]

    def _normalize(self, text: str) -> str:
        """标准化答案文本。"""
        if not text:
            return ""
        ans = text.strip()
        # 去空格
        ans = re.sub(r"\s+", "", ans)
        # 去末尾标点
        ans = ans.rstrip(".,;")
        # 统一 \frac 格式
        ans = ans.replace("\\frac", "")
        # 统一 ^ 和 **
        ans = ans.replace("**", "^")
        return ans.lower()

    def _extract_number(self, text: str) -> Optional[float]:
        """从文本中提取数值。"""
        if not text:
            return None

        # 分数格式
        frac_match = re.match(r"^(-?\d+)\s*/\s*(-?\d+)$", text.strip())
        if frac_match:
            num, den = int(frac_match.group(1)), int(frac_match.group(2))
            if den != 0:
                return num / den

        # 小数/科学计数法
        num_match = re.search(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", text)
        if num_match:
            try:
                return float(num_match.group(0))
            except ValueError:
                pass

        return None

    def _sympy_equivalent(self, e1: str, e2: str) -> bool:
        """通过 SymPy 判定表达式等价性。"""
        try:
            import sympy

            # 尝试解析为 SymPy 表达式
            s1 = sympy.sympify(e1, evaluate=False)
            s2 = sympy.sympify(e2, evaluate=False)
            diff = sympy.simplify(s1 - s2)
            return diff == 0
        except Exception:
            return False

    def _sets_equal(self, a1: str, a2: str) -> bool:
        """检查两个答案是否为等价的集合表示。"""
        # 提取用逗号分隔的元素
        pattern = r"[{[（(]?(.*?)[}）)]?"
        m1 = re.search(pattern, a1)
        m2 = re.search(pattern, a2)

        if m1 and m2:
            elems1 = set(e.strip() for e in m1.group(1).split(",") if e.strip())
            elems2 = set(e.strip() for e in m2.group(1).split(",") if e.strip())
            if elems1 and elems2 and elems1 == elems2:
                return True

        return False
