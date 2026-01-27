#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
behavior_analysis.py

对 graded_samples.csv 做行为分析：
- 每条样本提取：
  - 是否包含合法 \boxed{number}
  - 输出长度（字符数）
  - <think> 段长度（若存在）
- 按 task_name 汇总统计，并保存到 behavior_stats.csv
- 生成带标注的 graded_samples_with_flags.csv，方便后续深入分析

假设 graded_samples.csv 列包含：
    task_name, example_id, sample_id, is_correct, question, ground_truth, response
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any

import pandas as pd


def extract_boxed_contents(text: str):
    """
    从 response 中提取所有 \boxed{...} 的内容。
    返回一个 list[str]，其中只保留非空且包含数字的内容，
    以避免把 prompt 里的 \boxed{}（空括号）当成有效答案。
    """
    if not isinstance(text, str):
        return []

    pattern = r"\\boxed\{([^}]*)\}"
    matches = re.findall(pattern, text)
    results = []
    for m in matches:
        content = m.strip()
        # 至少包含一个数字，且不为空
        if content and re.search(r"\d", content):
            results.append(content)
    return results


def has_valid_boxed_number(text: str) -> bool:
    """是否包含至少一个非空且含数字的 \boxed{...}。"""
    return len(extract_boxed_contents(text)) > 0


def compute_think_length(text: str) -> int:
    """
    粗略估计 <think> 段长度（按字符数）。
    - 如果同时存在 "<think>" 和 "</think>"，取二者之间的部分长度；
    - 如果只有 "<think>"，取从 "<think>" 到结尾的长度；
    - 如果没有，则返回 0。
    """
    if not isinstance(text, str):
        return 0

    start_tag = "<think>"
    end_tag = "</think>"

    start_idx = text.find(start_tag)
    if start_idx == -1:
        # 有些 DeepSeek 样本可能是 "<｜Assistant｜><think>" 这种，
        # 可以再宽松一点匹配
        alt = "<｜Assistant｜><think>"
        start_idx = text.find(alt)
        if start_idx != -1:
            start_tag = alt

    if start_idx == -1:
        return 0

    end_idx = text.find(end_tag, start_idx)
    if end_idx == -1:
        # 没有结束标签，就认为一直到结尾
        return len(text) - start_idx

    # 包含 end_tag
    return end_idx + len(end_tag) - start_idx


def analyze_behavior(df: pd.DataFrame) -> pd.DataFrame:
    """
    对整张 DataFrame 做行为分析，返回每个 task 的统计信息 DataFrame。
    """
    required_cols = ["task_name", "is_correct", "response"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV 中缺少必要列: {col}")

    # 转成合适类型
    df["task_name"] = df["task_name"].astype(str)
    # 有些 is_correct 可能是 bool，有些是 0/1，这里统一成 int
    df["is_correct"] = df["is_correct"].astype(int)

    # 增加行为标注列
    df["has_valid_boxed"] = df["response"].apply(has_valid_boxed_number)
    df["char_len"] = df["response"].astype(str).str.len()
    df["think_char_len"] = df["response"].apply(compute_think_length)

    # 方便后续导出
    print("[INFO] 示例：前 3 条带标注的样本：")
    print(df[["task_name", "example_id", "sample_id", "is_correct",
              "has_valid_boxed", "char_len", "think_char_len"]].head(3))

    # 按 task_name 聚合统计
    rows = []
    task_groups = df.groupby("task_name")

    for task_name, g in task_groups:
        total = len(g)
        correct = g["is_correct"].sum()
        acc = correct / total if total > 0 else 0.0

        has_box_rate = g["has_valid_boxed"].mean() if total > 0 else 0.0
        has_box_in_correct = g.loc[g["is_correct"] == 1, "has_valid_boxed"]
        has_box_in_incorrect = g.loc[g["is_correct"] == 0, "has_valid_boxed"]

        box_rate_correct = (
            has_box_in_correct.mean() if len(has_box_in_correct) > 0 else 0.0
        )
        box_rate_incorrect = (
            has_box_in_incorrect.mean() if len(has_box_in_incorrect) > 0 else 0.0
        )

        avg_len = g["char_len"].mean() if total > 0 else 0.0
        avg_len_correct = (
            g.loc[g["is_correct"] == 1, "char_len"].mean()
            if correct > 0
            else 0.0
        )
        avg_len_incorrect = (
            g.loc[g["is_correct"] == 0, "char_len"].mean()
            if (total - correct) > 0
            else 0.0
        )

        avg_think_len = g["think_char_len"].mean() if total > 0 else 0.0

        rows.append(
            {
                "task_name": task_name,
                "num_samples": total,
                "num_correct": int(correct),
                "acc@1": acc,
                "has_valid_boxed_rate": has_box_rate,
                "boxed_rate_in_correct": box_rate_correct,
                "boxed_rate_in_incorrect": box_rate_incorrect,
                "avg_char_len": avg_len,
                "avg_char_len_correct": avg_len_correct,
                "avg_char_len_incorrect": avg_len_incorrect,
                "avg_think_char_len": avg_think_len,
            }
        )

    stats_df = pd.DataFrame(rows).sort_values("task_name")
    return df, stats_df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze baseline behavior from graded_samples.csv"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="graded_samples.csv 的路径（由 run_grade_baseline.py 生成）",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="输出目录（默认与 csv 文件同目录）",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")

    out_dir = Path(args.out_dir) if args.out_dir is not None else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 读取 CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    df_with_flags, stats_df = analyze_behavior(df)

    # 保存统计结果
    stats_path = out_dir / "behavior_stats.csv"
    df_path = out_dir / "graded_samples_with_flags.csv"

    stats_df.to_csv(stats_path, index=False)
    df_with_flags.to_csv(df_path, index=False)

    print(f"\n[INFO] 行为统计已保存到: {stats_path}")
    print(f"[INFO] 带标注的样本已保存到: {df_path}\n")

    # 终端打印一个简要 summary
    print("========== Per-task behavior summary ==========")
    for _, row in stats_df.iterrows():
        print(
            f"{row['task_name']:20s} "
            f"acc@1={row['acc@1']:.3f}  "
            f"box_rate={row['has_valid_boxed_rate']:.3f}  "
            f"avg_len={row['avg_char_len']:.1f}  "
            f"avg_think_len={row['avg_think_char_len']:.1f}"
        )


if __name__ == "__main__":
    import sys

    sys.argv = ["behavior_analysis.py", "--csv-path", "/mnt/d/OneDrive/兰州交大/博士课题-大模型/LLMProject/eval_outputs/AF-CuRL/graded_samples.csv"]
    main()
