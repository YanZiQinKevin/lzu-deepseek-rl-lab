# reward_analysis.py
# -*- coding: utf-8 -*-

"""
对已有的 graded_samples.csv 做离线 reward 分布分析。

输入:
    graded_samples.csv （由 run_grade_baseline.py 生成）
    列通常包含:
        - task_name
        - example_id
        - sample_id
        - is_correct
        - question
        - ground_truth
        - response

输出:
    - graded_samples_with_reward.csv : 每一条样本的 reward 及子项

# R_total_mean
# 该任务所有样本的总奖励
# R 的平均值。
# 反映这个任务整体上“被奖励”还是“被惩罚”。
#
# R_total_std
# 总奖励
# R 的标准差。
# 反映这个任务内 reward 的波动大小。
#
# R_correct_mean
# R_correct 的平均值，其实就是准确率 acc@1（因为对的=1，错的=0）。
#
# R_format_mean
# R_format 的平均值：
# 有合法 \boxed{数字} 的样本记 +1，没有记 -1，取平均。
# 越接近 1 表示越多样本有规范答案，越接近 -1 表示几乎都没给答案。
#
# R_efficiency_mean
# R_efficiency 的平均值（长度惩罚）。
# 为 0 或接近 0：长度大多没被惩罚；
# 负得越多：说明这个任务的输出整体偏长、被扣分多。
#
# R_closeness_mean
# R_closeness 的平均值（数值接近奖励）。
# 大致反映“答案数值与标准答案接近”的样本比例。
# R_total_mean_correct
# 只在正确样本中，R_total 的平均值。
# 正得越高，说明“做对时奖励很足”。
#
# R_total_mean_wrong
# 只在错误样本中，R_total 的平均值。
# 通常是负数，负得越多，说明“做错时惩罚很重
    - reward_stats_by_task.csv       : 按 task_name 聚合的统计数据
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from reward_fn import compute_reward, DEFAULT_CONFIG


def analyze_rewards(df: pd.DataFrame):
    """
    对整张 DataFrame 计算 reward，并返回:
        - 带 reward 列的 DataFrame
        - 按 task_name 汇总的统计 DataFrame
    """
    required_cols = ["task_name", "response", "ground_truth"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV 中缺少必要列: {col}")

    # 确保 task_name 存在
    df["task_name"] = df["task_name"].astype(str)

    # 逐行计算 reward
    R_total_list = []
    R_correct_list = []
    R_format_list = []
    R_eff_list = []
    R_close_list = []

    print("[INFO] 开始逐样本计算 reward ...")

    for idx, row in df.iterrows():
        response = row.get("response", "")
        gt = row.get("ground_truth", "")
        comps = compute_reward(
            response=response,
            ground_truth=gt,
            cfg=DEFAULT_CONFIG,
            return_components=True,
        )
        R_total_list.append(comps["R_total"])
        R_correct_list.append(comps["R_correct"])
        R_format_list.append(comps["R_format"])
        R_eff_list.append(comps["R_efficiency"])
        R_close_list.append(comps["R_closeness"])

        if (idx + 1) % 1000 == 0:
            print(f"[INFO] 已处理 {idx + 1} 条样本")

    df["R_total"] = R_total_list
    df["R_correct"] = R_correct_list
    df["R_format"] = R_format_list
    df["R_efficiency"] = R_eff_list
    df["R_closeness"] = R_close_list

    # 如果有 is_correct，就统一成 int
    if "is_correct" in df.columns:
        df["is_correct"] = df["is_correct"].astype(int)

    # 按 task_name 聚合
    stats_rows = []
    group = df.groupby("task_name")

    for task_name, g in group:
        n = len(g)
        row = {
            "task_name": task_name,
            "num_samples": n,
            "R_total_mean": g["R_total"].mean(),
            "R_total_std": g["R_total"].std(),
            "R_correct_mean": g["R_correct"].mean(),
            "R_format_mean": g["R_format"].mean(),
            "R_efficiency_mean": g["R_efficiency"].mean(),
            "R_closeness_mean": g["R_closeness"].mean(),
        }

        if "is_correct" in g.columns:
            # 根据 is_correct == 1/0 分别统计
            g_correct = g[g["is_correct"] == 1]
            g_wrong = g[g["is_correct"] == 0]

            if len(g_correct) > 0:
                row.update(
                    {
                        "R_total_mean_correct": g_correct["R_total"].mean(),
                        "R_total_mean_wrong": g_wrong["R_total"].mean()
                        if len(g_wrong) > 0
                        else 0.0,
                    }
                )
            else:
                row.update(
                    {
                        "R_total_mean_correct": 0.0,
                        "R_total_mean_wrong": g_wrong["R_total"].mean()
                        if len(g_wrong) > 0
                        else 0.0,
                    }
                )

        stats_rows.append(row)

    stats_df = pd.DataFrame(stats_rows).sort_values("task_name")
    return df, stats_df


def main():
    parser = argparse.ArgumentParser(
        description="Offline reward analysis based on graded_samples.csv"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="graded_samples.csv 的路径",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="输出目录（默认与 csv 同目录）",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV 文件: {csv_path}")

    out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 读取: {csv_path}")
    df = pd.read_csv(csv_path)

    df_with_reward, stats_df = analyze_rewards(df)

    out_csv = out_dir / "graded_samples_with_reward.csv"
    out_stats = out_dir / "reward_stats_by_task.csv"

    df_with_reward.to_csv(out_csv, index=False)
    stats_df.to_csv(out_stats, index=False)

    print(f"\n[INFO] 每条样本的 reward 已保存到: {out_csv}")
    print(f"[INFO] 按任务汇总的 reward 统计已保存到: {out_stats}\n")

    print("========== Reward summary (per task) ==========")
    for _, row in stats_df.iterrows():
        print(
            f"{row['task_name']:20s} "
            f"R_total_mean={row['R_total_mean']:.3f} "
            f"(correct={row.get('R_total_mean_correct', 0.0):.3f}, "
            f"wrong={row.get('R_total_mean_wrong', 0.0):.3f})"
        )


if __name__ == "__main__":
    import sys

    sys.argv = ["reward_analysis.py", "--csv-path",
                "/mnt/d/OneDrive/兰州交大/博士课题-大模型/LLMProject/eval_outputs/DeepSeek-R1-Distill-Qwen-1.5B/graded_samples.csv"]
    main()
