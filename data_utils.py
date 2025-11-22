# data_utils.py
"""
数据加载与标准化工具。

核心目标：
- 从各个 parquet 文件中读出题目与标准答案；
- 统一成 Sample 对象列表，以方便后续生成与评测脚本使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


@dataclass
class Sample:
    """单个题目的标准化表示。"""
    task_name: str      # 例如 "AIME24"
    example_id: int     # 在该任务中的索引（从 0 开始）
    question: str       # 题目文本 / prompt
    answer: str         # 标准答案（ground truth）


def _load_parquet(filepath: Path) -> pd.DataFrame:
    """简单封装，后面如果你想加缓存/日志可以在这里改。"""
    return pd.read_parquet(filepath)


def load_task_samples(
    task_name: str,
    parquet_path: Path,
    n: Optional[int] = None,
) -> List[Sample]:
    """
    从单个任务的 parquet 文件中读出样本，并转换为 Sample 列表。

    参数
    ----
    task_name : 任务名称，例如 "AIME24"
    parquet_path : 该任务对应的 .parquet 文件路径
    n : 如果为 None，使用全部样本；否则只取前 n 条（用于快速测试）

    返回
    ----
    samples : List[Sample]
    """
    df = _load_parquet(parquet_path)

    # 兼容两种格式：
    # 1) BRUMO25 / CMIMC25 / HMMT25:
    #       - 列 "problem" 存题目
    #       - 列 "answer"  存标准答案
    # 2) 其他数据集（DAPO 风格）:
    #       - 列 "prompt" 是一个 list[{"role":..., "content": ...}, ...]
    #       - 列 "reward_model" 是一个 dict，里面有 "ground_truth"
    is_simple_qa = any(
        key in str(parquet_path)
        for key in ["BRUMO25", "CMIMC25", "HMMT25"]
    )

    samples: List[Sample] = []

    # 决定要遍历多少条
    total = len(df) if n is None else min(n, len(df))

    for i in range(total):
        if is_simple_qa:
            # 直接从 "problem" / "answer" 取
            question = str(df.at[i, "problem"]).strip()
            answer = str(df.at[i, "answer"]).strip()
        else:
            # DAPO / 其他：prompt 是对话形式，取第一条 user 的 content
            prompt = df.at[i, "prompt"]
            if isinstance(prompt, list) and len(prompt) > 0:
                question = str(prompt[0]["content"]).strip()
            else:
                # 兜底：直接转成字符串
                question = str(prompt).strip()

            reward_info = df.at[i, "reward_model"]
            # 通常是个 dict，有 "ground_truth" 键
            if isinstance(reward_info, dict):
                answer = str(reward_info.get("ground_truth", "")).strip()
            else:
                answer = str(reward_info).strip()

        samples.append(
            Sample(
                task_name=task_name,
                example_id=i,
                question=question,
                answer=answer,
            )
        )

    return samples


def load_all_tasks(task_configs: Iterable[dict]) -> List[Sample]:
    """
    一次性加载多个任务的样本（可选）。

    参数
    ----
    task_configs : 来自 config_eval.TASKS 的列表，每个包含
                   {"name": str, "path": Path, "N": Optional[int]}

    返回
    ----
    all_samples : 把所有任务的 Sample 拼在一起的列表
    """
    all_samples: List[Sample] = []
    for cfg in task_configs:
        task_name = cfg["name"]
        path: Path = cfg["path"]
        n = cfg.get("N", None)
        task_samples = load_task_samples(task_name, path, n)
        all_samples.extend(task_samples)
    return all_samples
