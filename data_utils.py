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
import ast
import pandas as pd
import numpy as np
import sys, os, signal
import re

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
    df = _load_parquet(parquet_path)

    is_simple_qa = any(
        key in str(parquet_path)
        for key in ["BRUMO25", "CMIMC25", "HMMT25"]
    )
    is_GSM8K =any(
        key in str(parquet_path)
        for key in ["GSM8K"]
    )

    samples: List[Sample] = []
    total = len(df) if n is None else min(n, len(df))

    for i in range(total):
        if is_simple_qa:
            question = str(df.at[i, "problem"]).strip()
            answer = str(df.at[i, "answer"]).strip()
        elif is_GSM8K:
            question = str(df.at[i, "question"]).strip()
            full_answer = str(df.at[i, "answer"])
            #print(repr(full_answer))
            answer = re.search(r'####\s*(.+)', full_answer).group(1).strip()

        else:
            # ---------- 处理 prompt ----------
            prompt_raw = df.at[i, "prompt"]
            prompt = prompt_raw

            # 1) 如果是 numpy 数组，先拿出里面的元素
            if isinstance(prompt_raw, np.ndarray):
                # 一般是 1 维，里面就一个元素
                if prompt_raw.size > 0:
                    prompt = prompt_raw[0]
                    #print("数组",prompt,type(prompt))
                else:
                    prompt = ""

            # 2) 如果是字符串，看起来像 list/dict 的，再尝试解析
            if isinstance(prompt, str):
                try:

                    parsed = ast.literal_eval(prompt)
                    prompt = parsed
                    #print("字符:",prompt,type(prompt))
                except Exception:
                    # 不是合法的 Python 表达式就保持字符串
                    pass

            # 3) 现在期望 prompt 是 list[dict]，取第一个 message 的 content
            if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
                # 形式: [{'content': '...', 'role': 'user'}, ...]
                question = str(prompt[0].get("content", "")).strip()
            elif isinstance(prompt, dict):
                # 形式: {'content': '...', 'role': 'user'}
                question = str(prompt.get("content", "")).strip()
            else:
                # 兜底：直接转成字符串
                question = str(prompt).strip()
            #print("DEBUG prompt type:", type(prompt_raw), "parsed type:", type(prompt), "question:", question[:60])

            #os.kill(os.getpid(), signal.SIGINT)

            # ---------- 处理 reward_model ----------
            reward_info_raw = df.at[i, "reward_model"]
            reward_info = reward_info_raw

            if isinstance(reward_info_raw, np.ndarray):
                if reward_info_raw.size > 0:
                    reward_info = reward_info_raw[0]
                else:
                    reward_info = ""

            if isinstance(reward_info, str):
                try:
                    parsed_rm = ast.literal_eval(reward_info)
                    reward_info = parsed_rm
                except Exception:
                    pass

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
