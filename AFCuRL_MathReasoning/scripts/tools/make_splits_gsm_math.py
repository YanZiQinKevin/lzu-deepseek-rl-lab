#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从 AFCuRL_MathReasoning/data/benchmarks/train_gsm_math.csv
（包含列: task_name, question, ground_truth，且仅有 GSM8K / MATH-500）
切分出：

data/splits/gsm8k/
    - gsm8k_train_rl.csv
    - gsm8k_dev.csv
    - gsm8k_test.csv

data/splits/math500/
    - math500_train_rl.csv
    - math500_dev.csv
    - math500_test.csv

data/splits/combo/
    - gsm8k_math500_train_rl.csv    # 合并后的 RL 训练池

请在 AFCuRL_MathReasoning 根目录运行：
    python scripts/tools/make_splits_gsm_math.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.utils import shuffle


# 1. 路径设置
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]   # .../AFCuRL_MathReasoning

ORIG_COMBO_CSV = PROJECT_ROOT / "data" / "benchmarks" / "train_gsm_math.csv"

SPLIT_DIR_GSM8K   = PROJECT_ROOT / "data" / "splits" / "gsm8k"
SPLIT_DIR_MATH500 = PROJECT_ROOT / "data" / "splits" / "math500"
SPLIT_DIR_COMBO   = PROJECT_ROOT / "data" / "splits" / "combo"

RANDOM_SEED = 42

# 你可以根据实际数量调整这几个参数
GSM8K_TRAIN_RL = 200
GSM8K_DEV      = 150
# 剩余 = test

MATH_TRAIN_RL  = 200
MATH_DEV       = 150
# 剩余 = test


def split_df(
    df: pd.DataFrame,
    n_train_rl: int,
    n_dev: int,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """打乱后切分为 train_rl / dev / test。"""
    df = shuffle(df, random_state=seed).reset_index(drop=True)
    n_total = len(df)

    if n_train_rl + n_dev > n_total:
        raise ValueError(
            f"样本太少：需要 {n_train_rl + n_dev}，但只有 {n_total} 条。"
        )

    df_train_rl = df.iloc[:n_train_rl].reset_index(drop=True)
    df_dev      = df.iloc[n_train_rl:n_train_rl + n_dev].reset_index(drop=True)
    df_test     = df.iloc[n_train_rl + n_dev:].reset_index(drop=True)

    return df_train_rl, df_dev, df_test


def main():
    print(f"[INFO] Project root        : {PROJECT_ROOT}")
    print(f"[INFO] Loading combo CSV   : {ORIG_COMBO_CSV}")

    if not ORIG_COMBO_CSV.exists():
        raise FileNotFoundError(
            f"找不到 train_gsm_math.csv：{ORIG_COMBO_CSV}\n"
            f"请确认你已经按之前的脚本生成了该文件。"
        )

    df = pd.read_csv(ORIG_COMBO_CSV)
    required_cols = {"task_name", "question", "ground_truth"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"train_gsm_math.csv 必须包含列: {required_cols}，"
            f"当前列为: {df.columns.tolist()}"
        )

    # 按 task_name 分开
    df_gsm8k = df[df["task_name"] == "GSM8K"].reset_index(drop=True)
    df_math  = df[df["task_name"] == "MATH-500"].reset_index(drop=True)

    print(f"[INFO] Total combo samples : {len(df)}")
    print(f"[INFO] GSM8K samples       : {len(df_gsm8k)}")
    print(f"[INFO] MATH-500 samples    : {len(df_math)}")

    # GSM8K 切分
    gsm_train_rl, gsm_dev, gsm_test = split_df(
        df_gsm8k,
        n_train_rl=GSM8K_TRAIN_RL,
        n_dev=GSM8K_DEV,
        seed=RANDOM_SEED,
    )

    # MATH-500 切分
    math_train_rl, math_dev, math_test = split_df(
        df_math,
        n_train_rl=MATH_TRAIN_RL,
        n_dev=MATH_DEV,
        seed=RANDOM_SEED,
    )

    # 合并 RL 训练集（GSM8K+MATH-500）
    combo_train_rl = pd.concat(
        [
            gsm_train_rl.assign(task_name="GSM8K"),
            math_train_rl.assign(task_name="MATH-500"),
        ],
        ignore_index=True,
    )

    # 确保输出目录存在
    SPLIT_DIR_GSM8K.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR_MATH500.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR_COMBO.mkdir(parents=True, exist_ok=True)

    # 保存 GSM8K
    gsm_train_rl.to_csv(SPLIT_DIR_GSM8K / "gsm8k_train_rl.csv", index=False, encoding="utf-8-sig")
    gsm_dev.to_csv(     SPLIT_DIR_GSM8K / "gsm8k_dev.csv",      index=False, encoding="utf-8-sig")
    gsm_test.to_csv(    SPLIT_DIR_GSM8K / "gsm8k_test.csv",     index=False, encoding="utf-8-sig")

    # 保存 MATH-500
    math_train_rl.to_csv(SPLIT_DIR_MATH500 / "math500_train_rl.csv", index=False, encoding="utf-8-sig")
    math_dev.to_csv(     SPLIT_DIR_MATH500 / "math500_dev.csv",      index=False, encoding="utf-8-sig")
    math_test.to_csv(    SPLIT_DIR_MATH500 / "math500_test.csv",     index=False, encoding="utf-8-sig")

    # 保存合并 RL 训练集
    combo_train_rl.to_csv(
        SPLIT_DIR_COMBO / "gsm8k_math500_train_rl.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("[INFO] Done. Files saved to:")
    print("  -", SPLIT_DIR_GSM8K   / "gsm8k_train_rl.csv")
    print("  -", SPLIT_DIR_GSM8K   / "gsm8k_dev.csv")
    print("  -", SPLIT_DIR_GSM8K   / "gsm8k_test.csv")
    print("  -", SPLIT_DIR_MATH500 / "math500_train_rl.csv")
    print("  -", SPLIT_DIR_MATH500 / "math500_dev.csv")
    print("  -", SPLIT_DIR_MATH500 / "math500_test.csv")
    print("  -", SPLIT_DIR_COMBO   / "gsm8k_math500_train_rl.csv")


if __name__ == "__main__":
    main()
