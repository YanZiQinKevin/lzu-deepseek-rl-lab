#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

gsm_dev = PROJECT_ROOT / "data" / "splits" / "gsm8k" / "gsm8k_dev.csv"
math_dev = PROJECT_ROOT / "data" / "splits" / "math500" / "math500_dev.csv"

out_path = PROJECT_ROOT / "data" / "splits" / "combo" / "gsm8k_math500_dev.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)

df_gsm = pd.read_csv(gsm_dev)
df_math = pd.read_csv(math_dev)

# 只保留 train/eval 必需列
keep_cols = ["question", "ground_truth"]
if "task_name" in df_gsm.columns:
    keep_cols = ["task_name"] + keep_cols

df_gsm = df_gsm[keep_cols].copy()
df_math = df_math[keep_cols].copy()

df_combo = pd.concat([df_gsm, df_math], ignore_index=True)
df_combo.to_csv(out_path, index=False, encoding="utf-8-sig")

print(f"[INFO] saved combo dev to {out_path}, n={len(df_combo)}")
