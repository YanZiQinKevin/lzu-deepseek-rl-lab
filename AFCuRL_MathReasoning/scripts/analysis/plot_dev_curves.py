#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def ema_smooth(x, alpha=0.2):
    y = []
    last = None
    for v in x:
        last = v if last is None else (1-alpha)*last + alpha*v
        y.append(last)
    return y

def plot_curves(monitor_csv, out_dir, smooth_alpha=0.2):
    df = pd.read_csv(monitor_csv)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = df["step"].values

    # --- 1) Acc curve ---
    acc = df["acc"].values
    acc_s = ema_smooth(acc, alpha=smooth_alpha)

    plt.figure(figsize=(6.0, 4.0), dpi=200)
    plt.plot(steps, acc, linewidth=1.0, alpha=0.35, label="raw")
    plt.plot(steps, acc_s, linewidth=2.0, label=f"EMA α={smooth_alpha}")
    plt.xlabel("RL steps")
    plt.ylabel("Dev accuracy")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "dev_acc_curve.png")
    plt.savefig(out_dir / "dev_acc_curve.pdf")   # 论文建议用 pdf
    plt.close()

    # --- 2) Boxed rate curve ---
    br = df["boxed_rate"].values
    br_s = ema_smooth(br, alpha=smooth_alpha)

    plt.figure(figsize=(6.0, 4.0), dpi=200)
    plt.plot(steps, br, linewidth=1.0, alpha=0.35, label="raw")
    plt.plot(steps, br_s, linewidth=2.0, label=f"EMA α={smooth_alpha}")
    plt.xlabel("RL steps")
    plt.ylabel("Dev boxed rate")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "dev_boxed_curve.png")
    plt.savefig(out_dir / "dev_boxed_curve.pdf")
    plt.close()

    # --- 3) Avg length curve ---
    al = df["avg_len"].values
    al_s = ema_smooth(al, alpha=smooth_alpha)

    plt.figure(figsize=(6.0, 4.0), dpi=200)
    plt.plot(steps, al, linewidth=1.0, alpha=0.35, label="raw")
    plt.plot(steps, al_s, linewidth=2.0, label=f"EMA α={smooth_alpha}")
    plt.xlabel("RL steps")
    plt.ylabel("Avg response length")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "dev_len_curve.png")
    plt.savefig(out_dir / "dev_len_curve.pdf")
    plt.close()

    print("[INFO] Saved curves to:", out_dir)

if __name__ == "__main__":
    monitor_csv = "path/to/dev_monitor.csv"
    out_dir = "path/to/figs"
    plot_curves(monitor_csv, out_dir)
