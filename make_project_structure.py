#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
在 LLMProject 下面创建一个新的“干净论文工程”目录：
    AFCuRL_MathReasoning/

并在其内部生成我们约定好的子目录结构。
请在 LLMProject 根目录下直接运行本脚本。
"""

from pathlib import Path

# 1. 设定 LLMProject 根目录（当前脚本所在目录的父目录）
# 如果你把脚本放在 LLMProject 根目录下，也可以直接用 Path.cwd()
LLMPROJECT_ROOT = Path(__file__).resolve().parent

# 2. 新工程目录名（你可以改成自己喜欢的）
PROJECT_NAME = "AFCuRL_MathReasoning"

# 3. 目标根目录
ROOT = LLMPROJECT_ROOT / PROJECT_NAME

def make_dirs():
    # 顶层目录
    dirs = [
        ROOT / "models",
        ROOT / "data" / "benchmarks",
        ROOT / "data" / "splits" / "gsm8k",
        ROOT / "data" / "splits" / "math500",
        ROOT / "data" / "splits" / "aime24",
        ROOT / "data" / "splits" / "aime25",
        ROOT / "data" / "splits" / "amc23",
        ROOT / "data" / "splits" / "minerva",
        ROOT / "data" / "splits" / "olympiad",
        ROOT / "data" / "splits" / "combo",    # gsm8k+math500 合并RL集

        ROOT / "rl_runs" / "DeepSeek-R1-Distill-Qwen-1.5B",
        ROOT / "eval_outputs" / "baselines",
        ROOT / "eval_outputs" / "rl",

        ROOT / "scripts" / "eval",
        ROOT / "scripts" / "rl",
        ROOT / "scripts" / "analysis",
        ROOT / "scripts" / "tools",

        ROOT / "logs",
        ROOT / "notes",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # 可选：每个空目录里放一个 .gitkeep（如果你以后打算用 git）
    for d in dirs:
        gitkeep = d / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()

    print(f"[INFO] Project root created at: {ROOT}")
    print("[INFO] Subdirectories:")
    for d in dirs:
        print("  -", d.relative_to(LLMPROJECT_ROOT))


if __name__ == "__main__":
    print(f"[INFO] LLMProject root: {LLMPROJECT_ROOT}")
    make_dirs()
