#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_generate_with_lora_afcurl.py

用途：
- 加载 base 模型 + 可选 LoRA 适配器（例如 AF-CuRL 训练结果）
- 对指定 csv (question / ground_truth) 做 K=4 采样，best-of-4
- 使用 reward_fn.compute_reward 做打分，选出每题 reward 最高的一条
- 导出包含预测与打分结果的 csv，并打印整体 summary

运行方式（在 AFCuRL_MathReasoning 根目录）：
    python scripts/eval/run_generate_with_lora_afcurl.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 从项目根目录的 reward_fn.py 导入（确保 reward_fn.py 在 AFCuRL_MathReasoning/ 下）
from reward_fn import compute_reward, RL_PHASE2_CONFIG


# ================== 1. 配置 ================== #

@dataclass
class EvalConfig:
    model_path: str
    lora_path: Optional[str]  # None 表示只用 base 模型
    data_csv: str
    output_csv: str

    k_samples: int = 4            # 每题采样次数 (best-of-k)
    max_prompt_tokens: int = 384
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# 工程根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ------ 在这里改路径，切换不同的数据 / LoRA ------ #
MODEL_PATH = PROJECT_ROOT / "models" / "DeepSeek-R1-Distill-Qwen-1.5B"
# 若不用 LoRA，设为 None
LORA_PATH = PROJECT_ROOT / "rl_runs" / "DeepSeek-R1-Distill-Qwen-1.5B" / "afcurl_gsm_math_k1_s42" / "final"

# 示例：在 gsm8k_dev 上评测 AF-CuRL
DATA_CSV = PROJECT_ROOT / "data" / "splits" / "gsm8k" / "gsm8k_dev.csv"
OUTPUT_CSV = PROJECT_ROOT / "eval_outputs" / "rl" / "afcurl_gsm_math_k1_s42" / "gsm8k_dev_k4.csv"

CFG = EvalConfig(
    model_path=str(MODEL_PATH),
    lora_path=str(LORA_PATH) if LORA_PATH is not None else None,
    data_csv=str(DATA_CSV),
    output_csv=str(OUTPUT_CSV),
)


# ================== 2. 工具函数 ================== #

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "question" not in df.columns or "ground_truth" not in df.columns:
        raise ValueError("data_csv 需要至少包含列: 'question', 'ground_truth'")
    # 保留可能存在的 task_name
    keep_cols = []
    if "task_name" in df.columns:
        keep_cols.append("task_name")
    keep_cols += ["question", "ground_truth"]
    return df[keep_cols].reset_index(drop=True)


def build_prompt(question: str) -> str:
    system = (
        "You are an expert competition mathematician. "
        "Directly give the final numerical answer in the format \\boxed{}. "
        "Do NOT show any reasoning steps."
    )
    return f"{system}\n\nProblem:\n{question}\n"


def create_model_and_tokenizer(cfg: EvalConfig):
    print(f"[INFO] Loading tokenizer from {cfg.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] Loading base model from {cfg.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    if cfg.lora_path is not None:
        print(f"[INFO] Loading LoRA adapter from {cfg.lora_path}")
        model = PeftModel.from_pretrained(
            model,
            cfg.lora_path,
        )

    model.to(cfg.device)
    model.eval()
    model.config.use_cache = True  # 推理阶段可以开缓存

    return model, tokenizer


def generate_one(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)

    # 截断过长 prompt
    if input_ids.shape[1] > max_prompt_tokens:
        input_ids = input_ids[:, -max_prompt_tokens:]
        attention_mask = attention_mask[:, -max_prompt_tokens:]

    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_text = tokenizer.decode(
        gen_ids[0][input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return gen_text


# ================== 3. 主流程 ================== #

def eval_with_lora(cfg: EvalConfig):
    set_seed(cfg.seed)

    df = load_data(cfg.data_csv)
    print(f"[INFO] Loaded {len(df)} samples from {cfg.data_csv}")

    # 确保输出目录存在
    out_path = Path(cfg.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = create_model_and_tokenizer(cfg)

    all_records: List[Dict[str, Any]] = []

    for idx, row in df.iterrows():
        task_name = row["task_name"] if "task_name" in row.index else None
        q = str(row["question"])
        gt = str(row["ground_truth"])

        prompt = build_prompt(q)

        best_sample = None
        best_reward = -1e9
        best_comps: Optional[Dict[str, Any]] = None

        for k in range(cfg.k_samples):
            resp = generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=cfg.device,
                max_prompt_tokens=cfg.max_prompt_tokens,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            )

            # 使用 Phase2 的 reward 配置做评测（更重视正确性）
            comps = compute_reward(
                response=resp,
                ground_truth=gt,
                cfg=RL_PHASE2_CONFIG,
                return_components=True,
            )
            R = comps["R_total"]

            if R > best_reward:
                best_reward = R
                best_sample = resp
                best_comps = comps

        assert best_sample is not None and best_comps is not None

        is_correct = best_comps.get("is_correct", None)
        has_valid_boxed = best_comps.get("has_valid_boxed", None)
        resp_len = best_comps.get("len_total", len(best_sample))

        record = {
            "index": idx,
            "question": q,
            "ground_truth": gt,
            "response": best_sample,
            "reward_total": best_reward,
            "is_correct": is_correct,
            "has_valid_boxed": has_valid_boxed,
            "response_len": resp_len,
        }
        if task_name is not None:
            record["task_name"] = task_name

        all_records.append(record)

        if (idx + 1) % 20 == 0:
            print(f"[INFO] Processed {idx+1}/{len(df)} examples...")

    out_df = pd.DataFrame(all_records)
    out_df.to_csv(cfg.output_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved eval results to {cfg.output_csv}")

    # ---- 简单 SUMMARY ----
    # 过滤掉 is_correct 为 None 的（例如 reward_fn 没能判断）
    if "is_correct" in out_df.columns:
        valid_mask = out_df["is_correct"].notnull()
        n_valid = valid_mask.sum()
        if n_valid > 0:
            acc = out_df.loc[valid_mask, "is_correct"].mean()
        else:
            acc = float("nan")
    else:
        acc = float("nan")

    if "has_valid_boxed" in out_df.columns:
        boxed_rate = out_df["has_valid_boxed"].mean()
    else:
        boxed_rate = float("nan")

    avg_len = out_df["response_len"].mean() if "response_len" in out_df.columns else float("nan")

    print(
        f"[SUMMARY] acc={acc:.3f}, "
        f"has_boxed_rate={boxed_rate:.3f}, "
        f"avg_response_len={avg_len:.1f}"
    )


if __name__ == "__main__":
    print("[INFO] Eval config:", CFG)
    eval_with_lora(CFG)
