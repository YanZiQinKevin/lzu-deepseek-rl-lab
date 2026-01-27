#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_generate_with_lora_v2.py

用途：
- 加载 base DeepSeek-R1-Distill-Qwen-1.5B + 训练好的 LoRA（AF-CuRL v2）
- 在指定 eval CSV 上做 K 次采样（best-of-K，K=cfg.k_samples）
- 用 RL_PHASE2_CONFIG 的 reward 做打分
- 写出结果 CSV，并打印 SUMMARY（acc, boxed_rate, avg_len）

使用示例（从项目根目录）：
    python scripts/rl/run_generate_with_lora_v2.py
"""

from dataclasses import dataclass
from pathlib import Path
import sys

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---------- 项目根路径 & reward_fn 导入 ----------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import reward_fn  # noqa: E402
from reward_fn import compute_reward, RL_PHASE2_CONFIG  # noqa: E402

print("[DEBUG] reward_fn loaded from:", reward_fn.__file__)


# ---------- 配置 ----------

@dataclass
class EvalConfig:
    base_model_path: str
    lora_path: str
    eval_csv: str
    output_csv: str

    k_samples: int = 2
    max_prompt_tokens: int = 384
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- 工具函数 ----------

def build_prompt(q: str) -> str:
    system = (
        "You are an expert competition mathematician. "
        "Directly give the final numerical answer in the format \\boxed{}. "
        "Do NOT show any reasoning steps."
    )
    return f"{system}\n\nProblem:\n{q}\n"


def load_eval_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert {"question", "ground_truth"}.issubset(df.columns), \
        f"Eval CSV 必须包含 question, ground_truth 列，目前列为: {df.columns.tolist()}"
    return df


def create_model_and_tokenizer(cfg: EvalConfig):
    print(f"[INFO] Loading tokenizer from {cfg.base_model_path}")
    tok = AutoTokenizer.from_pretrained(
        cfg.base_model_path,
        use_fast=False,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[INFO] Loading base model from {cfg.base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        trust_remote_code=True,
    )

    print(f"[INFO] Loading LoRA adapter from {cfg.lora_path}")
    model = PeftModel.from_pretrained(
        base_model,
        cfg.lora_path,
    )

    model.to(cfg.device)
    model.eval()

    print("[INFO] Model + LoRA ready. Example param device:",
          next(model.parameters()).device)
    return model, tok


# ---------- 主评测逻辑 ----------

def evaluate_with_lora(cfg: EvalConfig):
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    df = load_eval_data(cfg.eval_csv)
    print(f"[INFO] Loaded eval data: {len(df)} samples from {cfg.eval_csv}")

    out_dir = Path(cfg.output_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok = create_model_and_tokenizer(cfg)

    records = []

    for idx, row in df.iterrows():
        q = str(row["question"])
        gt = str(row["ground_truth"])
        prompt = build_prompt(q)

        enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
        in_ids = enc.input_ids.to(cfg.device)
        attn = enc.attention_mask.to(cfg.device)

        # 截断过长 prompt（保留末尾）
        if in_ids.shape[1] > cfg.max_prompt_tokens:
            in_ids = in_ids[:, -cfg.max_prompt_tokens:]
            attn = attn[:, -cfg.max_prompt_tokens:]

        cand = []
        with torch.no_grad():
            for k in range(cfg.k_samples):
                print(f"[DEBUG] idx={idx}, k={k} - generate", flush=True)
                gen_ids = model.generate(
                    input_ids=in_ids,
                    attention_mask=attn,
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=True,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    pad_token_id=tok.eos_token_id,
                )
                resp = tok.decode(
                    gen_ids[0][in_ids.shape[1]:],
                    skip_special_tokens=True,
                )

                comps = compute_reward(
                    resp,
                    gt,
                    cfg=RL_PHASE2_CONFIG,
                    return_components=True,
                )
                # 要求 reward_fn 返回这些字段
                assert (
                    "R_total" in comps
                    and "is_correct" in comps
                    and "has_valid_boxed" in comps
                    and "len_total" in comps
                ), "reward_fn.compute_reward(return_components=True) 必须返回 R_total/is_correct/has_valid_boxed/len_total"

                cand.append((resp, comps))

        best_resp, best = max(cand, key=lambda x: x[1]["R_total"])

        rec = {
            "index": idx,
            "question": q,
            "ground_truth": gt,
            "response": best_resp,
            "reward_total": best["R_total"],
            "is_correct": best["is_correct"],
            "has_valid_boxed": best["has_valid_boxed"],
            "response_len": best["len_total"],
        }
        if "task_name" in row:
            rec["task_name"] = row["task_name"]

        records.append(rec)

        if (idx + 1) % 10 == 0:
            print(f"[INFO] Processed {idx + 1}/{len(df)} samples")

    out_df = pd.DataFrame(records)
    out_df.to_csv(cfg.output_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved eval results to {cfg.output_csv}")

    acc = out_df["is_correct"].mean()
    boxed_rate = out_df["has_valid_boxed"].mean()
    avg_len = out_df["response_len"].mean()

    print(f"[SUMMARY] acc={acc:.3f}, boxed={boxed_rate:.3f}, len={avg_len:.1f}")


# ---------- 入口 ----------

if __name__ == "__main__":
    cfg = EvalConfig(
        base_model_path=str(
            PROJECT_ROOT / "models" / "DeepSeek-R1-Distill-Qwen-1.5B"
        ),
        # 注意：这里的 lora_path 按你的实际保存路径修改
        lora_path=str(
            PROJECT_ROOT
            / "rl_runs"
            / "DeepSeek-R1-Distill-Qwen-1.5B"
            / "afcurl_v2_gsm_math_k1_s42"
            / "final"
        ),
        eval_csv=str(
            PROJECT_ROOT
            / "data"
            / "splits"
            / "combo"
            / "gsm8k_math500_dev.csv"
        ),
        output_csv=str(
            PROJECT_ROOT
            / "eval_outputs"
            / "rl"
            / "afcurl_step200_gsm_math_dev_k4"
        ),
        k_samples=2,  # 和 baseline 一致
    )
    evaluate_with_lora(cfg)
