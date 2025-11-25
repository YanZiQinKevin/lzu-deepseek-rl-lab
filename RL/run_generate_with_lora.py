#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_generate_with_lora.py

用途：
- 用 DeepSeek-R1-Distill-Qwen-1.5B + 可选 LoRA adapter，
  对一份 CSV 数据集生成答案并做判题，方便比较 RL 前后效果。

输入 CSV 至少包含列：
  - question
  - ground_truth
（可选）task_name

输出 CSV 包含：
  - task_name（如果有）
  - question
  - ground_truth
  - response
  - is_correct
  - has_valid_boxed
  - response_len
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from lenient_grader import grade_answer_lenient as grade_fn
from reward_fn import has_valid_boxed  # 我们之前在 reward_fn.py 里已经写过


# ===================== 1. 在这里配置你的路径 ===================== #

# 基座模型（不带 LoRA 时就是原始模型，用于对比基线）
MODEL_PATH = "/mnt/d/OneDrive/兰州交大/博士课题-大模型/models/DeepSeek-R1-Distill-Qwen-1.5B"

# LoRA 适配器路径（例如 RL 训练完输出的 final 目录）
# 如果暂时只跑 base 模型，可以先随便写或留空
LORA_PATH = "/mnt/d/OneDrive/兰州交大/博士课题-大模型/LLMProject/rl_outputs/DeepSeek-R1-Distill-Qwen-1.5B/k1_fp16_trial/final"

# 是否加载 LoRA
USE_LORA = True  # 对比基线时改成 False，再改回 True

# 评测数据（CSV），至少要有 question / ground_truth 列
INPUT_CSV = "/mnt/d/OneDrive/兰州交大/博士课题-大模型/LLMProject/eval_outputs/DeepSeek-R1-Distill-Qwen-1.5B/train_gsm_math.csv"

# 输出结果 CSV
#eval_lora_gsm_math.csv
#eval_gsm_math
OUTPUT_CSV = "/mnt/d/OneDrive/兰州交大/博士课题-大模型/LLMProject/eval_outputs/DeepSeek-R1-Distill-Qwen-1.5B/eval_lora_gsm_math.csv"

# 生成参数
MAX_PROMPT_TOKENS = 384
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.0  # 评测时建议用 0（确定性）
TOP_P = 0.9

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===================== 2. 工具函数 ===================== #

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "question" not in df.columns or "ground_truth" not in df.columns:
        raise ValueError("INPUT_CSV 必须包含列: 'question' 和 'ground_truth'")
    # 保留额外列（例如 task_name），后面会一起写回
    return df.reset_index(drop=True)


def build_prompt(question: str) -> str:
    """
    与 RL 训练时使用的 prompt 模板保持一致，方便对比。
    """
    system = (
        "You are an expert competition mathematician. "
        "Directly give the final numerical answer in the format \\boxed{}. "
        "Do NOT show any reasoning steps."
    )
    prompt = f"{system}\n\nProblem:\n{question}\n"
    return prompt


def create_model_and_tokenizer(
    model_path: str,
    lora_path: Optional[str] = None,
    use_lora: bool = False,
):
    print(f"[INFO] Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] Loading base model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.to(DEVICE)

    if use_lora:
        if not lora_path or not Path(lora_path).exists():
            raise ValueError(f"USE_LORA=True 但找不到 LORA_PATH: {lora_path}")
        print(f"[INFO] Loading LoRA adapter from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model.to(DEVICE)

    model.eval()
    model.config.use_cache = True  # 推理时可以开启 cache 加速
    return model, tokenizer


# ===================== 3. 主评测逻辑 ===================== #

def run_eval():
    set_seed(SEED)

    df = load_data(INPUT_CSV)
    print(f"[INFO] Loaded {len(df)} samples from {INPUT_CSV}")

    model, tokenizer = create_model_and_tokenizer(
        model_path=MODEL_PATH,
        lora_path=LORA_PATH,
        use_lora=USE_LORA,
    )

    all_responses = []
    all_is_correct = []
    all_has_boxed = []
    all_lengths = []

    with torch.no_grad():
        for idx, row in df.iterrows():
            question = str(row["question"])
            gt = str(row["ground_truth"])

            prompt = build_prompt(question)

            enc = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = enc.input_ids.to(DEVICE)
            attention_mask = enc.attention_mask.to(DEVICE)

            if input_ids.shape[1] > MAX_PROMPT_TOKENS:
                input_ids = input_ids[:, -MAX_PROMPT_TOKENS:]
                attention_mask = attention_mask[:, -MAX_PROMPT_TOKENS:]

            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=(TEMPERATURE > 0.0),
                temperature=TEMPERATURE if TEMPERATURE > 0.0 else 1.0,
                top_p=TOP_P,
                pad_token_id=tokenizer.eos_token_id,
            )

            gen_text = tokenizer.decode(
                gen_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True,
            )

            # 判题
            try:
                correct = bool(grade_fn(gen_text, gt))
            except Exception:
                correct = False

            boxed_flag = has_valid_boxed(gen_text)
            length = len(gen_text)

            all_responses.append(gen_text)
            all_is_correct.append(int(correct))
            all_has_boxed.append(int(boxed_flag))
            all_lengths.append(length)

            if (idx + 1) % 20 == 0:
                print(
                    f"[INFO] Processed {idx+1}/{len(df)} samples. "
                    f"Current acc = {sum(all_is_correct)/(idx+1):.3f}"
                )

    # 写回到 DataFrame
    df["response"] = all_responses
    df["is_correct"] = all_is_correct
    df["has_valid_boxed"] = all_has_boxed
    df["response_len"] = all_lengths

    # 保存
    out_path = Path(OUTPUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved eval results to: {out_path}")

    # 简单整体准确率打印
    acc = sum(all_is_correct) / len(all_is_correct)
    boxed_rate = sum(all_has_boxed) / len(all_has_boxed)
    avg_len = sum(all_lengths) / len(all_lengths)
    print(
        f"[SUMMARY] acc={acc:.3f}, has_boxed_rate={boxed_rate:.3f}, "
        f"avg_response_len={avg_len:.1f}"
    )


if __name__ == "__main__":
    print(f"[INFO] USE_LORA={USE_LORA}, LORA_PATH={LORA_PATH}")
    run_eval()
