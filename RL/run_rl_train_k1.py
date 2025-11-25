#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_rl_train_k1.py  (no 4bit, config 写在代码里，方便 PyCharm 直接运行)

功能：
- 使用 DeepSeek-R1-Distill-Qwen-1.5B + LoRA（全模型 bfloat16 / float16）做一个 K=1 的简化 RL 循环
- 算法：单样本 REINFORCE
- 数据：从一个 CSV 中读取 `question` 和 `ground_truth` 两列

使用方法：
- 直接在 PyCharm 打开本文件
- 修改顶部的 MODEL_PATH / TRAIN_CSV / OUTPUT_DIR
- 点击 Run 即可
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import LoraConfig, get_peft_model

from reward_fn import compute_reward, RL_TRAIN_CONFIG as REWARD_CFG


# ===================== 1. 在这里填写你的本地路径 ===================== #

# TODO: 把这三个路径改成你自己的
MODEL_PATH = "/mnt/d/OneDrive/兰州交大/博士课题-大模型/models/DeepSeek-R1-Distill-Qwen-1.5B"
TRAIN_CSV = "/mnt/d/OneDrive/兰州交大/博士课题-大模型/LLMProject/eval_outputs/DeepSeek-R1-Distill-Qwen-1.5B/train_gsm_math.csv"
OUTPUT_DIR = "/mnt/d/OneDrive/兰州交大/博士课题-大模型/LLMProject/rl_outputs/DeepSeek-R1-Distill-Qwen-1.5B/k1_fp16_trial"


# ===================== 2. 训练配置 ===================== #

@dataclass
class RLTrainConfig:
    model_path: str
    train_csv: str
    output_dir: str

    max_steps: int = 200          # 先试 200 步，看显存和效果
    logging_steps: int = 10
    save_steps: int = 100

    batch_size: int = 1           # 强烈建议 1（8G 显存）
    max_prompt_tokens: int = 384  # prompt 最长 token 数
    max_new_tokens: int = 256     # 生成的最大新 token 数
    temperature: float = 0.7
    top_p: float = 0.9

    lr: float = 5e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    # LoRA 超参
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = RLTrainConfig(
    model_path=MODEL_PATH,
    train_csv=TRAIN_CSV,
    output_dir=OUTPUT_DIR,
)


# ===================== 3. 工具函数 ===================== #

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_train_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "question" not in df.columns or "ground_truth" not in df.columns:
        raise ValueError("train_csv 必须包含列: 'question' 和 'ground_truth'")
    keep_cols = ["question", "ground_truth"]
    if "task_name" in df.columns:
        keep_cols.append("task_name")
    return df[keep_cols].reset_index(drop=True)


def build_prompt(question: str) -> str:
    """
    简单 prompt 模板。后面可以按需改。
    """
    system = (
        "You are an expert competition mathematician. "
        "Directly give the final numerical answer in the format \\boxed{}. "
        "Do NOT show any reasoning steps."
    )
    prompt = f"{system}\n\nProblem:\n{question}\n"
    return prompt


def create_model_and_tokenizer(cfg: RLTrainConfig):
    print(f"[INFO] Loading tokenizer from {cfg.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path,
        use_fast=False,
    )


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] Loading model (fp16/bf16) from {cfg.model_path}")
    # bf16 如果你的显卡支持（30 系列通常支持），否则可以改成 torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.to(cfg.device)

    # 配置 LoRA，只在注意力和 FFN 上加低秩适配器
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)

    print("[INFO] Model loaded. Trainable parameters:")
    model.print_trainable_parameters()

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    # 减少显存：
    #model.gradient_checkpointing_enable()
    model.config.use_cache = False  # 训练时必须关掉 cache

    return model, tokenizer


def calc_logprob_of_response(
    model,
    tokenizer,
    prompt: str,
    response: str,
    device: str,
    max_prompt_tokens: int,
) -> torch.Tensor:
    """
    给定 prompt 和 response，计算 log p(response | prompt)。
    返回一个标量 tensor（sum over generated tokens）。
    """
    with torch.no_grad():
        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.to(device)

    if prompt_ids.shape[1] > max_prompt_tokens:
        prompt_ids = prompt_ids[:, -max_prompt_tokens:]

    full_text = prompt + response
    enc = tokenizer(
        full_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = enc.input_ids.to(device)
    attn_mask = enc.attention_mask.to(device)

    prompt_len = prompt_ids.shape[1]

    outputs = model(
        input_ids=input_ids,
        attention_mask=attn_mask,
    )
    logits = outputs.logits  # [1, seq_len, vocab]

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_attn = attn_mask[:, 1:]

    start = max(prompt_len - 1, 0)

    shift_logits = shift_logits[:, start:, :].contiguous()
    shift_labels = shift_labels[:, start:].contiguous()
    shift_attn = shift_attn[:, start:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    target_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)

    if shift_attn is not None:
        target_log_probs = target_log_probs * shift_attn

    logprob_seq = target_log_probs.sum()
    return logprob_seq


# ===================== 4. 主训练循环 ===================== #

def train_k1_rl(cfg: RLTrainConfig):
    set_seed(cfg.seed)

    df = load_train_data(cfg.train_csv)
    print(f"[INFO] Loaded train data: {len(df)} samples from {cfg.train_csv}")

    model, tokenizer = create_model_and_tokenizer(cfg)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    running_R_mean = None
    alpha = 0.05  # 平滑系数


    for step in range(cfg.max_steps):
        model.train()

        batch_indices = random.sample(range(len(df)), cfg.batch_size)
        batch = df.loc[batch_indices]

        rewards = []
        logprobs_info = []

        # 1) 生成 + 计算 reward（不求梯度）
        with torch.no_grad():
            for _, row in batch.iterrows():
                q = str(row["question"])
                gt = str(row["ground_truth"])
                prompt = build_prompt(q)

                enc = tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                input_ids = enc.input_ids.to(cfg.device)
                attention_mask = enc.attention_mask.to(cfg.device)

                if input_ids.shape[1] > cfg.max_prompt_tokens:
                    input_ids = input_ids[:, -cfg.max_prompt_tokens:]
                    attention_mask = attention_mask[:, -cfg.max_prompt_tokens:]

                gen_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,  # 现在 attention_mask 已经定义好了
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=True,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )

                gen_text = tokenizer.decode(
                    gen_ids[0][input_ids.shape[1]:],
                    skip_special_tokens=True,
                )

                comps = compute_reward(
                    response=gen_text,
                    ground_truth=gt,
                    cfg=REWARD_CFG,
                    return_components=True,
                )
                R = comps["R_total"]
                rewards.append(R)
                logprobs_info.append((prompt, gen_text, R))

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=cfg.device)
        avg_reward = rewards_tensor.mean().item()

        #baseline = rewards_tensor.mean().item()
        baseline = 0.0

        # 2) 反向：对 log p(y|x) 乘 (R - baseline)
        optimizer.zero_grad()
        total_loss = 0.0


        for prompt, resp, R in logprobs_info:
            adv = R - baseline
            logprob_seq = calc_logprob_of_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                response=resp,
                device=cfg.device,
                max_prompt_tokens=cfg.max_prompt_tokens,
            )
            loss = -(adv * logprob_seq) / cfg.batch_size
            loss.backward()
            total_loss += float(loss.detach().cpu().item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        global_step += 1


        if running_R_mean is None:
            running_R_mean = avg_reward
        else:
            running_R_mean = (1 - alpha) * running_R_mean + alpha * avg_reward



        if global_step % cfg.logging_steps == 0:

            std_reward = rewards_tensor.std().item() if cfg.batch_size > 1 else 0.0
            print(
                f"[Step {global_step}] "
                f"loss={total_loss:.4f}  "
                f"R_mean={avg_reward:.4f}  R_std={std_reward:.4f}  "
                f"baseline={baseline:.4f} "
                f"R_running={running_R_mean:.4f}"
            )

        if global_step % cfg.save_steps == 0:
            save_path = output_dir / f"step_{global_step}"
            save_path.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Saving LoRA adapter to {save_path}")
            model.save_pretrained(save_path)

    final_path = output_dir / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Training finished. Saving final LoRA adapter to {final_path}")
    model.save_pretrained(final_path)


# ===================== 5. 直接运行 ===================== #

if __name__ == "__main__":
    print("[INFO] RL config:", CFG)
    train_k1_rl(CFG)
