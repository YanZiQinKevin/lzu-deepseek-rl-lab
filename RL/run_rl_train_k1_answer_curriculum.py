#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_rl_train_k1_answer_curriculum.py

特点：
- 仍然是 K=1 REINFORCE + LoRA
- 加入：
    1) 答案区域加权：\boxed{} 之后的 token 权重大，前面的推理 token 权重小
    2) 两阶段 reward：
       - Phase 1：重点优化格式 + 长度
       - Phase 2：重点优化正确性
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

from reward_fn import (
    compute_reward,
    RL_PHASE1_CONFIG,
    RL_PHASE2_CONFIG,
)


# ================== 1. 配置 ================== #

MODEL_PATH = "/mnt/d/OneDrive/兰州交大/博士课题-大模型/models/DeepSeek-R1-Distill-Qwen-1.5B"
TRAIN_CSV = "/mnt/d/OneDrive/兰州交大/博士课题-大模型/LLMProject/eval_outputs/DeepSeek-R1-Distill-Qwen-1.5B/train_gsm_math_200.csv"
OUTPUT_DIR = "/mnt/d/OneDrive/兰州交大/博士课题-大模型/LLMProject/rl_outputs/DeepSeek-R1-Distill-Qwen-1.5B/k1_answer_curriculum"

@dataclass
class RLTrainConfig:
    model_path: str
    train_csv: str
    output_dir: str

    max_steps: int = 400       # 总步数
    phase1_steps: int = 200    # 前多少步用 Phase 1 配置

    logging_steps: int = 10
    save_steps: int = 200

    batch_size: int = 1
    max_prompt_tokens: int = 384
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

    lr: float = 5e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # 答案区域加权
    answer_weight: float = 1.0
    other_weight: float = 0.2

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


CFG = RLTrainConfig(
    model_path=MODEL_PATH,
    train_csv=TRAIN_CSV,
    output_dir=OUTPUT_DIR,
)


# ================== 2. 工具函数 ================== #

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
    system = (
        "You are an expert competition mathematician. "
        "Directly give the final numerical answer in the format \\boxed{}. "
        "Do NOT show any reasoning steps."
    )
    return f"{system}\n\nProblem:\n{question}\n"


def create_model_and_tokenizer(cfg: RLTrainConfig):
    print(f"[INFO] Loading tokenizer from {cfg.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] Loading base model from {cfg.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.to(cfg.device)

    # LoRA
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
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

    # 简化：暂时不启用 gradient checkpointing，避免 requires_grad 问题
    model.config.use_cache = False

    return model, tokenizer


def find_answer_start_token_idx(
    tokenizer,
    input_ids: torch.Tensor,
    full_text: str,
) -> Optional[int]:
    """
    通过暴力 decode 前缀的方式，大致找出第一个包含 '\\boxed{' 的 token index。
    返回 token index（0-based），如果找不到则返回 None。
    """
    if "\\boxed{" not in full_text:
        return None

    seq_len = input_ids.shape[1]
    for j in range(seq_len):
        # decode 到第 j 个 token（包含）
        text_prefix = tokenizer.decode(
            input_ids[0, : j + 1],
            skip_special_tokens=False,
        )
        if "\\boxed{" in text_prefix:
            return j  # token index
    return None


def calc_logprob_answer_weighted(
    model,
    tokenizer,
    prompt: str,
    response: str,
    device: str,
    max_prompt_tokens: int,
    answer_weight: float,
    other_weight: float,
) -> torch.Tensor:
    """
    计算 answer-focused 的 logprob：
    - prompt 部分不算 loss
    - response 中：
        - \boxed{...} 之后的 token 权重大 (answer_weight)
        - 之前的 response token 权重小 (other_weight)
    """
    # prompt 的 token 长度
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.to(device)
    prompt_len = prompt_ids.shape[1]

    # full text
    full_text = prompt + response
    enc = tokenizer(
        full_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    input_ids = enc.input_ids.to(device)
    attn_mask = enc.attention_mask.to(device)

    # 截断太长的 prompt（一般不会超）
    if prompt_len > max_prompt_tokens:
        # 在极端情况下截断 prompt，重新计算 prompt_len
        prompt_ids = prompt_ids[:, -max_prompt_tokens:]
        prompt_len = prompt_ids.shape[1]

    seq_len = input_ids.shape[1]

    # forward
    outputs = model(
        input_ids=input_ids,
        attention_mask=attn_mask,
    )
    logits = outputs.logits  # [1, seq_len, vocab]

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_attn = attn_mask[:, 1:]  # [1, seq_len-1]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logprobs = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.unsqueeze(-1),
    ).squeeze(-1)  # [1, seq_len-1]

    # 构造权重向量
    # 对应的 token index 是 t+1
    positions = torch.arange(
        token_logprobs.shape[1],
        device=device,
    )  # 0..seq_len-2
    token_indices = positions + 1  # 1..seq_len-1

    # prompt 最后一个 token index = prompt_len - 1
    prompt_boundary = prompt_len - 1

    # 找答案起始 token index
    full_decoded = tokenizer.decode(
        input_ids[0],
        skip_special_tokens=False,
    )
    ans_start_tok = find_answer_start_token_idx(
        tokenizer=tokenizer,
        input_ids=input_ids,
        full_text=full_decoded,
    )

    weights = torch.zeros_like(token_logprobs)

    # response token：token_index > prompt_boundary
    resp_mask = token_indices > prompt_boundary
    weights[0, resp_mask] = other_weight

    # 如果找到答案起点，则从该 token 开始加大权重
    if ans_start_tok is not None:
        ans_mask = token_indices >= ans_start_tok
        weights[0, ans_mask] = answer_weight

    # 乘上 attention_mask 和权重
    token_logprobs = token_logprobs * shift_attn * weights

    logprob_seq = token_logprobs.sum()
    return logprob_seq


# ================== 3. 主训练循环 ================== #

def train_k1_answer_curriculum(cfg: RLTrainConfig):
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
    alpha_rm = 0.05  # 运行平均平滑系数

    for step in range(cfg.max_steps):
        model.train()

        batch_indices = random.sample(range(len(df)), cfg.batch_size)
        batch = df.loc[batch_indices]

        rewards = []
        logprobs_info = []

        # 1) 采样 + 计算 reward（不求梯度）
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
                    attention_mask=attention_mask,
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

                # 选择当前阶段的 reward 配置
                if global_step < cfg.phase1_steps:
                    reward_cfg = RL_PHASE1_CONFIG
                    phase = 1
                else:
                    reward_cfg = RL_PHASE2_CONFIG
                    phase = 2

                comps = compute_reward(
                    response=gen_text,
                    ground_truth=gt,
                    cfg=reward_cfg,
                    return_components=True,
                )
                R = comps["R_total"]

                rewards.append(R)
                logprobs_info.append((prompt, gen_text, R, phase))

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=cfg.device)
        avg_reward = rewards_tensor.mean().item()

        # 更新运行平均
        if running_R_mean is None:
            running_R_mean = avg_reward
        else:
            running_R_mean = (1 - alpha_rm) * running_R_mean + alpha_rm * avg_reward

        # 2) 反向传播
        optimizer.zero_grad()
        total_loss = 0.0
        baseline = 0.0  # 当前阶段 baseline 仍然设为 0，有需要可后续改成 EMA baseline

        for prompt, resp, R, phase in logprobs_info:
            adv = R - baseline
            logprob_seq = calc_logprob_answer_weighted(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                response=resp,
                device=cfg.device,
                max_prompt_tokens=cfg.max_prompt_tokens,
                answer_weight=cfg.answer_weight,
                other_weight=cfg.other_weight,
            )
            loss = -(adv * logprob_seq) / cfg.batch_size
            loss.backward()
            total_loss += float(loss.detach().cpu().item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        global_step += 1

        if global_step % cfg.logging_steps == 0:
            print(
                f"[Step {global_step}] "
                f"loss={total_loss:.4f}  "
                f"R_mean={avg_reward:.4f}  "
                f"R_running={running_R_mean:.4f}  "
                f"phase={'1' if global_step <= cfg.phase1_steps else '2'}"
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


if __name__ == "__main__":
    print("[INFO] RL config:", CFG)
    train_k1_answer_curriculum(CFG)
