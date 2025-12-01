#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_rl_train_k1_afcurl_v2.py

AF-CuRL v2 (Answer-Focused Curriculum REINFORCE for Low-Resource Math Reasoning)

核心：
- Base: DeepSeek-R1-Distill-Qwen-1.5B
- LoRA 适配器（≈0.5% 参数）
- K=1 REINFORCE 训练（低资源）
- Answer-focused token reweighting
- Reward curriculum: Phase1(格式+长度) -> Phase2(正确性主导)
- 训练中 dev 监控（K=4 best-of-4），输出 paper-ready 曲线数据

运行：
    python scripts/rl/run_rl_train_k1_afcurl_v2.py
"""

from __future__ import annotations

import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datetime import datetime
import time


# ---- 工程根目录 ----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reward_fn import (
    compute_reward,
    RL_PHASE1_CONFIG,
    RL_PHASE2_CONFIG,
)
print("[DEBUG] reward_fn loaded from:", sys.modules["reward_fn"].__file__)
# ================== 1. 配置 ================== #

@dataclass
class RLTrainConfig:
    model_path: str
    train_csv: str
    dev_csv: str
    output_dir: str

    max_steps: int = 200
    phase1_steps: int = 0

    logging_steps: int = 5
    save_steps: int = 200

    batch_size: int = 4
    max_prompt_tokens: int = 384
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    lr: float = 2e-5#5e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Answer-focused 权重
    answer_weight: float = 1.0
    other_weight: float = 0.2

    # EMA baseline / smoothing
    ema_alpha: float = 0.05  # reward running mean smoothing
    use_ema_baseline: bool = True

    # dev monitor
    dev_k_samples = 1  # 先用 2 次采样评估，差不多能看趋势
    dev_max_samples = 15  # 每次只随机抽 50 题
    dev_eval_every = 100  # 保持 100 step 一评

    seed: int = 42
    clip_grad_norm: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


MODEL_PATH = PROJECT_ROOT / "models" / "DeepSeek-R1-Distill-Qwen-1.5B"
TRAIN_CSV  = PROJECT_ROOT / "data" / "splits" / "combo" / "gsm8k_math500_train_rl.csv"
DEV_CSV    = PROJECT_ROOT / "data" / "splits" / "combo" / "gsm8k_math500_dev.csv"
#OUTPUT_DIR = PROJECT_ROOT / "rl_runs" / "DeepSeek-R1-Distill-Qwen-1.5B" / "afcurl_v2_gsm_math_k1_s42"j
#OUTPUT_DIR = PROJECT_ROOT / "rl_runs" / "DeepSeek-R1-Distill-Qwen-1.5B" / "plain_rl_gsm_math_k1_s42" #plain_rl
OUTPUT_DIR = PROJECT_ROOT / "rl_runs" / "DeepSeek-R1-Distill-Qwen-1.5B" / "answer_only_gsm_math_k1_s42"
#OUTPUT_DIR = PROJECT_ROOT / "rl+runs" / "DeepSeek-R1-Distill-Qwen-1.5B" / "curriculum_only_gsm_math_k1_s42"

CFG = RLTrainConfig(
    model_path=str(MODEL_PATH),
    train_csv=str(TRAIN_CSV),
    dev_csv=str(DEV_CSV),
    output_dir=str(OUTPUT_DIR),
)

# ================== 2. 工具函数 ================== #
def get_time():
    timestamp = time.time()
    dt = datetime.fromtimestamp(timestamp)
    return  dt.strftime("%Y-%m-%d %H:%M:%S")
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "question" not in df.columns or "ground_truth" not in df.columns:
        raise ValueError("CSV 必须包含列: 'question' 和 'ground_truth'")
    keep_cols = ["question", "ground_truth"]
    if "task_name" in df.columns:
        keep_cols.insert(0, "task_name")
    return df[keep_cols].reset_index(drop=True)

def build_prompt(question: str) -> str:
    system = (
        "You are an expert competition mathematician. "
        "Think carefully, but keep your reasoning concise (no more than 5 short steps)."
        "Then give the final numerical answer in the format \boxed{...}."
        "Directly give the final numerical answer in the format \\boxed{}. "
        "Do NOT output anything after the boxed answer."
    )
    return f"{system}\n\nProblem:\n{question}\n"

def create_model_and_tokenizer(cfg: RLTrainConfig):
    print(f"[INFO] Loading tokenizer from {cfg.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[INFO] Loading base model from {cfg.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    model.to(cfg.device)

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

    model.config.use_cache = False
    return model, tokenizer

def truncate_prompt_ids(tokenizer, prompt: str, max_prompt_tokens: int, device: str):
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.to(device)
    attn_mask = enc.attention_mask.to(device)
    if input_ids.shape[1] > max_prompt_tokens:
        input_ids = input_ids[:, -max_prompt_tokens:]
        attn_mask = attn_mask[:, -max_prompt_tokens:]
    return input_ids, attn_mask

def find_answer_start_token_idx(tokenizer, input_ids: torch.Tensor) -> Optional[int]:
    """
    更稳妥的实现：直接在 full decoded 中找 \\boxed{ 的字符位置，
    然后逐 token 前缀 decode 直到覆盖该位置。
    若无 boxed，返回 None。
    """
    full_decoded = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    pos = full_decoded.find("\\boxed{")
    if pos < 0:
        return None

    seq_len = input_ids.shape[1]
    for j in range(seq_len):
        prefix = tokenizer.decode(input_ids[0, : j + 1], skip_special_tokens=False)
        if "\\boxed{" in prefix:
            return j
    return None

def calc_logprob_answer_weighted(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    prompt_attn: torch.Tensor,
    response: str,
    device: str,
    answer_weight: float,
    other_weight: float,
) -> torch.Tensor:
    """
    计算 Answer-focused 的 logprob：
    - prompt 不计 loss
    - response 中：
        - \\boxed{...} 之后 token 权重大（answer_weight）
        - 之前 response token 权重小（other_weight）
    - 这里与 generate 的 prompt 截断完全一致（用 prompt_ids 作为边界）
    - 最后按有效权重 token 数 normalize，防梯度爆炸
    """
    prompt_len = prompt_ids.shape[1]

    # full input = truncated_prompt + response
    truncated_prompt_text = tokenizer.decode(prompt_ids[0], skip_special_tokens=False)
    full_text = truncated_prompt_text + response

    enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.to(device)
    attn_mask = enc.attention_mask.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = outputs.logits  # [1, seq_len, vocab]

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_attn = attn_mask[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logprobs = torch.gather(
        log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)  # [1, L-1]

    positions = torch.arange(token_logprobs.shape[1], device=device)
    token_indices = positions + 1  # 对应 input_ids 的真实 token index

    prompt_boundary = prompt_len - 1

    ans_start_tok = find_answer_start_token_idx(tokenizer, input_ids)

    weights = torch.zeros_like(token_logprobs)
    resp_mask = token_indices > prompt_boundary
    weights[0, resp_mask] = other_weight

    if ans_start_tok is not None:
        ans_mask = token_indices >= ans_start_tok
        weights[0, ans_mask] = answer_weight

    weighted_logprobs = token_logprobs * shift_attn * weights

    # normalize by number of effective tokens
    denom = weights.sum() + 1e-8
    logprob_seq = weighted_logprobs.sum() / denom
    return logprob_seq

def check_lora_nan(model) -> bool:
    """返回是否检测到 LoRA NaN/Inf。"""
    for n, p in model.named_parameters():
        if "lora" in n:
            if torch.isnan(p).any() or torch.isinf(p).any():
                print(f"[WARN] NaN/Inf detected in {n}")
                return True
    return False

# ================== 3. dev 评估 ================== #

def evaluate_on_dev(model, tokenizer, dev_df: pd.DataFrame, cfg: RLTrainConfig):
    """
    在 dev 上做 K = cfg.dev_k_samples 的 best-of-K 评估，返回：
        acc, boxed_rate, avg_len, n

    注意：
      - 每次最多评 cfg.dev_max_samples 条样本
      - compute_reward(return_components=True) 需要至少返回：
            is_correct, has_valid_boxed, len_total
    """
    model.eval()

    # ---- 1) 子采样：每次最多评 dev_max_samples 条 ----
    if len(dev_df) == 0:
        return {"acc": 0.0, "boxed_rate": 0.0, "avg_len": 0.0, "n": 0}

    sub_df = dev_df.sample(
        n=min(cfg.dev_max_samples, len(dev_df)),
        random_state=cfg.seed,   # 固定子集，方便复现；想要每次不同可以换成 None
    ).reset_index(drop=True)

    is_correct_list, boxed_list, len_list = [], [], []

    with torch.no_grad():
        for _, row in sub_df.iterrows():
            q = str(row["question"])
            gt = str(row["ground_truth"])
            prompt = build_prompt(q)

            prompt_ids, prompt_attn = truncate_prompt_ids(
                tokenizer, prompt, cfg.max_prompt_tokens, cfg.device
            )

            # ---- 2) K 次采样，取 reward 最大的一条（K=1 时就只算一次） ----
            candidates = []
            for _ in range(cfg.dev_k_samples):
                gen_ids = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_attn,
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=True,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    pad_token_id=tokenizer.eos_token_id,

                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    top_k=50,
                )
                resp = tokenizer.decode(
                    gen_ids[0][prompt_ids.shape[1]:],
                    skip_special_tokens=True,
                )

                comps = compute_reward(
                    resp,
                    gt,
                    cfg=RL_PHASE2_CONFIG,
                    return_components=True,
                )

                # 这些字段必须存在，否则说明 reward_fn 还没补齐
                assert (
                    "is_correct" in comps
                    and "has_valid_boxed" in comps
                    and "len_total" in comps
                ), "reward_fn.compute_reward(return_components=True) 必须返回 is_correct/has_valid_boxed/len_total"

                candidates.append((resp, comps))

            # best-of-K（K=1 时就是那一条）
            _, best_comps = max(candidates, key=lambda x: x[1]["R_total"])

            is_correct_list.append(bool(best_comps["is_correct"]))
            boxed_list.append(bool(best_comps["has_valid_boxed"]))
            len_list.append(float(best_comps["len_total"]))

    model.train()

    n = len(is_correct_list)
    acc = float(sum(is_correct_list) / n) if n > 0 else 0.0
    boxed_rate = float(sum(boxed_list) / n) if n > 0 else 0.0
    avg_len = float(sum(len_list) / n) if n > 0 else 0.0

    return {
        "acc": acc,
        "boxed_rate": boxed_rate,
        "avg_len": avg_len,
        "n": int(n),
    }


# ================== 4. 主训练循环 ================== #

def train_k1_afcurl_v2(cfg: RLTrainConfig):
    set_seed(cfg.seed)

    train_df = load_data(cfg.train_csv)
    dev_df = load_data(cfg.dev_csv)
    print(f"[INFO] Loaded train data: {len(train_df)} samples")
    print(f"[INFO] Loaded dev data:   {len(dev_df)} samples")

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

    # dev monitor CSV
    monitor_path = output_dir / "dev_monitor.csv"
    if not monitor_path.exists():
        with open(monitor_path, "w", encoding="utf-8") as f:
            f.write("step,phase,acc,boxed_rate,avg_len,n,wall_time\n")

    global_step = 0
    running_R_mean = None
    start_time = time.time()

    for step in range(cfg.max_steps):
        model.train()

        batch_indices = random.sample(range(len(train_df)), cfg.batch_size)
        batch = train_df.loc[batch_indices]

        rewards: List[float] = []
        logprobs_info: List[Tuple[torch.Tensor, torch.Tensor, str, float, int]] = []

        # 1) 采样 + reward（不求梯度）
        with torch.no_grad():
            for _, row in batch.iterrows():
                q = str(row["question"])
                gt = str(row["ground_truth"])
                prompt = build_prompt(q)

                prompt_ids, prompt_attn = truncate_prompt_ids(
                    tokenizer, prompt, cfg.max_prompt_tokens, cfg.device
                )

                gen_ids = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_attn,
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=True,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    pad_token_id=tokenizer.eos_token_id,

                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    top_k=50,
                )
                resp = tokenizer.decode(
                    gen_ids[0][prompt_ids.shape[1]:],
                    skip_special_tokens=True,
                )

                if global_step < cfg.phase1_steps:
                    reward_cfg = RL_PHASE1_CONFIG
                    phase = 1
                else:
                    reward_cfg = RL_PHASE2_CONFIG
                    phase = 2

                comps = compute_reward(
                    response=resp,
                    ground_truth=gt,
                    cfg=reward_cfg,
                    return_components=True,
                )
                R = float(comps["R_total"])

                rewards.append(R)
                logprobs_info.append((prompt_ids, prompt_attn, resp, R, phase))

                # debug: 保存最后一个 sample
                debug_info = {
                    "global_step": int(global_step),
                    "phase": int(phase),
                    "question": q,
                    "ground_truth": gt,
                    "response": resp,
                    "reward_components": comps,
                }
                with open(output_dir / "debug_lastTrain_sample.json", "w", encoding="utf-8") as f:
                    json.dump(debug_info, f, ensure_ascii=False, indent=2)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=cfg.device)
        avg_reward = rewards_tensor.mean().item()

        # EMA reward mean
        if running_R_mean is None:
            running_R_mean = avg_reward
        else:
            running_R_mean = (1 - cfg.ema_alpha) * running_R_mean + cfg.ema_alpha * avg_reward

        # 2) 反向传播
        optimizer.zero_grad()
        total_loss = 0.0

        baseline = running_R_mean if cfg.use_ema_baseline else 0.0

        for prompt_ids, prompt_attn, resp, R, phase in logprobs_info:
            adv = R - baseline
            logprob_seq = calc_logprob_answer_weighted(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                prompt_attn=prompt_attn,
                response=resp,
                device=cfg.device,
                answer_weight=cfg.answer_weight,
                other_weight=cfg.other_weight,
            )
            loss = -(adv * logprob_seq) / cfg.batch_size
            loss.backward()
            total_loss += float(loss.detach().cpu().item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
        optimizer.step()

        global_step += 1

        # 3) NaN/Inf 早停保护
        if check_lora_nan(model):
            print("[ERROR] LoRA exploded (NaN/Inf). Saving emergency checkpoint and stopping.")
            bad_path = output_dir / f"nan_step_{global_step}"
            bad_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(bad_path)
            break

        # 4) logging
        if global_step % cfg.logging_steps == 0:
            print(
                f"[Step {global_step}] "
                f"loss={total_loss:.4f}  "
                f"R_mean={avg_reward:.4f}  "
                f"R_running={running_R_mean:.4f}  "
                f"phase={'1' if global_step <= cfg.phase1_steps else '2'} "
                f"time at {get_time()}"
            )

        # 5) save lora
        if global_step % cfg.save_steps == 0:
            save_path = output_dir / f"step_{global_step}"
            save_path.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Saving LoRA adapter to {save_path}")
            model.save_pretrained(save_path)

        # 6) dev monitor
        if global_step % cfg.dev_eval_every == 0:
            print(f"[DEBUG] Enter dev eval at step={global_step}, time at {get_time()}", flush=True)
            metrics = evaluate_on_dev(model, tokenizer, dev_df, cfg)
            print(f"[DEBUG] Leave dev eval at step={global_step}, time at {get_time()}", flush=True)
            wall = time.time() - start_time
            phase_now = 1 if global_step < cfg.phase1_steps else 2

            with open(monitor_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{global_step},{phase_now},{metrics['acc']},"
                    f"{metrics['boxed_rate']},{metrics['avg_len']},"
                    f"{metrics['n']},{wall}\n"
                )

            print(
                f"[DEV] step={global_step} "
                f"acc={metrics['acc']:.3f} "
                f"boxed={metrics['boxed_rate']:.3f} "
                f"len={metrics['avg_len']:.1f} "
                f"(n={metrics['n']})"
            )

    # final save
    final_path = output_dir / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Training finished. Saving final LoRA adapter to {final_path}")
    model.save_pretrained(final_path)


if __name__ == "__main__":
    print("[INFO] RL config:", CFG)
    train_k1_afcurl_v2(CFG)
