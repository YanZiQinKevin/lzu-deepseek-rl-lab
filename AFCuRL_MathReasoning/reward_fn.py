# reward_fn.py (v2)
# -*- coding: utf-8 -*-

"""
奖励函数（数学推理 / 竞赛数学）

R = α R_correct + β R_format + γ R_efficiency + δ R_closeness

- R_correct    : 是否答对（lenient 判题）
- R_format     : 是否给出合法的 \\boxed{number}
- R_efficiency : 长度惩罚（超过 len_free 线性扣分）
- R_closeness  : 数值接近奖励（可选，目前默认 0）

v2 变化：
- compute_reward(return_components=True) 会额外返回：
    is_correct, has_valid_boxed, len_total
用于 dev monitor / 行为分析 / 论文表格。
"""

from __future__ import annotations

import math
import re
import signal
from dataclasses import dataclass
from typing import Optional, Dict, Any

from lenient_grader import grade_answer_lenient as grade_fn


# ================= Timeout for grader ================= #

class _TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise _TimeoutError()


# ================= Reward Config ================= #

@dataclass
class RewardConfig:
    alpha_correct: float = 1.0
    beta_format: float = 0.5
    gamma_eff: float = 0.8
    delta_close: float = 0.3

    len_free: int = 4500
    len_lambda: float = 0.00025

    closeness_rel_tol: float = 0.05


DEFAULT_CONFIG = RewardConfig()

RL_TRAIN_CONFIG = RewardConfig(
    alpha_correct=1.0,
    beta_format=0.5,
    gamma_eff=0.5,
    delta_close=0.0,
    len_free=1500,
    len_lambda=0.0005,
)

RL_PHASE1_CONFIG = RewardConfig(
    alpha_correct = 0.3,   # 正确性有一点点权重，但不是主角
    beta_format   = 1.0,   # 强推“必须给出合法 boxed”
    gamma_eff     = 1.5,   # 长度惩罚和格式一样重要：越啰嗦越亏
    delta_close   = 0.0,   # 先不搞数值接近，Phase 2 再考虑

    len_free      = 2000,      # <=2000 字符不惩罚，允许少量自然语言
    len_lambda    = 0.0015,    # 超过的每个字符扣 0.0007
)

RL_PHASE2_CONFIG = RewardConfig(
    alpha_correct = 3,   # 正确性现在是 C 位
    beta_format   = 0.3,   # 只做轻量约束：别完全没 boxed 就行
    gamma_eff     = 0.2,   # 输出长度稍微看一下，别过长即可
    delta_close   = 0.0,

    len_free      = 2000,
    len_lambda    = 0.001,

)


# ================= Boxing / Parsing Utils ================= #

_BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")

def extract_valid_boxed_contents(text: str):
    """
    提取所有“合法”的 \\boxed{...} 内容：
    - 内容非空
    - 含至少一个数字
    """
    if not isinstance(text, str):
        return []
    matches = _BOXED_PATTERN.findall(text)
    results = []
    for m in matches:
        content = m.strip()
        if content and re.search(r"\d", content):
            results.append(content)
    return results

def has_valid_boxed(text: str) -> bool:
    return len(extract_valid_boxed_contents(text)) > 0

def parse_number_from_text(s: str) -> Optional[float]:
    """
    粗略解析整数 / 小数 / 分数。
    """
    if not isinstance(s, str):
        return None
    s = s.strip()
    s = s.replace(" ", "").replace("\\,", "").replace("\\ ", "")
    s = s.replace("(", "").replace(")", "")

    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                num = float(parts[0])
                den = float(parts[1])
                if den != 0:
                    return num / den
            except ValueError:
                pass

    try:
        return float(s)
    except ValueError:
        return None


# ================= Sub Rewards ================= #

def compute_R_correct(response: str, ground_truth: str) -> float:
    """
    Lenient 判题器判断是否答对。
    - 超时或异常 -> 判错
    """
    TIME_LIMIT = 3  # 秒

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)

    try:
        signal.alarm(TIME_LIMIT)
        is_correct = bool(grade_fn(response, ground_truth))
    except _TimeoutError:
        is_correct = False
    except Exception:
        is_correct = False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    return 1.0 if is_correct else 0.0


def compute_R_format(response: str) -> float:
    """
    格式 reward：
    - 有合法 boxed -> +1
    - 无 boxed -> -1
    """
    return 1.0 if has_valid_boxed(response) else -1.0


def compute_R_efficiency(response: str, cfg: RewardConfig) -> float:
    """
    长度惩罚（按字符数）
    """
    if not isinstance(response, str):
        return 0.0
    L = len(response)
    if L < 30:
        return -0.7
    if L <= cfg.len_free:
        return 0.0
    return -cfg.len_lambda * float(L - cfg.len_free)


def compute_R_closeness(response: str, ground_truth: str, cfg: RewardConfig) -> float:
    """
    可选数值接近 reward：
    - 取 response 最后一个 boxed content
    - 相对误差 <= tol -> 1 else 0
    """
    boxed_list = extract_valid_boxed_contents(response)
    if not boxed_list:
        return 0.0

    last_pred_str = boxed_list[-1]
    pred_val = parse_number_from_text(last_pred_str)
    gt_val = parse_number_from_text(ground_truth)

    if pred_val is None or gt_val is None:
        return 0.0

    denom = abs(gt_val) + 1e-8
    rel_err = abs(pred_val - gt_val) / denom

    return 1.0 if rel_err <= cfg.closeness_rel_tol else 0.0


# ================= Total Reward Interface ================= #

def compute_reward(
    response: str,
    ground_truth: str,
    cfg: RewardConfig = DEFAULT_CONFIG,
    return_components: bool = False,
) -> float | Dict[str, Any]:
    """
    计算总 reward。
    若 return_components=True，额外返回行为指标：
        is_correct, has_valid_boxed, len_total
    """

    R_c = compute_R_correct(response, ground_truth)
    R_f = compute_R_format(response)
    R_e = compute_R_efficiency(response, cfg)
    R_cl = compute_R_closeness(response, ground_truth, cfg)

    R_total = (
        cfg.alpha_correct * R_c
        + cfg.beta_format * R_f
        + cfg.gamma_eff * R_e
        + cfg.delta_close * R_cl
    )

    if not return_components:
        return float(R_total)

    is_corr = bool(R_c > 0.5)
    has_box = has_valid_boxed(response)
    L_total = len(response) if isinstance(response, str) else 0

    return {
        "R_total": float(R_total),
        "R_correct": float(R_c),
        "R_format": float(R_f),
        "R_efficiency": float(R_e),
        "R_closeness": float(R_cl),

        # ---- 行为 / 监控字段（v2 必需） ----
        "is_correct": is_corr,
        "has_valid_boxed": has_box,
        "len_total": int(L_total),
    }
