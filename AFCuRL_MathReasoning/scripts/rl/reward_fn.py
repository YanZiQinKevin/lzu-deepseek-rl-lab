# reward_fn.py
# -*- coding: utf-8 -*-

"""
定义针对数学推理任务的奖励函数，用于后续 RL 训练与离线分析。

R = α R_correct + β R_format + γ R_efficiency + δ R_closeness

- R_correct    : 是否答对（基于现有判题器）
- R_format     : 是否给出合法的 \boxed{number}
- R_efficiency : 长度惩罚（过长输出线性扣分）
- R_closeness  : 数值接近奖励（可选）

后续 RL 训练时可以直接调用 compute_reward(...)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any

from lenient_grader import grade_answer_lenient as grade_fn

import signal

class _TimeoutError(Exception):
    pass

def _timeout_handler(signum, frame):
    raise _TimeoutError()
# ----------------- 配置超参数（后续可以在一个地方统一管理） ----------------- #

@dataclass
class RewardConfig:
    alpha_correct: float = 1.0   # 正确与否的权重
    beta_format: float = 0.5     # 输出格式 / 答题行为 reward 的权重
    gamma_eff: float = 0.8       # 长度惩罚权重
    delta_close: float = 0.3     # 数值接近奖励权重

    len_free: int = 4500         # L0: 不惩罚的最大字符长度
    len_lambda: float = 0.00025  # λ: 每超过 1 个字符扣多少 reward

    closeness_rel_tol: float = 0.05  # 数值相对误差阈值，例如 5%


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
    alpha_correct=0.2,   # 正确性先弱一点
    beta_format=1.0,     # 强调答题格式
    gamma_eff=0.8,       # 强调长度惩罚
    delta_close=0.0,     # 先不管 closeness

    len_free=1500,
    len_lambda=0.0005,
)

RL_PHASE2_CONFIG = RewardConfig(
    alpha_correct=1.0,   # 正确性变成主角
    beta_format=0.5,     # 适当保持格式约束
    gamma_eff=0.3,       # 长度约束降一点
    delta_close=0.0,     # 后面如果想加 closeness 再说

    len_free=1500,
    len_lambda=0.0005,
)

# ----------------- 工具函数 ----------------- #

_BOXED_PATTERN = re.compile(r"\\boxed\{([^}]*)\}")


def extract_valid_boxed_contents(text: str):
    """
    提取所有“合法”的 \boxed{...} 内容：
    - 内容非空
    - 含至少一个数字

    返回 list[str]，按出现顺序排列。
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
    """是否至少包含一个合法的 \boxed{...}。"""
    return len(extract_valid_boxed_contents(text)) > 0


def parse_number_from_text(s: str) -> Optional[float]:
    """
    尝试从字符串中解析一个数字，用于“数值接近奖励”。
    粗略处理几种常见形式：
    - 纯整数： "123"
    - 小数：   "3.14"
    - 分数：   "123/456"
    - 去除空格、逗号等简单符号

    若解析失败，返回 None。
    """
    if not isinstance(s, str):
        return None

    # 去掉空格和常见的 LaTeX 符号
    s = s.strip()
    s = s.replace(" ", "")
    s = s.replace("\\,", "")
    s = s.replace("\\ ", "")
    # 去掉多余的括号
    s = s.replace("(", "").replace(")", "")

    # 尝试分数 a/b
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

    # 尝试直接转 float
    try:
        return float(s)
    except ValueError:
        return None


# ----------------- 各子 reward 的计算 ----------------- #

def compute_R_correct(response: str, ground_truth: str) -> float:
    """
    利用现有判题器判断是否答对。
    返回 1.0 或 0.0。
    """
    """
        利用现有判题器判断是否答对。
        - 正常情况下：grade_fn 和以前一样严格判分
        - 如果判题超过 T 秒（例如 3 秒），就当作错误（0.0），防止训练卡死
        """
    TIME_LIMIT = 3  # 判题超时阈值（秒），可以根据实际情况调成 2 或 3

    # 安装超时 handler（只需在主线程使用）
    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)

    try:
        signal.alarm(TIME_LIMIT)
        is_correct = bool(grade_fn(response, ground_truth))
    except _TimeoutError:
        # 判题超时：视为错误
        is_correct = False
    except Exception:
        # 判题器内部任何异常，也视为错误
        is_correct = False
    finally:
        # 关闭 alarm，并恢复原来的 handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    return 1.0 * is_correct



def compute_R_format(response: str) -> float:
    """
    格式 / 答题行为 reward:
    - 至少出现一个合法 \boxed{number} : +1
    - 否则 : -1
    """
    if has_valid_boxed(response):
        return 1.0
    else:
        return -1.0


def compute_R_efficiency(response: str, cfg: RewardConfig) -> float:
    """
    长度惩罚：
    - 长度 L <= len_free 时不惩罚
    - L > len_free 时线性惩罚: -λ * (L - L0)
    """
    if not isinstance(response, str):
        return 0.0

    L = len(response)
    if L <= cfg.len_free:
        return 0.0
    else:
        return -cfg.len_lambda * float(L - cfg.len_free)


def compute_R_closeness(response: str, ground_truth: str, cfg: RewardConfig) -> float:
    """
    数值接近奖励（可选）：
    - 从 response 中取“最后一个合法 \boxed{content}”，解析为 pred
    - 从 ground_truth 解析出 gt
    - 若解析成功且相对误差 <= closeness_rel_tol，则给一个小奖励
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

    if rel_err <= cfg.closeness_rel_tol:
        return 1.0  # 先给一个基准 1，后面乘以 delta_close
    else:
        return 0.0


# ----------------- 总 reward 计算接口 ----------------- #

def compute_reward(
    response: str,
    ground_truth: str,
    cfg: RewardConfig = DEFAULT_CONFIG,
    return_components: bool = False,
) -> float | Dict[str, Any]:
    """
    计算一条样本的总 reward（以及各子项）。

    参数:
        response       : 模型输出
        ground_truth   : 标准答案（通常是 \boxed{...} 形式）
        cfg            : RewardConfig 超参数
        return_components : 若为 True，则返回包含各子 component 的 dict

    返回:
        若 return_components=False: 仅返回 total_reward: float
        若 return_components=True: 返回 dict，例如:
            {
                "R_total": ...,
                "R_correct": ...,
                "R_format": ...,
                "R_efficiency": ...,
                "R_closeness": ...,
            }
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
        return R_total

    return {
        "R_total": R_total,
        "R_correct": R_c,
        "R_format": R_f,
        "R_efficiency": R_e,
        "R_closeness": R_cl,
    }
