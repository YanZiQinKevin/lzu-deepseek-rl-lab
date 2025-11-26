from answer_checker import grade_answer_verl, extract_answer
import re

def fallback_extract_numeric(passage: str):
    # 去掉 LaTeX 和空白后，从后往前找数字
    txt = passage.replace("\\boxed{}", "")
    # 找最后一个类似 123 或 123.45 的数字
    matches = re.findall(r"-?\d+(?:\.\d+)?", txt)
    if not matches:
        return None
    return matches[-1]

def grade_answer_lenient(solution_str, ground_truth):
    # 先按原来的严格方式试一遍
    from answer_checker import extract_answer
    strict_ans = extract_answer(solution_str)
    if strict_ans is not None and strict_ans.strip():
        return grade_answer_verl(solution_str, ground_truth)

    # 没有 \boxed{...}，用 fallback 抓一个数字
    guess = fallback_extract_numeric(solution_str)
    if guess is None:
        return False

    # 把 ground_truth 里面的 \boxed{} 去掉
    from answer_checker import extract_answer as extract_gt
    if "\\boxed" in ground_truth:
        gt = extract_gt(ground_truth)
    else:
        gt = ground_truth

    # 用 mathd / sympy 比较
    from answer_checker import grade_answer_mathd, grade_answer_sympy
    return grade_answer_mathd(guess, gt) or grade_answer_sympy(guess, gt)
