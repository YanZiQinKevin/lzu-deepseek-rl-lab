# run_grade_baseline.py
"""
读取 run_generate_baseline.py 生成的 jsonl 文件，
使用判题器对每条样本判题，并统计各个任务的 baseline 正确率。

支持多次采样：
    - 对同一题 (task_name, example_id)，如果任意一个 sample_id 的回答是正确的，
      则认为该题在 @K 下答对（类似 accuracy@K / pass@K）。

同时导出一个 graded_samples.csv，方便人工检查：
    task_name, example_id, sample_id, is_correct, question, ground_truth, response

依赖:
    - config_eval.py (提供 OUT_DIR)
    - answer_checker.py / lenient_grader.py (提供判题函数)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd  # 用于导出 CSV

from config_eval import OUT_DIR
from lenient_grader import grade_answer_lenient as grade_fn  # 宽松版判题器


# ----------------- 统计结构 ----------------- #

@dataclass
class Stat:
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


# ----------------- 参数解析 ----------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grade baseline jsonl outputs and compute accuracy (support multi-sample pass@K)."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(OUT_DIR),
        help=f"jsonl 结果所在目录，默认 {OUT_DIR}",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="要评测的任务名称，逗号分隔；默认 all 表示目录下所有任务",
    )
    parser.add_argument(
        "--per-seed",
        action="store_true",
        help="（保留参数）是否按 seed 维度分别统计；当前实现主要关注按题目聚合的 accuracy@K。",
    )
    return parser.parse_args()


# ----------------- 文件过滤 ----------------- #

def should_use_file(path: Path, task_filter: set[str]) -> bool:
    """根据任务过滤规则决定是否使用某个 jsonl 文件。"""
    if not path.name.endswith(".jsonl"):
        return False

    if not task_filter:
        return True

    # 约定文件名形如: AIME24_seed0.jsonl
    stem = path.stem
    for t in task_filter:
        if t in stem:
            return True
    return False


# ----------------- 核心打分函数 ----------------- #

def grade_jsonl_file(
    jsonl_path: Path,
    per_example_results: Dict[Tuple[str, int], List[bool]],
    rows: List[dict],
) -> None:
    """
    对单个 jsonl 文件打分，并将样本级结果记录到 per_example_results。

    jsonl 每行应至少包含:
        - task_name
        - example_id
        - sample_id  (如果没有则默认 0)
        - answer     (ground truth)
        - response   (model output)
        - seed       (可选，目前仅用于 debug)
    """
    print(f"\n[INFO] Grading file: {jsonl_path}")

    num_lines = 0
    num_correct = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] Skip invalid json line in {jsonl_path}")
                continue

            task_name = obj.get("task_name", "UNKNOWN")
            example_id = obj.get("example_id")
            sample_id = obj.get("sample_id", 0)
            ground_truth = obj.get("answer")
            response = obj.get("response", "")

            # 调用判题器（宽松版）
            try:
                is_correct = bool(grade_fn(response, ground_truth))
            except Exception as e:
                print(
                    f"[ERROR] Grading error in {jsonl_path} "
                    f"(task={task_name}, example_id={example_id}, sample_id={sample_id}): {e}"
                )
                is_correct = False

            num_lines += 1
            if is_correct:
                num_correct += 1

            # 记录到 per-example 结果：一题的多个 sample 对错列表
            key = (task_name, int(example_id))
            per_example_results[key].append(is_correct)

            # 为导出 CSV 记录一行（样本级）
            rows.append(
                {
                    "task_name": task_name,
                    "example_id": example_id,
                    "sample_id": sample_id,
                    "is_correct": int(is_correct),
                    "question": obj.get("question"),
                    "ground_truth": ground_truth,
                    "response": response,
                }
            )

    acc_sample_level = num_correct / num_lines if num_lines > 0 else 0.0
    print(
        f"[INFO] File summary (sample-level): {jsonl_path.name} "
        f"| correct={num_correct}/{num_lines} "
        f"({acc_sample_level:.3f})"
    )


# ----------------- 主流程 ----------------- #

def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    # 解析任务过滤
    if args.tasks == "all":
        task_filter: set[str] = set()
    else:
        task_filter = {t.strip() for t in args.tasks.split(",") if t.strip()}

    print(f"[INFO] Using input dir: {input_dir}")
    print(f"[INFO] Task filter: {task_filter if task_filter else 'ALL'}")
    print(f"[INFO] Per-seed stats flag (currently unused): {args.per_seed}")

    # (task_name, example_id) -> list[bool] (该题每个 sample 的对错)
    per_example_results: Dict[Tuple[str, int], List[bool]] = defaultdict(list)

    # 收集所有样本行，用于导出 CSV
    graded_rows: List[dict] = []

    # 遍历目录下所有 jsonl 文件
    jsonl_files: List[Path] = sorted(input_dir.glob("*.jsonl"))
    jsonl_files = [p for p in jsonl_files if should_use_file(p, task_filter)]

    if not jsonl_files:
        print("[WARN] No jsonl files found for grading.")
        return

    print(f"[INFO] Found {len(jsonl_files)} jsonl files to grade.")

    for path in jsonl_files:
        grade_jsonl_file(
            jsonl_path=path,
            per_example_results=per_example_results,
            rows=graded_rows,
        )

    # -------- 按题目聚合，计算 accuracy@K -------- #
    totals_by_task: Dict[str, Stat] = defaultdict(Stat)

    for (task_name, example_id), result_list in per_example_results.items():
        any_correct = any(result_list)   # 有任一 sample 答对，就算这题对
        stat = totals_by_task[task_name]
        stat.total += 1
        if any_correct:
            stat.correct += 1

    # -------- 打印总体结果 -------- #
    print("\n========== Per-task summary (problem-level, pass@K) ==========")
    for task_name, stat in sorted(totals_by_task.items()):
        print(
            f"{task_name:20s}  "
            f"{stat.correct:4d}/{stat.total:4d}  "
            f"acc@K = {stat.accuracy:.3f}"
        )

    if totals_by_task:
        total_all = Stat(
            correct=sum(s.correct for s in totals_by_task.values()),
            total=sum(s.total for s in totals_by_task.values()),
        )
        print("--------------------------------------")
        print(
            f"{'ALL':20s}  "
            f"{total_all.correct:4d}/{total_all.total:4d}  "
            f"acc@K = {total_all.accuracy:.3f}"
        )

    # -------- 保存 summary 到 JSON -------- #
    result_dict: Dict[str, dict] = {
        task: {
            "correct": stat.correct,
            "total": stat.total,
            "accuracy": stat.accuracy,
        }
        for task, stat in totals_by_task.items()
    }

    out_path = input_dir / "baseline_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Results saved to: {out_path}")

    # -------- 导出所有样本到 CSV -------- #
    if graded_rows:
        csv_path = input_dir / "graded_samples.csv"
        df = pd.DataFrame(
            graded_rows,
            columns=[
                "task_name",
                "example_id",
                "sample_id",
                "is_correct",
                "question",
                "ground_truth",
                "response",
            ],
        )
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Graded samples (per-sample) saved to: {csv_path}")
    else:
        print("[INFO] No graded rows collected; CSV not generated.")


if __name__ == "__main__":
    main()
