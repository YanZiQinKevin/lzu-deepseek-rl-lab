# run_grade_baseline.py
"""
读取 run_generate_baseline.py 生成的 jsonl 文件，
使用 answer_checker.grade_answer_verl 对每条样本判题，
并统计各个任务的 baseline 正确率。

依赖:
    - config_eval.py (提供 OUT_DIR)
    - answer_checker.py (提供 grade_answer_verl)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

from config_eval import OUT_DIR
from answer_checker import grade_answer_verl


@dataclass
class Stat:
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grade baseline jsonl outputs and compute accuracy."
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
        help="是否按 seed 维度分别统计 (task, seed) 精度",
    )
    return parser.parse_args()


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


def grade_jsonl_file(
    jsonl_path: Path,
    totals_by_task: Dict[str, Stat],
    totals_by_task_seed: Dict[Tuple[str, int], Stat] | None = None,
) -> None:
    """
    对单个 jsonl 文件打分并更新统计信息。

    jsonl 每行应至少包含:
        - task_name
        - example_id
        - answer   (ground truth)
        - response (model output)
        - seed     (optional, 默认 0)
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
            seed = int(obj.get("seed", 0))
            ground_truth = obj.get("answer")
            solution = obj.get("response", "")

            # 调用你提供的判题器
            try:
                is_correct = bool(grade_answer_verl(solution, ground_truth))
            except Exception as e:
                # 判题器异常时，记为错误并打印一行日志
                print(
                    f"[ERROR] Grading error in {jsonl_path} "
                    f"(task={task_name}, seed={seed}): {e}"
                )
                is_correct = False

            num_lines += 1
            if is_correct:
                num_correct += 1

            # 更新按 task 统计
            stat_task = totals_by_task[task_name]
            stat_task.total += 1
            if is_correct:
                stat_task.correct += 1

            # 更新按 (task, seed) 统计
            if totals_by_task_seed is not None:
                key = (task_name, seed)
                stat_ts = totals_by_task_seed[key]
                stat_ts.total += 1
                if is_correct:
                    stat_ts.correct += 1

    acc = num_correct / num_lines if num_lines > 0 else 0.0
    print(
        f"[INFO] File summary: {jsonl_path.name} "
        f"| correct={num_correct}/{num_lines} "
        f"({acc:.3f})"
    )


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
    print(f"[INFO] Per-seed stats: {args.per_seed}")

    totals_by_task: Dict[str, Stat] = defaultdict(Stat)
    totals_by_task_seed: Dict[Tuple[str, int], Stat] | None = (
        defaultdict(Stat) if args.per_seed else None
    )

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
            totals_by_task=totals_by_task,
            totals_by_task_seed=totals_by_task_seed,
        )

    # -------- 打印总体结果 -------- #
    print("\n========== Per-task summary ==========")
    for task_name, stat in sorted(totals_by_task.items()):
        print(
            f"{task_name:20s}  "
            f"{stat.correct:4d}/{stat.total:4d}  "
            f"acc = {stat.accuracy:.3f}"
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
            f"acc = {total_all.accuracy:.3f}"
        )

    # -------- 按 (task, seed) 的详细结果（可选） -------- #
    result_dict: Dict[str, dict] = {
        task: {
            "correct": stat.correct,
            "total": stat.total,
            "accuracy": stat.accuracy,
        }
        for task, stat in totals_by_task.items()
    }

    if totals_by_task_seed is not None:
        per_seed_dict: Dict[str, dict] = {}
        for (task, seed), stat in totals_by_task_seed.items():
            key = f"{task}__seed{seed}"
            per_seed_dict[key] = {
                "task_name": task,
                "seed": seed,
                "correct": stat.correct,
                "total": stat.total,
                "accuracy": stat.accuracy,
            }
        result_dict["_per_seed"] = per_seed_dict

    # 保存到 json 文件
    out_path = input_dir / "baseline_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Results saved to: {out_path}")


if __name__ == "__main__":
    main()
