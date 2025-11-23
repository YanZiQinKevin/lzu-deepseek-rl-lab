# run_generate_baseline.py
"""
使用 DeepSeek-R1-Distill-Qwen-1.5B 在各个数学 benchmark 上生成解题过程，
并保存为 jsonl 文件，后续由 run_grade_baseline.py 评分。

依赖:
    - config_eval.py
    - data_utils.py
    - transformers
    - torch
    - tqdm
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from config_eval import (
    BASE_MODEL,
    TASKS,
    TEMPERATURE,
    TOP_P,
    MAX_NEW_TOKENS,
    OUT_DIR,
)
from data_utils import Sample, load_task_samples


# ------------------------- 工具函数 ------------------------- #

def set_random_seed(seed: int) -> None:
    """同时设置 python / numpy / torch 的随机种子。"""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_prompt(question: str) -> str:
    """
    构造给模型的 user prompt。

    这里专门强调:
    - 直接给出数值答案
    - 最终答案用 \\boxed{} 包起来
    方便后续判题器直接解析。
    """
    return (
        "You are an expert competition mathematician."
        "Directly give the final numerical answer in the format \\boxed{}."
        "Do NOT show any reasoning steps.\n\n"
        "Question:\n"
        f"{question.strip()}\n"
    )


def load_model_and_tokenizer(
    model_name: str,
    use_4bit: bool = False,
    device: Optional[str] = None,
):
    """
    加载基座模型与 tokenizer。

    - use_4bit=True 时使用 bitsandbytes 4bit 量化（节省显存）
    - device 默认自动放到 GPU / CPU
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Loading tokenizer from {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    # 某些 Qwen 系列没定义 pad_token，这里统一设置为 eos
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[INFO] Loading model from {model_name} (4bit={use_4bit}) ...")

    model_kwargs: Dict = dict(
        trust_remote_code=True,
    )

    if use_4bit:
        # 4bit 量化配置（需要 bitsandbytes）
        model_kwargs.update(
            dict(
                load_in_4bit=True,
                device_map="auto",
            )
        )
    else:
        # fp16 + 自动放到 GPU
        model_kwargs.update(
            dict(
                torch_dtype=torch.float16,
                device_map="auto" if device == "cuda" else None,
            )
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )

    model.eval()
    print("[INFO] Model loaded.")
    return model, tokenizer


def generate_answer_for_sample(
    sample: Sample,
    model,
    tokenizer,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    device: Optional[str] = None,
) -> str:
    """
    对单个样本生成一个回答（完整推理过程 + \\boxed{} 最终答案）。

    返回: 模型生成的整个字符串。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    user_prompt = build_prompt(sample.question)
    messages = [{"role": "user", "content": user_prompt}]

    # 使用 chat template 构造输入文本
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
    )

    # 放到对应设备
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        outputs = model.generate(**inputs, **gen_kwargs)

    full_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # 去掉前面的 prompt，只保留 assistant 的回答部分
    if full_text.startswith(input_text):
        answer_text = full_text[len(input_text):]
    else:
        # 兜底：如果模板有差异，至少返回完整文本
        answer_text = full_text

    return answer_text.strip()


def get_task_config_by_name(name: str) -> Optional[Dict]:
    """在 config_eval.TASKS 中按名字寻找配置。"""
    for cfg in TASKS:
        if cfg["name"] == name:
            return cfg
    return None


def read_existing_example_ids(jsonl_path: Path) -> set[int]:
    """
    如果已经跑过一部分样本，可以通过读取现有 jsonl，
    把已经完成的 example_id 收集起来，实现断点续跑。

    注意：这里是按 example_id 粒度判断是否“已完成”；
    如果你修改了 --num-samples，最好删掉旧的 jsonl 再重跑。
    """
    done_ids: set[int] = set()
    if not jsonl_path.exists():
        return done_ids

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                ex_id = int(obj.get("example_id"))
                done_ids.add(ex_id)
            except Exception:
                continue
    return done_ids


# ------------------------- 主逻辑 ------------------------- #

def generate_for_task(
    task_name: str,
    model,
    tokenizer,
    output_dir: Path,
    max_samples: Optional[int],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
    num_samples: int,
    device: Optional[str] = None,
) -> None:
    """
    对单个任务生成 baseline 结果，并写入 jsonl 文件。

    文件命名:
        <task_name>_seed<seed>.jsonl

    每个样本会生成 num_samples 个回答（sample_id 从 0..num_samples-1）。
    """
    cfg = get_task_config_by_name(task_name)
    if cfg is None:
        raise ValueError(f"Task {task_name} not found in config_eval.TASKS")

    parquet_path: Path = cfg["path"]
    n_cfg = cfg.get("N", None)

    # 覆盖 config 中的 N
    effective_n = max_samples if max_samples is not None else n_cfg

    print(f"\n[INFO] Loading samples for task={task_name} from {parquet_path}")
    samples: List[Sample] = load_task_samples(
        task_name=task_name,
        parquet_path=parquet_path,
        n=effective_n,
    )
    print(f"[INFO] Loaded {len(samples)} samples for task {task_name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{task_name}_seed{seed}.jsonl"

    # 支持断点续跑：如果文件存在，先读取已经完成的 example_id
    done_ids = read_existing_example_ids(out_file)
    if done_ids:
        print(f"[INFO] Found existing file {out_file}, "
              f"{len(done_ids)} examples already done. Will skip them.")

    mode = "a" if out_file.exists() else "w"
    fout = out_file.open(mode, encoding="utf-8")

    progress = tqdm(
        samples,
        desc=f"Generating [{task_name}]",
        dynamic_ncols=True,
    )

    for sample in progress:
        if sample.example_id in done_ids:
            continue

        for k in range(num_samples):
            # 为了让每个 sample_id 的输出不同，可以在 base seed 基础上偏移
            this_seed = seed * 1000 + k
            set_random_seed(this_seed)

            try:
                response = generate_answer_for_sample(
                    sample=sample,
                    model=model,
                    tokenizer=tokenizer,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
            except RuntimeError as e:
                # 一些 OOM 或其他异常在这里被捕获，你可以按需处理
                print(f"\n[ERROR] Failed on {task_name}#{sample.example_id} (sample_id={k}): {e}")
                # 简单策略：写一个空响应，标记错误
                response = f"[GENERATION_ERROR] {repr(e)}"

            record = {
                "task_name": sample.task_name,
                "example_id": sample.example_id,
                "sample_id": k,
                "question": sample.question,
                "answer": sample.answer,
                "response": response,
                "seed": this_seed,
                "gen_config": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": max_new_tokens,
                    "model_name": BASE_MODEL,
                },
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    fout.close()
    print(f"[INFO] Finished task={task_name}, results saved to {out_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate baseline answers with DeepSeek-R1-Distill-Qwen-1.5B"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="要评测的任务名称，逗号分隔；默认 all 表示 config_eval.TASKS 中的全部",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子（影响 sampling）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="每个任务最多生成多少条样本；默认 None 使用 config_eval.TASKS 中的 N",
    )
    parser.add_argument(
        "--use-4bit",
        action="store_true",
        help="使用 4bit 量化加载模型以节省显存（推荐在 8G 显存上打开）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="指定设备，例如 cuda 或 cpu；默认自动选择",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE,
        help=f"采样温度，默认 {TEMPERATURE}",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=TOP_P,
        help=f"top-p 采样阈值，默认 {TOP_P}",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help=f"最大生成长度（新 token 数），默认 {MAX_NEW_TOKENS}",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="每个样本生成多少个回答（即 @K 里的 K），默认 1。",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    set_random_seed(args.seed)

    # 解析任务列表
    if args.tasks == "all":
        task_names = [cfg["name"] for cfg in TASKS]
    else:
        task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]

    print(f"[INFO] Tasks to run: {task_names}")
    print(f"[INFO] Using seed={args.seed}, 4bit={args.use_4bit}, device={args.device}")
    print(f"[INFO] Generation params: temperature={args.temperature}, "
          f"top_p={args.top_p}, max_new_tokens={args.max_new_tokens}, "
          f"num_samples={args.num_samples}")

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(
        model_name=BASE_MODEL,
        use_4bit=args.use_4bit,
        device=args.device,
    )

    # 逐任务生成
    for task_name in task_names:
        generate_for_task(
            task_name=task_name,
            model=model,
            tokenizer=tokenizer,
            output_dir=OUT_DIR,
            max_samples=args.max_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            num_samples=args.num_samples,
            device=args.device,
        )

    print("[INFO] All tasks finished.")


if __name__ == "__main__":
    import sys
    #sys.argv = ["run_generate_baseline.py", "--tasks", "AIME24", "--max-samples", "3","--num-samples","3"]
    sys.argv = ["run_generate_baseline.py", "--tasks", "MATH-500", "--max-samples", "10"]
    #sys.argv = ["run_generate_baseline.py", "--tasks", "AIME24", "--max-samples", "3"]
    main()