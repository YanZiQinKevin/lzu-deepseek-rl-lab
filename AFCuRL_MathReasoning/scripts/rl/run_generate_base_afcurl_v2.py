# scripts/rl/run_generate_base_afcurl_v2.py

from dataclasses import dataclass
from pathlib import Path
import torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from reward_fn import compute_reward, RL_PHASE2_CONFIG
print("[DEBUG] reward_fn loaded from:", sys.modules["reward_fn"].__file__)


@dataclass
class EvalConfig:
    base_model_path: str
    eval_csv: str
    output_csv: str
    k_samples: int = 2
    max_prompt_tokens: int = 384
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def build_prompt(q: str) -> str:
    system = (
        "You are an expert competition mathematician. "
        "Directly give the final numerical answer in the format \\boxed{}. "
        "Do NOT show any reasoning steps."
    )
    return f"{system}\n\nProblem:\n{q}\n"

def load_eval_data(path: str):
    df = pd.read_csv(path)
    assert {"question","ground_truth"}.issubset(df.columns)
    return df

def create_model_and_tokenizer(cfg: EvalConfig):
    tok = AutoTokenizer.from_pretrained(
        cfg.base_model_path,
        use_fast=False,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        trust_remote_code=True,
    ).to(cfg.device)
    model.eval()
    return model, tok

def evaluate(cfg: EvalConfig):
    df = load_eval_data(cfg.eval_csv)
    out_dir = Path(cfg.output_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tok = create_model_and_tokenizer(cfg)

    records = []
    for idx, row in df.iterrows():
        q = str(row["question"]); gt = str(row["ground_truth"])
        prompt = build_prompt(q)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
        in_ids = enc.input_ids.to(cfg.device)
        attn = enc.attention_mask.to(cfg.device)
        if in_ids.shape[1] > cfg.max_prompt_tokens:
            in_ids = in_ids[:, -cfg.max_prompt_tokens:]
            attn = attn[:, -cfg.max_prompt_tokens:]

        cand = []
        print(f"[DEBUG] idx={idx} -  generate", flush=True)
        with torch.no_grad():
            for k in range(cfg.k_samples):

                gen_ids = model.generate(
                    input_ids=in_ids,
                    attention_mask=attn,
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=True,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    pad_token_id=tok.eos_token_id,
                )
                resp = tok.decode(gen_ids[0][in_ids.shape[1]:], skip_special_tokens=True)
                comps = compute_reward(resp, gt, cfg=RL_PHASE2_CONFIG, return_components=True)
                cand.append((resp, comps))

        best_resp, best = max(cand, key=lambda x: x[1]["R_total"])
        records.append({
            "index": idx,
            "question": q,
            "ground_truth": gt,
            "response": best_resp,
            "reward_total": best["R_total"],
            "is_correct": best["is_correct"],
            "has_valid_boxed": best["has_valid_boxed"],
            "response_len": best["len_total"],
        })

    out_df = pd.DataFrame(records)
    out_df.to_csv(cfg.output_csv, index=False, encoding="utf-8-sig")

    acc = out_df["is_correct"].mean()
    boxed_rate = out_df["has_valid_boxed"].mean()
    avg_len = out_df["response_len"].mean()
    print(f"[SUMMARY] acc={acc:.3f}, boxed={boxed_rate:.3f}, len={avg_len:.1f}")
    summary_line = f"[SUMMARY] acc={acc:.3f}, boxed={boxed_rate:.3f}, len={avg_len:.1f}\n"
    with open(PROJECT_ROOT
              / "eval_outputs" / "baselines" / "DeepSeek-R1-Distill-Qwen-1.5B"
              / "base_distill_math_test_k2" /"base_distill_math_test_k2.txt", "a", encoding="utf-8") as f:
        f.write(summary_line)

if __name__ == "__main__":
    cfg = EvalConfig(
        base_model_path=str(PROJECT_ROOT / "models" / "DeepSeek-R1-Distill-Qwen-1.5B"),
        eval_csv=str(PROJECT_ROOT / "data" / "splits" / "math500" / "math500_test.csv"),
        output_csv=str(PROJECT_ROOT / "eval_outputs" / "baselines" / "DeepSeek-R1-Distill-Qwen-1.5B"
                       /"base_distill_math_test_k2"/"base_distill_math_test_k2.csv"),
    )
    evaluate(cfg)
