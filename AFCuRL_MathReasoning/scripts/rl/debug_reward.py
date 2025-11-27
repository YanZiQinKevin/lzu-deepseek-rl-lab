from pathlib import Path
import json
import sys

# 保证能导入 reward_fn
PROJECT_ROOT = Path(__file__).resolve().parents[0]  # 如果放在工程根
sys.path.insert(0, str(PROJECT_ROOT))

from reward_fn import compute_reward, RL_PHASE2_CONFIG

if __name__ == "__main__":
    with open("debug_last_sample.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    q = data["question"]
    gt = data["ground_truth"]
    resp = data["response"]

    print("idx =", data["idx"], "k =", data["k"])
    print("response snippet:", resp[:200])

    comps = compute_reward(
        response=resp,
        ground_truth=gt,
        cfg=RL_PHASE2_CONFIG,
        return_components=True,
    )
    print("R_total =", comps["R_total"])
    print("components:", comps)
