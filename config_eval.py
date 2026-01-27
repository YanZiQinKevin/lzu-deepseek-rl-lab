# config_eval.py
"""
评测/生成阶段的统一配置。

后续脚本（run_generate_baseline.py、run_grade_baseline.py）都只从这里
读取模型名、数据路径、任务列表和解码参数，方便以后统一改动。
"""

from pathlib import Path

# ----------------- 模型与设备 ----------------- #
# DeepSeek-R1-Distill-Qwen-1.5B 作为 baseline 模型
#BASE_MODEL: str = "/mnt/d/OneDrive/兰州交大/博士课题-大模型/models/DeepSeek-R1-Distill-Qwen-1.5B"
BASE_MODEL: str = "/mnt/d/OneDrive/兰州交大/博士课题-大模型/LLMProject/rl_outputs/AF-CuRL"

# 后面可以在脚本里用 torch.cuda.is_available() 动态决定是否用 cuda
DEFAULT_DEVICE: str = "cuda"

# ----------------- 数据 & 任务 ----------------- #
# 你的 benchmark 根目录：里面包含 AIME24、AIME25、... 等子文件夹
DATA_DIR: Path = Path("/mnt/d/OneDrive/兰州交大/博士课题-大模型/LLMProject/data")  # TODO: 按你实际目录修改

# 每个任务一个配置项；N=None 表示使用全部样本
TASKS = [
    {
        "name": "AIME24",
        "path": DATA_DIR / "AIME24" / "test.parquet",
        "N": None,  # 或者 30, 50 等，只采样前 N 条
    },
    {
        "name": "AIME25",
        "path": DATA_DIR / "AIME25" / "test.parquet",
        "N": None,
    },
    {
        "name": "AMC23",
        "path": DATA_DIR / "AMC23" / "test.parquet",
        "N": None,
    },
    {
        "name": "BRUMO25",
        "path": DATA_DIR / "BRUMO25" / "test.parquet",
        "N": None,
    },
    {
        "name": "CMIMC25",
        "path": DATA_DIR / "CMIMC25" / "test.parquet",
        "N": None,
    },
    {
        "name": "HMMT25",
        "path": DATA_DIR / "HMMT25" / "test.parquet",
        "N": None,
    },
    {
        "name": "MATH-500",
        "path": DATA_DIR / "MATH-500" / "test.parquet",
        "N": None,
    },
    {
        "name": "Minerva",
        "path": DATA_DIR / "Minerva" / "test.parquet",
        "N": None,
    },
    {
        "name": "Olympiad-Bench",
        "path": DATA_DIR / "Olympiad-Bench" / "test.parquet",
        "N": None,
    },
    {
        "name": "GSM8K",
        "path": DATA_DIR / "GSM8K" / "test.parquet",
        "N": 500,
    },
]

# ----------------- 生成 / 解码超参 ----------------- #

# 单次采样的随机种子列表（baseline 可以只用一个）
SEED_LIST = [0]

# 采样参数（和 JustRL 原文接近，可以之后再微调）
TEMPERATURE: float = 0.7
TOP_P: float = 0.9
MAX_NEW_TOKENS: int = 2048

# ----------------- 输出目录 ----------------- #

# 所有生成结果（jsonl）、评测结果（json）都放在这个目录下
OUT_DIR: Path = Path("eval_outputs") / BASE_MODEL.split("/")[-1]
OUT_DIR.mkdir(parents=True, exist_ok=True)
