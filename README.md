# LLMProject

本仓库包含数学推理方向的 RL 训练与评测代码。主程序位于 `AFCuRL_MathReasoning/`。

## 目录概览

- `AFCuRL_MathReasoning/`：AF-CuRL 训练与评测主工程（主要脚本、数据、模型与输出）。
- `data/`、`eval/`、`RL/`、`analysis/`：其他实验数据与脚本（非主入口）。

## 快速开始（主程序）

在仓库根目录执行：

```bash
cd AFCuRL_MathReasoning
```

### 1) 数据准备（可选）

如需从 `data/benchmarks/train_gsm_math.csv` 生成切分数据：

```bash
python scripts/tools/make_splits_gsm_math.py
```

输出会写入：
- `data/splits/gsm8k/`
- `data/splits/math500/`
- `data/splits/combo/`

### 2) 训练（AF-CuRL, K=1）

```bash
python scripts/rl/run_rl_train_k1_afcurl.py
```

该脚本默认使用：
- `models/DeepSeek-R1-Distill-Qwen-1.5B`
- `data/splits/combo/gsm8k_math500_train_rl.csv`
- 输出到 `rl_runs/DeepSeek-R1-Distill-Qwen-1.5B/afcurl_gsm_math_k1_s42/`

如需自定义路径/超参，直接修改脚本顶部的配置区。

### 3) 评测（best-of-k）

基于训练后的 LoRA 权重进行评测：

```bash
python scripts/rl/run_generate_with_lora_afcurl.py
```

或使用 v2 脚本：

```bash
python scripts/rl/run_generate_with_lora_v2.py
```

评测默认输出到 `eval_outputs/` 目录下相应子文件夹。

## 依赖环境（建议）

- Python 3.9+
- torch
- transformers
- peft
- pandas
- scikit-learn

> 说明：仓库未提供统一的依赖文件，请根据脚本 import 安装。

## 常见问题

- `ImportError: No module named 'reward_fn'`
  - 目前 `reward_fn.py` 位于 `scripts/rl/` 下，而脚本默认从项目根目录导入。
  - 解决方式：
    - 方式 A：将 `AFCuRL_MathReasoning/scripts/rl` 加入 `PYTHONPATH`
    - 方式 B：将 `reward_fn.py` 复制或移动到 `AFCuRL_MathReasoning/` 根目录

## 备注

训练与评测脚本均默认从 `AFCuRL_MathReasoning` 根目录运行，请保持当前工作目录一致。
