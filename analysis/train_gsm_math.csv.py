import pandas as pd


df = pd.read_csv("/mnt/d/OneDrive/兰州交大/博士课题-大模型/LLMProject/eval_outputs/DeepSeek-R1-Distill-Qwen-1.5B/graded_samples.csv")

# 例如：只保留 GSM8K 和 MATH-500 的样本
df_train = df[df["task_name"].isin(["GSM8K", "MATH-500"])]

# 只保留 question / ground_truth
df_train = df_train[["question", "ground_truth"]].reset_index(drop=True).tail(200)

df_train.to_csv("/mnt/d/OneDrive/兰州交大/博士课题-大模型/LLMProject/eval_outputs/DeepSeek-R1-Distill-Qwen-1.5B/train_gsm_math.csv", index=False)