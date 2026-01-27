import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_model(base_model_path, lora_path, output_path):
    print(f"=== 开始合并流程 ===")
    print(f"Base Model: {base_model_path}")
    print(f"LoRA Path: {lora_path}")
    print(f"Output Path: {output_path}")

    # 1. 加载 Tokenizer
    print("\n[1/4] 正在加载 Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            use_fast=False,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"加载 Tokenizer 失败: {e}")
        return

    # 2. 加载基础模型 (Base Model)
    print("[2/4] 正在加载基础模型 (Base Model)...")
    # 注意：合并时通常建议使用 float16 或 bfloat16，避免使用 quantization (load_in_8bit)
    # 因为量化后的模型通常很难无损合并
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    except Exception as e:
        print(f"加载基础模型失败: {e}")
        return

    # 3. 加载 LoRA Adapter 并合并
    print("[3/4] 正在加载 LoRA 并执行合并 (merge_and_unload)...")
    try:
        # 将 LoRA 挂载到 Base 模型上
        model_to_merge = PeftModel.from_pretrained(base_model, lora_path)

        # --- 核心步骤：合并权重并卸载 LoRA 结构 ---
        merged_model = model_to_merge.merge_and_unload()

        # 切换到评估模式
        merged_model.eval()
    except Exception as e:
        print(f"合并模型失败: {e}")
        return

    # 4. 保存最终模型和 Tokenizer
    print(f"[4/4] 正在保存合并后的模型到: {output_path} ...")
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # 保存模型权重
        merged_model.save_pretrained(
            output_path,
            safe_serialization=True  # 使用 safetensors 格式保存，更安全且加载更快
        )

        # 保存 Tokenizer
        tokenizer.save_pretrained(output_path)

        print(f"=== 合并成功！模型已保存至 {output_path} ===")

    except Exception as e:
        print(f"保存模型失败: {e}")


if __name__ == "__main__":
    # ================= 配置路径 =================
    # 请在这里修改你的实际路径
    # 如果是在 Windows 上，路径建议使用 r"D:\path\to\model" 格式

    BASE_MODEL_DIR = "/mnt/d/OneDrive/兰州交大/博士课题-大模型/models/DeepSeek-R1-Distill-Qwen-1.5B"  # 替换为你的底模路径
    LORA_ADAPTER_DIR = "/mnt/d/OneDrive/兰州交大/博士课题-大模型/LLMProject/eval_outputs/afcurl_v2_gsm_math_k1"  # 替换为你的 LoRA 路径
    OUTPUT_DIR = "/mnt/d/OneDrive/兰州交大/博士课题-大模型/LLMProject/rl_outputs/AF-CuRL"  # 合并后模型保存的路径
    # ===========================================

    merge_model(BASE_MODEL_DIR, LORA_ADAPTER_DIR, OUTPUT_DIR)