from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 待AMD GPU算力支持后测试7B/14B模型
model_name = "Qwen/Qwen2-0.5B-Instruct"
device = "cuda"  # ROCm环境下可直接映射为HIP设备

def main():
    print("Loading model with ROCm...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    prompt = "请简单解释什么是LoRA微调"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print("Generating...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nResult:", result)

if __name__ == "__main__":
    main()
