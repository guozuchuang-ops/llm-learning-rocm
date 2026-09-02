from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2-0.5B-Instruct"

def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

    prompt = "请简单解释什么是大模型量化"
    inputs = tokenizer(prompt, return_tensors="pt")

    print("Generating...")
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nResult:", result)

if __name__ == "__main__":
    main()
