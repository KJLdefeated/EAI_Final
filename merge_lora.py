from peft import AutoPeftModelForCausalLM
import torch

model_id = "BensonW/EAI-Final-LoRA"  # model_id = "BensonW/EAI-Final-LoRA-1B"
peft_model = AutoPeftModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
print(f"Model before merged: {type(peft_model)}")

merged_model = peft_model.merge_and_unload()
print(f"Model before merged: {type(merged_model)}")

merged_model.save_pretrained("./Llama-3.2-3B-Instruct-lora")
