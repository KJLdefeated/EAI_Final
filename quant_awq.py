from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset


def load_wikitext():
    data = load_dataset(path="wikitext", name="wikitext-2-raw-v1", split="train")
    return [text for text in data["text"] if text.strip() != '']


original_path = "meta-llama/Llama-3.2-3B-Instruct"
model_path = "Llama-3.2-3B-Instruct-lora"
quant_path = "Llama-3.2-3B-Instruct-lora-awq"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path, **{"low_cpu_mem_usage": True, "use_cache": False}
)
tokenizer = AutoTokenizer.from_pretrained(original_path, trust_remote_code=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config, calib_data=load_wikitext())

# Save quantized model
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)

print(f'Model is quantized and saved at "{quant_path}"')
