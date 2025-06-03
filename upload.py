from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import logging

quantized_model_dir = "Llama-3.2-3B-Instruct-lora-4bit-g128"

model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0")

repo_id = "c1uc/{}".format(quantized_model_dir)

model.push_to_hub(repo_id)
