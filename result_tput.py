import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from datasets import load_dataset
import random
import numpy as np

from vllm import LLM, SamplingParams
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

#####################################################################
# === SPEC NOTICE ===
# Only "load model" and "generate" function selection can be modified.
# DO NOT change PPL calculation, timing, or throughput logic.
#####################################################################
sp = SamplingParams(max_tokens=256, temperature=0.0)  # greedy
sp_ppl = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0.0)  # greedy


# === (Optional) Define your own custom generate function. ===
# This is useful if you want full control over KV cache and generation steps.
# You can modify this function to suit your needs.
# By default, we use model.generate() for simplicity and general use.
def generate(model, input_ids, past_key_values, max_new_tokens):
    input_ids = input_ids.clone()
    with torch.no_grad():
        # Prefill
        outputs = model.prefill_forward(
            input_ids,
            past_key_values=past_key_values,
            position_ids=None,
            attention_mask=None,
            cache_position=None,
            logits_to_keep=1,
        )
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits, dim=-1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # Token-by-token Decoding
        for _ in range(max_new_tokens):
            pos = input_ids.shape[1]
            cache_position = torch.arange(
                pos, pos + 1, device=input_ids.device, dtype=torch.long
            )

            outputs = model(
                next_token,
                past_key_values=past_key_values,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position,
            )
            logits = outputs.logits
            next_token = torch.argmax(logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values

    return input_ids


def evaluate_ppl(model, tokenizer, device="cuda:0"):
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    test_enc = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    seqlen = 2048
    test_enc = test_enc.input_ids

    nsamples = test_enc.numel() // seqlen
    nll, ntok = 0.0, 0
    for i in tqdm(range(nsamples), desc="Evaluating..."):
        batch = test_enc[:, (i * seqlen) : ((i + 1) * seqlen)]  # shape: (1, 2048)

        with torch.no_grad():
            text = tokenizer.decode(
                batch[0], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            generated_info = model.generate([text], sp_ppl)[0]
            logps = [
                p[id].logprob
                for p, id in zip(
                    generated_info.prompt_logprobs[1:-1],
                    generated_info.prompt_token_ids[1:-1],
                )
            ]
            # print(len(logps))

        nll -= sum(logps)
        ntok += len(logps)

    return np.exp(nll / ntok).item()


def main():
    ############## Set Up ##############
    torch.manual_seed(0)
    random.seed(0)

    max_new_tokens = 256  # Number of new tokens to generate
    device = "cuda:0"

    ### === TODO: Load your model (you may change this part) ===

    # model_name = "meta-llama/Llama-3.2-3B-Instruct"
    quantized_model_dir = "c1uc/Llama-3.2-3B-Instruct-lora-4bit-g128"
    #original_model_dir = "meta-llama/Llama-3.2-3B-Instruct"
    quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
        desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    )
    # model = AutoGPTQForCausalLM.from_quantized(model_name, device="cuda:0")
    model = LLM(
        model=quantized_model_dir,
        tokenizer=quantized_model_dir,
        dtype="auto",
        quantization='gptq',
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
        speculative_config={
            "model": "BensonW/EAI-Final-draft-model-gptq",
            "quantization": "gptq",
            "num_speculative_tokens": 3,
        },
        compilation_config={
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
            "max_capture_size": 16,
        },
    )

    sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)  # greedy
    #####################################

    # model.eval()
    tokenizer = AutoTokenizer.from_pretrained(quantized_model_dir)

    # === (Optional) Uncomment the following lines if using the custom generate() function. ===
    # model.prefill_forward = model.forward

    warmup_prompt = "Explain what AI is."
    inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # === (Optional) Set up StaticCache for manual KV cache management ===
    # from transformers import StaticCache
    # past_key_values = StaticCache(
    #     config=model.config,
    #     max_batch_size=1,
    #     max_cache_len=max_new_tokens + 16,
    #     device=model.device,
    #     dtype=torch.float16
    # )
    ####################################################################

    for i in tqdm(range(5), desc="Warm Up..."):
        #  === Default: use model.generate() for end-to-end warm-up ===
        # _ = model.generate(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     max_new_tokens=max_new_tokens,
        #     pad_token_id=tokenizer.eos_token_id,
        # )

        # === (Optional) Use custom generate() if uncommented ===
        # generated = generate(model, input_ids, past_key_values, max_new_tokens)
        # past_key_values.reset()
        generated_info = model.generate([warmup_prompt], sp)[0]
        generated = torch.tensor([generated_info.outputs[0].token_ids])

    prompt = "How to learn a new language?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    tputs = []
    time_record = []
    for _ in tqdm(range(10), desc="Test Inference"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # === Default: Use model.generate() for end-to-end timing ===
        # generated = model.generate(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     max_new_tokens=max_new_tokens,
        #     pad_token_id=tokenizer.eos_token_id,
        # )

        # === Optional: Use custom generate() if uncommented ===
        # generated = generate(model, input_ids, past_key_values, max_new_tokens)
        # past_key_values.reset()
        generated_info = model.generate([prompt], sp)[0]
        generated = torch.tensor([generated_info.outputs[0].token_ids])

        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        tput = generated[0][input_ids.shape[1] :].shape[0] / (elapsed_ms / 1000)
        time_record.append(elapsed_ms / 1000)
        tputs.append(tput)

    response = tokenizer.decode(
        generated[0][input_ids.shape[1] :], skip_special_tokens=True
    )
    sorted_tputs = np.sort(tputs)[2:-2]
    org_tput = np.mean(sorted_tputs)
    print(f"Prompt: {prompt}\nResponse: {response}\n")

    print(f"Time Record: {time_record}")
    print(f"Throughput Record: {tputs} toks/s\n")

    ### Your final throughput result ###
    print(f"Throughput: {org_tput} toks/s")
    # ppl = evaluate_ppl(model, tokenizer, device)
    # print(f"Perplexity (PPL): {ppl}")

    # Save results to CSV
    import csv

    rounded_tput = round(org_tput, 1)
    #ppl = round(ppl, 2)

    with open("result_tput.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "value"])
        #writer.writerow([0, ppl])
        writer.writerow([1, rounded_tput])


if __name__ == "__main__":
    main()
    torch.distributed.destroy_process_group()
