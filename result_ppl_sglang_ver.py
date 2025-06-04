# -*- coding: utf-8 -*-
"""
SGLang offline-engine throughput benchmark
model: Llama-3.2-3B-Instruct-LoRA-GPTQ
"""

import sglang as sgl
from transformers import AutoTokenizer
import torch, random, numpy as np, csv
from tqdm.auto import tqdm


def main():
    # ---------- 0. 環境與模型 ----------
    torch.manual_seed(0)
    random.seed(0)

    MODEL_PATH = "c1uc/Llama-3.2-3B-Instruct-lora-4bit-g128"
    tokenizer  = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    engine = sgl.Engine(
        model_path=MODEL_PATH,
        quantization="gptq",
        dtype="float16",
        mem_fraction_static=0.7,
        # tensor_parallel_size=1,
        enable_torch_compile=True,
        cuda_graph_max_bs=80,
    )
    # engine = sgl.Engine(
    #     model_path=MODEL_PATH,
    #     quantization="gptq",
    #     dtype="float16",
    #     mem_fraction_static=0.6,
    #     enable_torch_compile=True,
    #     speculative_algorithm="EAGLE",
    #     speculative_draft_model_path="BensonW/EAI-Final-draft-model-gptq",
    #     speculative_num_steps=3,
    #     speculative_eagle_topk=4,
    #     speculative_num_draft_tokens=16,
    #     cuda_graph_max_bs=8,
    #     # 其他參數視需要加入
    # )

    # ---------- 1. Sampling 參數 ----------
    MAX_NEW_TOKENS = 256
    sp = {"max_new_tokens": MAX_NEW_TOKENS, "temperature": 0.0}

    # ---------- 2. Warm-up ----------
    for _ in tqdm(range(5), desc="Warm-up"):
        engine.generate(["Explain what AI is."], sp)

    # ---------- 3. Benchmark ----------
    test_prompt = "How to learn a new language?"
    input_ids   = tokenizer(test_prompt, return_tensors="pt")["input_ids"]

    tputs, times = [], []
    for _ in tqdm(range(10), desc="Benchmark"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True); start.record()

        out = engine.generate([test_prompt], sp)[0]
        # print(f"Output: {out}, keys = {out.keys()}")
        # out_len = len(out["token_ids"]) - input_ids.shape[1]
        out_len = out['meta_info']['completion_tokens']

        end = torch.cuda.Event(enable_timing=True); end.record()
        torch.cuda.synchronize()
        sec = start.elapsed_time(end) / 1000
        tputs.append(out_len / sec);  times.append(sec)

    # ---------- 4. 結果 ----------
    # resp_text = tokenizer.decode(
    #     out["token_ids"][input_ids.shape[1]:], skip_special_tokens=True
    # )
    resp_text = out['text']
    avg_tput = np.mean(np.sort(tputs)[2:-2])

    print(f"Prompt:\n{test_prompt}\n---\n{resp_text}\n")
    print(f"Throughput (avg): {avg_tput:.1f} toks/s")

    # ---------- 5. 存 CSV ----------
    with open("result.csv", "w", newline="") as f:
        csv.writer(f).writerows([["Id", "value"], [1, round(avg_tput, 1)]])

    engine.shutdown()


# ——— 必 須 有 這 行 ———
if __name__ == "__main__":
    main()
