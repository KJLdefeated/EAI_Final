# Large Language Model Acceleration

This project explores **inference-time acceleration of large language models (LLMs)** by combining
parameter-efficient fine-tuning, weight quantization, speculative decoding, and high-performance inference engines.

We focus on improving both **throughput** and **model quality** for the `meta-llama/Llama-3.2-3B-Instruct` model on constrained hardware such as NVIDIA T4/RTX3090 GPUs.

[Comprehensive Report](https://hackmd.io/@kBcUQuNKSQKpa1_ZlUZhCw/HyvQA3HZgg)

---

## Project Overview

### Goal
- Reduce inference latency and memory usage of LLMs without significant loss in accuracy.
- Combine complementary techniques:
  - LoRA fine-tuning for better perplexity
  - Post-training quantization (GPTQ / AWQ)
  - Speculative decoding for multi-token speedup
  - High-throughput inference frameworks (vLLM, SGLang)

### Workflow
![Pipeline](https://hackmd.io/_uploads/rJGvjcpfel.png)

---

## Methodology

We follow a **workflow-oriented order**: analyze the model → improve accuracy → compress weights → accelerate inference.

### 1. Model Analysis
![Model Structure](https://hackmd.io/_uploads/rkK4Zpr-le.png)

For each transformer layer in `Llama-3.2-3B-Instruct`, parameter sizes follow `mlp.gate_proj` = `mlp.up_proj` = `mlp.down_proj` > `self_attn.q_proj` = `self_attn.o_proj` > `self_attn.v_proj` = `self_attn.k_proj`. 
Understanding this structure guides us in choosing layers for fine-tuning or quantization.

---

### 2. Accuracy Enhancement — LoRA
[LoRA](https://arxiv.org/abs/2106.09685) inserts a pair of trainable low-rank matrices into linear layers:

For a weight matrix $\( W\in\mathbb{R}^{d\times k} \)$,
LoRA learns $\( A\in\mathbb{R}^{d\times r}, B\in\mathbb{R}^{r\times k} \)$,
producing a new weight: $W' = W + BA$
where $\(r\ll d,k\)$, so parameter overhead is only $\(2r(d+k)\)$.

**Our setup**
- Dataset: [Salesforce/wikitext-2-raw-v1](https://huggingface.co/datasets/Salesforce/wikitext/viewer/wikitext-2-raw-v1)
- Hardware: RTX3090
- Training: ~15 min
- Strategy: use a **low learning rate**, monitor perplexity frequently  
  (to avoid degenerate models with no response)

For details, see [`train_lora.py`](https://github.com/KJLdefeated/EAI_Final/blob/master/train_lora.py).

---

### 3. Compression — Quantization

We evaluate two weight-only post-training quantization methods.

#### 3.1 GPTQ
[GPTQ](https://arxiv.org/abs/2210.17323) minimizes layer-wise post-quantization error:
$\min_{\hat{W}\in\mathcal{Q}} \frac12\|X(\hat{W}-W)\|^2$
where $\(W\)$ is the original weight, $\(\hat{W}\)$ is the quantized weight,  
and $\(\mathcal{Q}\)$ is the integer set at the desired bit-width.

**Our setup**
- Merge PEFT weights
- 4-bit quantization, group size = 128
- Achieves good trade-off between compression and accuracy

#### 3.2 AWQ
[AWQ](https://dl.acm.org/doi/abs/10.1145/3714983.3714987) preserves the most **activation-sensitive 1%** of weight channels in higher precision (FP16/INT8) and quantizes the rest.

**Our setup**
- Merge PEFT weights
- 4-bit group quantization (group = 128)
- Retains high-influence channels to keep perplexity low

---

### 4. Inference-Time Speedup — Speculative Decoding
Speculative decoding uses a **two-model approach**:
1. A smaller, faster *draft* model proposes several candidate tokens.
2. The larger *target* model verifies them; accepted tokens are reused, rejected tokens are recomputed.

This yields **multi-token parallelism** during generation.

**Our setup**
- Draft model: `Llama-3.2-1B-Instruct`, fine-tuned and quantized as above
- Integrated in the [vLLM](https://github.com/vllm-project/vllm) framework
- Yields significant throughput improvement on GPU inference

---

## Implementation Notes

### Inference Frameworks
We tested two high-performance engines:

| Framework | KV-Cache Strategy | Batching |
|-----------|-------------------|----------|
| **vLLM**  | **Paged Attention** – splits KV cache into memory pages for dynamic reuse and defragmentation | Continuous batching |
| **SGLang**| **Radix Attention** – groups/alines KV cache to reduce cache-line thrashing | Persistent batching |

- **vLLM** natively supports speculative decoding.
- **SGLang** limits speculative decoding to [EAGLE/EAGLE-3](https://arxiv.org/abs/2401.15077), which requires identical hidden sizes for draft and target models — often leading to GPU resource contention.

### Environment & Build Tips
- Backend selection in vLLM:  
  `VLLM_ATTENTION_BACKEND` = `TORCH_SDPA` | `FLASH_ATTN` | `XFORMERS` | `ROCM_FLASH` | `FLASHINFER` | `FLASHMLA`  
  On T4 GPUs, **`XFORMERS`** gave the highest throughput.
- Avoid crashes on NYCU T4 servers by adjusting CUDA graph settings:
```python
compilation_config = {
    "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
    "max_capture_size": 16,
}
````

---

## Experimental Results

**Hardware:** NVIDIA Tesla T4
**Dataset:** wikitext-2-raw-v1
Base model perplexity: 11.12

| Method                      | Framework | Throughput (tok/s) | PPL   |
| --------------------------- | --------- | ------------------ | ----- |
| GPTQ                        | vLLM      | 84.01              | 11.12 |
| GPTQ + Speculative Decoding | vLLM      | **91.97**          | 11.12 |
| GPTQ                        | SGLang    | 90.10              | 11.12 |
| GPTQ + Speculative Decoding | SGLang    | 57.10              | 11.12 |
| AWQ + Speculative Decoding  | vLLM      | 71.87              | 10.78 |
| AWQ                         | vLLM      | 19.35              | 10.78 |
| HQQ (baseline)              | –         | 3.90               | 11.23 |

![Throughput Plot](https://hackmd.io/_uploads/HyCYB2pflx.png)
![Comparison](https://hackmd.io/_uploads/ryFG_3pMge.png)

**Key observations**

* GPTQ consistently outperforms AWQ in throughput on both vanilla and speculative decoding.
* Without speculative decoding, **SGLang > vLLM** due to better low-level kernel optimizations and Radix Attention.
* With speculative decoding, **vLLM > SGLang** because SGLang’s EAGLE-based approach restricts draft–target model choices and causes resource contention.

---

## Additional Insignts

### Radix vs Paged Attention

| Metric                | **Radix Attention (SGLang)**            | **Paged Attention (vLLM)**                    |
| --------------------- | --------------------------------------- | --------------------------------------------- |
| Problem solved        | Improves softmax computation efficiency | Efficient memory management for long contexts |
| Core idea             | Divide-and-conquer block attention      | KV-cache paging (like virtual memory)         |
| Best use case         | High-speed training / inference         | Long-context tasks (RAG, document QA)         |
| Large-context support | No                                      | Yes                                           |
| Inference-speed gain  | Significant                             | Significant                                   |

**JIT Layout Planning** in SGLang optimizes KV-cache memory layout at runtime, considering batch shape, prompt length, and GPU memory alignment.
Without speculative decoding, this contributes to a **+6.19 tok/s** gain over vLLM.

---

## References

* Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021
* Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, 2022
* Lin et al., *AWQ: Activation-Aware Weight Quantization for LLMs*, 2024
* Xu et al., *EAGLE: Speculative Decoding*, 2024
* Kwon et al., *Radix Attention*, 2023
* vLLM: [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
* SGLang: [https://github.com/sgl-project/sglang](https://github.com/sgl-project/sglang)
