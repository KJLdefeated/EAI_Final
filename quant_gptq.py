import numpy as np
import torch

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


tokenizer_model = "meta-llama/Llama-3.2-3B-Instruct"
pretrained_model_dir = "Llama-3.2-3B-Instruct-lora"
quantized_model_dir = "Llama-3.2-3B-Instruct-lora-4bit-g128"


def get_wikitext2(nsamples, seed, seqlen):
    from datasets import load_dataset

    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    traindata = traindata.filter(lambda x: len(x["text"]) > 128)

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    
    tokenizer.save_pretrained(quantized_model_dir)

    import random

    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)

    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})

    return traindataset


def main():
    traindataset = get_wikitext2(nsamples=256, seed=0, seqlen=2048)

    quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
        desc_act=True,  # desc_act and group size only works on triton
    )

    # load un-quantized model, the model will always be force loaded into cpu
    model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    # with value under torch.LongTensor type.
    model.quantize(traindataset, use_triton=True)

    # save quantized model
    model.save_quantized(quantized_model_dir)

    # save quantized model using safetensors
    model.save_quantized(quantized_model_dir, use_safetensors=True)


if __name__ == "__main__":

    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    main()
