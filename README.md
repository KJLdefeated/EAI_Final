# Edge AI Final Project: LLM Acceleration

## Features
1. Lora Fine-tune
2. GPTQ
3. Draft model for speculative decoding

## Environment Setup
clone this repo
```bash
git clone https://github.com/KJLdefeated/EAI_Final.git && cd EAI_Final
```
(optional) create a virtual environment (!important! please use python 3.11)
```bash
bash
conda create -n eai python=3.11 -y
```
install the requirements
```bash
pip install -r requirements.txt
```

## Pull From HF Hub and Evaluate
Remember to use `huggingface-cli login` with a valid token with `Read access to contents of all public gated repos you can access`

```bash
python result_tput.py
python result_ppl.py
```

## Quantize the LLMs from the a fine-tuned lora model
```bash
python merge_lora.py
python quant_gptq.py
```
Then make sure to modify the models in `result_*.py` into the saved model path.
```bash
python result_tput.py
python result_ppl.py
```
