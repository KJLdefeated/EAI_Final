import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

from hqq.core.quantize import BaseQuantizeConfig
from hqq_utils import AutoHQQHFModel

def get_quant_config_slm(model):
    quant_config = {}
    
    n_layers = model.config.num_hidden_layers
    q2_config = BaseQuantizeConfig(nbits=2, group_size=64)
    q4_config = BaseQuantizeConfig(nbits=4, group_size=64)
    q8_config = BaseQuantizeConfig(nbits=8, group_size=64)

    for i in range(n_layers):
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = q4_config
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = q2_config
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = q2_config
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = q4_config
        
        quant_config[f'model.layers.{i}.mlp.gate_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.up_proj'] = q4_config
        quant_config[f'model.layers.{i}.mlp.down_proj'] = q4_config
        
    return quant_config

def prepare_dataset():
    wiki_dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # Function to preprocess dataset
    def preprocess_function(examples):
        text_column = "text"
        texts = examples[text_column]
        
        # Filter out empty texts
        texts = [text for text in texts if len(text.strip()) > 0]
        
        tokenized_inputs = tokenizer(
            texts, 
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create language modeling labels
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
        
        return tokenized_inputs

    # Preprocess the dataset
    tokenized_wiki = wiki_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=wiki_dataset["train"].column_names,
    ) 

    return tokenized_wiki


if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.2-3B-Instruct"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # First load the model in FP16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Quantize the model using HQQ
    quant_config = get_quant_config_slm(model)
    AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.float16, device="cuda")

    # Prepare the model for LoRA training
    quantized_model = prepare_model_for_kbit_training(model)

    # Define LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
    )

    # Apply LoRA adapters
    lora_model = get_peft_model(quantized_model, peft_config)
    tokenized_wiki = prepare_dataset()

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./llama-3-2-3b-hqq-lora-wiki2",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        warmup_ratio=0.1,
        logging_steps=50,
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="tensorboard",
        lr_scheduler_type="cosine",
        fp16=True,
        remove_unused_columns=False,
    )

    # Initialize SFT Trainer
    trainer = SFTTrainer(
        model=lora_model,
        args=training_args,
        train_dataset=tokenized_wiki["train"],
        eval_dataset=tokenized_wiki["validation"],
        peft_config=peft_config,
    )

    # Start training
    trainer.train()

    # Save the final model
    lora_model.save_pretrained("./llama-3-2-3b-hqq-lora-wiki2-final-kv-2-bit")