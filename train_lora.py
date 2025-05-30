from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer
import torch


def get_dataset(tokenizer, split='train'):
    wiki_dataset = load_dataset(path="wikitext", name="wikitext-2-raw-v1", split=split)

    # Function to preprocess dataset
    def preprocess_function(examples):
        text_column = "text"
        texts = examples[text_column]
        
        # Filter out empty texts
        texts = [text for text in texts if len(text.strip()) > 0]
        
        tokenized_inputs = tokenizer(
            texts, 
            truncation=True,
            max_length=256,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create language modeling labels
        tokenized_inputs['labels'] = tokenized_inputs['input_ids'].clone()
        tokenized_inputs['labels'][tokenized_inputs['input_ids'] == tokenizer.pad_token_id] = -100
        return tokenized_inputs

    # Preprocess the dataset
    tokenized_wiki = wiki_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=wiki_dataset.column_names,
    )

    return tokenized_wiki


def get_lora_config(rank=16, alpha=32, dropout=0.05):
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
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
    return lora_config


def get_lora_model(model_name, lora_config):
    model = AutoModelForCausalLM.from_pretrained( 
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    lora_model = get_peft_model(model, lora_config)
    return lora_model


def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def run():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    lora_config = get_lora_config(rank=16, alpha=32)
    lora_model = get_lora_model(model_name, lora_config)
    tokenizer = get_tokenizer(model_name)
    train_dataset = get_dataset(tokenizer, split="train[:15000]")
    
    training_args = TrainingArguments(
        run_name="Llama-3.2-1B-Instruct-lora-finetuing",
        output_dir="Llama-3.2-1B-Instruct-lora-checkpoints",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=10,
        logging_steps=10,
        save_steps=10,
        learning_rate=1e-5,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        lr_scheduler_type="constant",
        report_to="wandb",
        seed=48763,
    )

    trainer = SFTTrainer(
        model=lora_model,
        train_dataset=train_dataset,
        peft_config=lora_config,
        args=training_args,
    )

    trainer.train()


if __name__ == '__main__':
    run()
