import yaml
import argparse
import torch
from trl import SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments
from unsloth import FastLanguageModel, is_bfloat16_supported, to_sharegpt, standardize_sharegpt, apply_chat_template

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main(config):
    # Load model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['model']['name'],
        max_seq_length=config['model']['max_seq_length'],
        load_in_4bit=config['model']['load_in_4bit'],
        dtype=None #config['model']['dtype'],
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=config['model']['peft']['r'],
        lora_alpha=config['model']['peft']['lora_alpha'],
        lora_dropout=config['model']['peft']['lora_dropout'],
        target_modules=config['model']['peft']['target_modules'],
        use_rslora=config['model']['peft']['use_rslora'],
        use_gradient_checkpointing=config['model']['peft']['use_gradient_checkpointing']
    )
    print(model.print_trainable_parameters())

    # Load dataset
    dataset = load_dataset(config['dataset']['name'], split=config['dataset']['split'])
    dataset = to_sharegpt(
        dataset,
        merged_prompt=config['dataset']['merged_prompt'],
        output_column_name=config['dataset']['output_column_name'],
        conversation_extension=config['dataset']['conversation_extension'],
    )
    dataset = standardize_sharegpt(dataset)

    # Apply chat template
    chat_template = config['template']['content']
    dataset = apply_chat_template(
        dataset,
        tokenizer=tokenizer,
        chat_template=chat_template
    )

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        warmup_steps=config['training']['warmup_steps'],
        max_steps=config['training']['max_steps'],
        learning_rate=config['training']['learning_rate'],
        fp16=not is_bfloat16_supported(),#config['training']['fp16'],
        bf16=is_bfloat16_supported(), #config['training']['bf16'],
        logging_steps=config['training']['logging_steps'],
        optim=config['training']['optim'],
        weight_decay=config['training']['weight_decay'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        seed=config['training']['seed'],
        output_dir=config['training']['output_dir'],
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config['model']['max_seq_length'],
        dataset_num_proc=2,
        packing=True,
        args=training_args,
    )
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LLMs with configurations from a YAML file.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)
    main(config)