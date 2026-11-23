"""
Llama3.2 1B Fine-tuning Script
"""

import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
import argparse
import warnings

try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    if transformers.__version__ < "4.43.0":
        warnings.warn("Consider upgrading transformers for better Llama 3.2 support")
except:
    pass

def tokenize_function(examples, tokenizer, max_length=512):
    texts = examples["text"]
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors=None,
    )

    tokenized["labels"] = tokenized["input_ids"].copy()

    tokenized["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in tokenized["labels"]
    ]

    if "attention_mask" not in tokenized:
        tokenized["attention_mask"] = [
            [1 if token != tokenizer.pad_token_id else 0 for token in input_ids]
            for input_ids in tokenized["input_ids"]
        ]

    return tokenized

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--data_path", default="./medical_processed/clinicalnotes")
    parser.add_argument("--output_dir", default="./models/llama32_1b_clinicalnotes")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_bf16", action="store_true", help="Use bfloat16 precision")
    parser.add_argument("--use_fp16", action="store_true", help="Use float16 precision")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    print(f"[Rank {local_rank}] Starting with Llama-3.2-1B...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True
    )

    tokenizer.padding_side = "right"

    dtype = torch.float32
    if args.use_bf16 and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        print(f"[Rank {local_rank}] Using bfloat16 precision")
    elif args.use_fp16:
        dtype = torch.float16
        print(f"[Rank {local_rank}] Using float16 precision")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "eager"
    ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        print(f"[Rank {local_rank}] Added [PAD]; resized embeddings to {len(tokenizer)}")

    model.config.pad_token_id = tokenizer.pad_token_id

    model.config.use_cache = False

    if local_rank == 0:
        print(f"[Rank 0] Using pad_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
        print(f"[Rank 0] Using eos_token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")
        print(f"[Rank 0] Model size: {sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters")
        print("[Rank 0] Loading dataset...")

    train_dataset = load_from_disk(f"{args.data_path}/train_simple")
    val_dataset = load_from_disk(f"{args.data_path}/val_simple")

    val_dataset = val_dataset.train_test_split(test_size=0.1, seed=42)["test"]

    if local_rank == 0:
        import numpy as np
        print("[Rank 0] Analyzing sequence lengths...")
        sample_size = min(100, len(train_dataset))
        lengths = [len(tokenizer.encode(text)) for text in train_dataset['text'][:sample_size]]
        print(f"[Rank 0] Token lengths - Avg: {np.mean(lengths):.1f}, Max: {max(lengths)}, Min: {min(lengths)}")
        print(f"[Rank 0] Padding overhead: {(args.max_length - np.mean(lengths))/args.max_length*100:.1f}% of tokens are padding")
        if np.mean(lengths) > args.max_length:
            print(f"[Rank 0] WARNING: Average length ({np.mean(lengths):.1f}) exceeds max_length ({args.max_length}). Consider increasing max_length.")

    effective_batch_size = args.batch_size * args.gradient_accumulation_steps * world_size
    steps_per_epoch = max(1, len(train_dataset) // effective_batch_size)
    save_steps = max(steps_per_epoch // 2, 100)

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=4 if local_rank == 0 else 1
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=val_dataset.column_names,
        num_proc=4 if local_rank == 0 else 1
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )


    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=1,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=max(10, steps_per_epoch // 10),
        lr_scheduler_type="cosine",
        eval_strategy="no",
        save_strategy="steps",
        save_steps=save_steps,
        load_best_model_at_end=False,
        save_total_limit=10,
        logging_steps=10,
        logging_strategy="steps",
        report_to=None,
        fp16=args.use_fp16 and not args.use_bf16,
        bf16=args.use_bf16 and torch.cuda.is_bf16_supported(),
        optim="adamw_torch",
        group_by_length=False,
        max_grad_norm=args.max_grad_norm,
        seed=42,
        logging_nan_inf_filter=True,
        skip_memory_metrics=True,
        local_rank=local_rank,
        gradient_checkpointing=True,
    )

    class NaNDetectionCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                for key, value in logs.items():
                    if isinstance(value, float) and (torch.isnan(torch.tensor(value)) or torch.isinf(torch.tensor(value))):
                        print(f"[Rank {local_rank}] WARNING: {key} is {value} at step {state.global_step}")
                        if key in ['train_loss', 'eval_loss']:
                            control.should_training_stop = True

    model.gradient_checkpointing_enable()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if val_dataset else None,
        data_collator=data_collator,
        callbacks=[NaNDetectionCallback()]
    )

    if local_rank == 0:
        print("[Rank 0] Starting training...")
        print(f"[Rank 0] Total training steps: {steps_per_epoch * args.epochs}")

    trainer.train()

    if local_rank == 0:
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        print("[Rank 0] Training complete. Model saved to", args.output_dir)


if __name__ == "__main__":
    main()
