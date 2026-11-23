"""
GPT2 Fine-tuning Script
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

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
except Exception:
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

    pad_id = tokenizer.pad_token_id
    tokenized["labels"] = [
        [-100 if tok == pad_id else tok for tok in row]
        for row in tokenized["labels"]
    ]

    return tokenized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="gpt2")
    parser.add_argument("--data_path", default="./medical_processed/clinicalnotes")
    parser.add_argument("--output_dir", default="./models/medical_simple")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_fp16", action="store_true", help="Enable fp16 training if supported")
    parser.add_argument("--use_bf16", action="store_true", help="Enable bf16 training if supported")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(f"Process {local_rank}/{world_size} starting...")

    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True
    )

    tokenizer.padding_side = "right"
    added_pad = False
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        added_pad = True
        if local_rank == 0:
            print("Added [PAD] token to tokenizer")
    else:
        if local_rank == 0:
            print(f"PAD token already present: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    
    
    model.config.use_cache = False 

    if added_pad:
        old_n = model.get_input_embeddings().num_embeddings
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        if local_rank == 0:
            print(f"Resized embeddings: {old_n} -> {len(tokenizer)}; pad_token_id={tokenizer.pad_token_id}")

    model.cuda()

    if local_rank == 0:
        print("Using full fine-tuning")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    if local_rank == 0:
        print(f"Loading datasets from {args.data_path}")
    train_dataset = load_from_disk(f"{args.data_path}/train_simple")
    val_dataset = load_from_disk(f"{args.data_path}/val_simple")

    val_dataset = val_dataset.train_test_split(test_size=0.1, seed=42)["test"]

    if local_rank == 0:
        print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    train_size = len(train_dataset)
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps * max(1, world_size)
    steps_per_epoch = max(1, train_size // max(1, effective_batch_size))
    save_steps = max(1, steps_per_epoch // 2)

    if local_rank == 0:
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Will save every {save_steps} steps (~0.5 epochs)")

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train dataset"
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation dataset"
    )

    if local_rank == 0:
        def _pad_overhead(ds, max_len):
            am = ds["attention_mask"]
            nonpad = sum(sum(row) for row in am)
            total = len(ds) * max_len
            pad = total - nonpad
            pct = 100.0 * pad / total if total > 0 else 0.0
            return pct, pad, total

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8, 
        return_tensors="pt",
    )

    use_bf16 = args.use_bf16 and bf16_supported
    use_fp16 = args.use_fp16 and (torch.cuda.is_available() and not use_bf16)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size * 2),
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        

        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=max(10, steps_per_epoch // 2), 
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=save_steps,
        load_best_model_at_end=False,
        save_total_limit=1000,
        logging_steps=10,
        logging_strategy="steps",
        report_to=None,
        bf16=use_bf16,
        fp16=use_fp16,
        bf16_full_eval=use_bf16,
        fp16_full_eval=use_fp16,
        gradient_checkpointing=True,
        dataloader_num_workers=1,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        optim="adamw_torch",
        group_by_length=False, 
        max_grad_norm=args.max_grad_norm,
        seed=42,
        logging_nan_inf_filter=True,
        skip_memory_metrics=True,
    )

    class NaNDetectionCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                for key, value in logs.items():
                    if isinstance(value, float) and (torch.isnan(torch.tensor(value)) or torch.isinf(torch.tensor(value))):
                        print(f"WARNING: {key} is {value} at step {state.global_step}")
                        if key in ['train_loss', 'eval_loss'] and torch.isnan(torch.tensor(value)):
                            print("Stopping training due to NaN loss")
                            control.should_training_stop = True

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[NaNDetectionCallback()]
    )

    if local_rank == 0:
        print("Starting training on dataset...")
        print(f"- Model: {args.model_name}")
        print(f"- Learning rate: {args.learning_rate}")
        print(f"- Weight decay: {args.weight_decay}")
        print(f"- Grad accum: {args.gradient_accumulation_steps}")
        print(f"- Max grad norm: {args.max_grad_norm}")
        print(f"- Mixed precision: bf16={use_bf16}, fp16={use_fp16}")
        print(f"- Training samples: {len(train_dataset)}, Val: {len(val_dataset)}")
        print(f"- Steps/epoch: {steps_per_epoch}, Total steps: {steps_per_epoch * args.epochs}")

    try:
        trainer.train()
    except Exception as e:
        print(f"Training error: {e}")
        print("If OOM, reduce batch_size or max_length, or enable fp16/bf16 if stable.")
        raise

    if local_rank == 0:
        print(f"Saving model to {args.output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        print("Training completed!")


if __name__ == "__main__":
    main()
