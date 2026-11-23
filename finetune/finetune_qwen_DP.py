"""
Qwen3 0.6B with Differential Privacy (DP)
Full Opacus integration. Uniform (non-Poisson) subsampling.
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
import argparse
from tqdm import tqdm
import math

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    from opacus.utils.batch_memory_manager import BatchMemoryManager
except ImportError:
    raise ImportError("Install opacus: pip install opacus")


def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize with proper label masking"""
    texts = examples["text"]
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    

    labels = tokenized["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }


class SimpleDataset(torch.utils.data.Dataset):
    """Simple dataset wrapper for tokenized data"""
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.data[idx]["input_ids"],
            "attention_mask": self.data[idx]["attention_mask"],
            "labels": self.data[idx]["labels"]
        }


def train_with_dp(
    model,
    train_loader,
    val_loader,
    optimizer,
    privacy_engine,
    args,
    tokenizer
):
    """Training loop with DP"""
    
    device = next(model.parameters()).device
    steps_per_epoch = len(train_loader)
    num_training_steps = steps_per_epoch * args.epochs
    
    warmup_steps = max(10, steps_per_epoch // 10)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    print(f"\n{'='*60}")
    print("Starting DP Training")
    print(f"{'='*60}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Warmup steps: {warmup_steps} ({warmup_steps/steps_per_epoch*100:.1f}% of epoch)")
    print(f"Total steps: {num_training_steps}")
    print(f"LR Scheduler: Cosine with warmup")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    best_val_loss = float('inf')
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        if args.physical_batch_size is not None:
            with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=args.physical_batch_size,
                optimizer=optimizer
            ) as memory_safe_data_loader:
                progress_bar = tqdm(memory_safe_data_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
                
                for batch in progress_bar:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss

                    loss.backward()

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    total_loss += loss.item()
                    global_step += 1

                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                    })
                    
                    if global_step % args.log_steps == 0:
                        epsilon = privacy_engine.get_epsilon(args.target_delta)
                        print(f"\nStep {global_step}: ε = {epsilon:.2f} (target: {args.target_epsilon})")
        else:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
            
            for batch in progress_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

                loss.backward()

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
                global_step += 1

                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })

                if global_step % args.log_steps == 0:
                    epsilon = privacy_engine.get_epsilon(args.target_delta)
                    print(f"\nStep {global_step}: ε = {epsilon:.2f} (target: {args.target_epsilon})")


        avg_train_loss = total_loss / len(train_loader)

        epsilon = privacy_engine.get_epsilon(args.target_delta)

        epoch_save_path = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        os.makedirs(epoch_save_path, exist_ok=True)

        if hasattr(model, '_module'):
            model._module.save_pretrained(epoch_save_path)
        else:
            model.save_pretrained(epoch_save_path)
        tokenizer.save_pretrained(epoch_save_path)
        print(f"  💾 Saved checkpoint: {epoch_save_path}")

        try:
            val_loss = evaluate(model, val_loader, device)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Skipping validation due to OOM")
                val_loss = float('nan')
                torch.cuda.empty_cache()
            else:
                raise e
        
        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        if not torch.isnan(torch.tensor(val_loss)):
            print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Privacy: ε = {epsilon:.2f}, δ = {args.target_delta:.1e}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

        epoch_privacy_metadata = {
            "epoch": epoch + 1,
            "epsilon": epsilon,
            "delta": args.target_delta,
            "train_loss": avg_train_loss,
            "val_loss": val_loss if not torch.isnan(torch.tensor(val_loss)) else None,
            "max_grad_norm": args.max_grad_norm,
            "noise_multiplier": getattr(optimizer, 'noise_multiplier', None),
        }
        import json
        with open(os.path.join(epoch_save_path, "privacy_metadata.json"), "w") as f:
            json.dump(epoch_privacy_metadata, f, indent=2)

        if not torch.isnan(torch.tensor(val_loss)) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_save_path = os.path.join(args.output_dir, "best_model")
            os.makedirs(best_save_path, exist_ok=True)

            if hasattr(model, '_module'):
                model._module.save_pretrained(best_save_path)
            else:
                model.save_pretrained(best_save_path)
            tokenizer.save_pretrained(best_save_path)
            print(f"Saved best model (val_loss: {val_loss:.4f})")

        if epsilon >= args.target_epsilon:
            print(f"\n Privacy budget exhausted (ε = {epsilon:.2f} >= {args.target_epsilon})")
            print("Stopping early to maintain privacy guarantee.")
            break

    final_epsilon = privacy_engine.get_epsilon(args.target_delta)
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Final Privacy: (ε = {final_epsilon:.2f}, δ = {args.target_delta:.1e})")
    print(f"{'='*60}\n")
    
    return model


def evaluate(model, dataloader, device, max_batches=100):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if num_batches >= max_batches:
                break
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_loss += outputs.loss.item()
            num_batches += 1

            if num_batches % 20 == 0:
                torch.cuda.empty_cache()
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--data_path", type=str, default="./general_processed/wikipedia")
    parser.add_argument("--output_dir", type=str, default="./output_dp")
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--target_epsilon", type=float, default=8.0,
                       help="Target privacy budget (lower = more private)")
    parser.add_argument("--target_delta", type=float, default=1e-5,
                       help="Target delta (typically 1/n)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Gradient clipping threshold")

    parser.add_argument("--log_steps", type=int, default=100)

    parser.add_argument("--physical_batch_size", type=int, default=None,
                       help="Split logical batch into smaller physical batches")
    
    args = parser.parse_args()
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Differential Privacy Training Setup")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Target Privacy: ε={args.target_epsilon}, δ={args.target_delta}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Max Grad Norm: {args.max_grad_norm}")
    print(f"{'='*60}\n")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        use_fast=True
    )
    
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    
    if tokenizer.pad_token_id != model.config.pad_token_id:
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    print("Making model DP-compatible...")
    model = ModuleValidator.fix(model)
    model.to(device)

    print("Loading datasets...")
    train_data = load_from_disk(f"{args.data_path}/train_simple")
    val_data = load_from_disk(f"{args.data_path}/val_simple")

    val_data = val_data.train_test_split(test_size=0.1, seed=42)["test"]
    
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")

    recommended_delta = 1.0 / len(train_data)
    if args.target_delta >= recommended_delta:
        print(f"Warning: delta={args.target_delta:.2e} should be < 1/n={recommended_delta:.2e}")
        print(f"   Consider using delta={recommended_delta:.2e} or smaller")

    print("Tokenizing...")
    train_tokenized = []
    for example in tqdm(train_data, desc="Tokenizing train"):
        tokenized = tokenize_function({"text": [example["text"]]}, tokenizer, args.max_length)
        train_tokenized.append({
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
            "labels": tokenized["labels"][0]
        })
    
    val_tokenized = []
    for example in tqdm(val_data, desc="Tokenizing val"):
        tokenized = tokenize_function({"text": [example["text"]]}, tokenizer, args.max_length)
        val_tokenized.append({
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
            "labels": tokenized["labels"][0]
        })

    train_dataset = SimpleDataset(train_tokenized)
    val_dataset = SimpleDataset(val_tokenized)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,  
        num_workers=0,
        pin_memory=True,
        drop_last=True,  
        generator=torch.Generator().manual_seed(42)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    print("\nSetting up Privacy Engine...")

    model.train()
    
    privacy_engine = PrivacyEngine()

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=args.epochs,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        max_grad_norm=args.max_grad_norm,
        poisson_sampling=False, 
    )
    
    sample_rate = args.batch_size / len(train_dataset)
    
    print(f"Privacy Engine configured:")
    print(f"Sampling: Uniform (Poisson disabled due to Opacus bug)")
    try:
        noise_mult = optimizer.noise_multiplier
        print(f"   Noise multiplier: {noise_mult:.3f}")
    except AttributeError:
        print(f"   Noise multiplier: [computed automatically]")
    print(f"Sample rate: {sample_rate:.4f}")
    print(f"Max grad norm: {args.max_grad_norm}")
    print(f"Expected ε after {args.epochs} epochs: {args.target_epsilon}")

    model = train_with_dp(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        privacy_engine=privacy_engine,
        args=args,
        tokenizer=tokenizer
    )

    print(f"Saving final model to {args.output_dir}")

    if hasattr(model, '_module'):
        model._module.save_pretrained(args.output_dir)
    else:
        model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    final_epsilon = privacy_engine.get_epsilon(args.target_delta)
    privacy_metadata = {
        "epsilon": final_epsilon,
        "delta": args.target_delta,
        "max_grad_norm": args.max_grad_norm,
        "noise_multiplier": getattr(optimizer, 'noise_multiplier', None),
        "epochs": args.epochs,
        "sampling_method": "uniform"
    }
    
    import json
    with open(os.path.join(args.output_dir, "privacy_metadata.json"), "w") as f:
        json.dump(privacy_metadata, f, indent=2)
    
    print(f"\nPrivacy metadata saved")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
