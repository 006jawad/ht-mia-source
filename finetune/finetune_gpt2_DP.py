"""
GPT2 Fine-tuning Script with Differential Privacy (DP)
"""

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import warnings
warnings.filterwarnings("ignore", message=".*Secure RNG turned off.*")
warnings.filterwarnings("ignore", message=".*Optimal order is the largest alpha.*")
warnings.filterwarnings("ignore", message=".*consider expanding the range of alphas.*")

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

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
except ImportError:
    raise ImportError("Install opacus: pip install opacus")


def tokenize_function(examples, tokenizer, max_length=512):
    texts = examples["text"]

    if isinstance(texts, str):
        texts = [texts]

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
    def __init__(self, data, max_length):
        self.data = data
        self.max_length = max_length

        print("Validating dataset shapes...")
        for idx, item in enumerate(self.data):
            for key in ['input_ids', 'attention_mask', 'labels']:
                tensor = item[key]
                if tensor.shape[0] != max_length:
                    raise ValueError(f"Item {idx} {key} has shape {tensor.shape}, expected ({max_length},)")
        print(f"All {len(self.data)} items validated")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        result = {
            "input_ids": item["input_ids"].squeeze() if item["input_ids"].dim() > 1 else item["input_ids"],
            "attention_mask": item["attention_mask"].squeeze() if item["attention_mask"].dim() > 1 else item["attention_mask"],
            "labels": item["labels"].squeeze() if item["labels"].dim() > 1 else item["labels"]
        }

        assert result["input_ids"].shape[0] == self.max_length
        assert result["attention_mask"].shape[0] == self.max_length
        assert result["labels"].shape[0] == self.max_length

        return result


def collate_fn(batch):
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    first_shape = batch[0]["input_ids"].shape
    for idx, item in enumerate(batch):
        for key in ['input_ids', 'attention_mask', 'labels']:
            if item[key].shape != first_shape:
                print(f"Shape mismatch at batch item {idx}: {key}")
                return None

    try:
        result = {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch])
        }

        batch_size = len(batch)
        seq_len = first_shape[0]
        expected_shape = (batch_size, seq_len)

        assert result["input_ids"].shape == expected_shape
        assert result["attention_mask"].shape == expected_shape
        assert result["labels"].shape == expected_shape

        return result

    except Exception as e:
        print(f"Collate error: {e}")
        return None


def freeze_position_embeddings(model):
    print("\nFreezing position embeddings for DP compatibility...")

    frozen_params = []

    if hasattr(model, 'transformer') and hasattr(model.transformer, 'wpe'):
        model.transformer.wpe.weight.requires_grad = False
        frozen_params.append('transformer.wpe.weight')
        print(f"Froze: transformer.wpe.weight (position embeddings)")

    for name, param in model.named_parameters():
        if 'position_embedding' in name.lower() or 'pos_emb' in name.lower():
            param.requires_grad = False
            frozen_params.append(name)
            print(f"Froze: {name}")

    if not frozen_params:
        print("No position embeddings found to freeze")

    return frozen_params


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

    print("Starting DP Training")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Device: {device}")

    best_val_loss = float('inf')
    global_step = 0
    successful_steps = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        epoch_successful_steps = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            if batch is None:
                continue

            try:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
            except Exception as e:
                print(f"Error moving to device: {e}")
                continue

            try:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
            except Exception as e:
                print(f"Forward failed: {e}")
                optimizer.zero_grad()
                continue

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss: {loss.item()}")
                optimizer.zero_grad()
                continue

            try:
                loss.backward()
            except Exception as e:
                print(f"Backward failed: {e}")
                optimizer.zero_grad()
                continue

            try:
                optimizer.step()
                successful_steps += 1
                epoch_successful_steps += 1
            except RuntimeError as e:
                error_msg = str(e)
                print(f"\nStep {global_step}: Optimizer step failed")
                print(f"   Error: {error_msg[:200]}")

                if "stack expects each tensor" in error_msg:
                    print(f"   Checking gradient shapes...")
                    for name, param in model.named_parameters():
                        if param.requires_grad and hasattr(param, 'grad_sample'):
                            if param.grad_sample is not None:
                                print(f"{name}: {param.grad_sample.shape}")

                optimizer.zero_grad()
                continue

            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            global_step += 1

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                'success': f'{epoch_successful_steps}/{batch_idx+1}'
            })

            if successful_steps > 0 and global_step % args.log_steps == 0:
                try:
                    epsilon = privacy_engine.get_epsilon(args.target_delta)
                    print(f"\nStep {global_step}: ε = {epsilon:.2f}")
                    print(f"   Success: {successful_steps}/{global_step} ({100*successful_steps/global_step:.1f}%)")
                except:
                    pass

        if epoch_successful_steps > 0:
            avg_loss = total_loss / epoch_successful_steps
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"Average loss: {avg_loss:.4f}")
            print(f"Successful steps: {epoch_successful_steps}/{steps_per_epoch}")

            print("Running validation...")
            model.eval()
            with torch.no_grad():
                val_loss = evaluate(model, val_loader, device)
            model.train()
            print(f"Val loss: {val_loss:.4f}")


            print(f"Saving epoch {epoch+1} checkpoint...")
            epoch_checkpoint_dir = f"{args.output_dir}/checkpoint_epoch_{epoch+1}"
            os.makedirs(epoch_checkpoint_dir, exist_ok=True)

            if hasattr(model, '_module'):
                model._module.save_pretrained(epoch_checkpoint_dir)
            else:
                model.save_pretrained(epoch_checkpoint_dir)
            tokenizer.save_pretrained(epoch_checkpoint_dir)

            import json
            training_state = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "successful_steps": successful_steps,
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            try:
                training_state["epsilon"] = privacy_engine.get_epsilon(args.target_delta)
            except:
                training_state["epsilon"] = None

            with open(os.path.join(epoch_checkpoint_dir, "training_state.json"), "w") as f:
                json.dump(training_state, f, indent=2)

            print(f"Saved to {epoch_checkpoint_dir}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"New best validation loss! Saving best checkpoint...")
                best_checkpoint_dir = f"{args.output_dir}/best_checkpoint"
                os.makedirs(best_checkpoint_dir, exist_ok=True)

                if hasattr(model, '_module'):
                    model._module.save_pretrained(best_checkpoint_dir)
                else:
                    model.save_pretrained(best_checkpoint_dir)
                tokenizer.save_pretrained(best_checkpoint_dir)

                with open(os.path.join(best_checkpoint_dir, "training_state.json"), "w") as f:
                    json.dump(training_state, f, indent=2)

            try:
                epsilon = privacy_engine.get_epsilon(args.target_delta)
                print(f"   Current ε: {epsilon:.2f}")
            except:
                pass
        else:
            print(f"\nEpoch {epoch+1}: No successful steps!")

    print(f"Training completed!")
    print(f"Total successful steps: {successful_steps}/{global_step}")
    if successful_steps > 0:
        try:
            final_epsilon = privacy_engine.get_epsilon(args.target_delta)
            print(f"Final ε: {final_epsilon:.2f}")
        except:
            pass

    return model


def evaluate(model, val_loader, device, max_batches=None):
    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(val_loader):
        if batch is None:
            continue

        if max_batches and batch_idx >= max_batches:
            break

        try:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                num_batches += 1
        except:
            continue

    return total_loss / max(num_batches, 1)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./models/gpt2_medical_dp")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--target_epsilon", type=float, default=10.0)
    parser.add_argument("--target_delta", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--log_steps", type=int, default=50)

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token = eos_token")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32
    )

    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False

    if hasattr(model.config, 'output_attentions'):
        model.config.output_attentions = False
    if hasattr(model.config, 'output_hidden_states'):
        model.config.output_hidden_states = False

    frozen_params = freeze_position_embeddings(model)

    print("\nMaking model DP-compatible...")
    model = ModuleValidator.fix(model)

    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        print("Model validation warnings:")
        for error in errors[:3]:
            print(f"   - {error}")

    model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

    print("Loading datasets...")
    train_data = load_from_disk(f"{args.data_path}/train_simple")
    val_data = load_from_disk(f"{args.data_path}/val_simple")

    val_data = val_data.train_test_split(test_size=0.1, seed=42)["test"]

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")

    print("Tokenizing...")
    train_tokenized = []
    for example in tqdm(train_data, desc="Tokenizing train"):
        tokenized = tokenize_function({"text": example["text"]}, tokenizer, args.max_length)
        item = {
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
            "labels": tokenized["labels"][0]
        }
        assert item["input_ids"].shape[0] == args.max_length
        train_tokenized.append(item)

    val_tokenized = []
    for example in tqdm(val_data, desc="Tokenizing val"):
        tokenized = tokenize_function({"text": example["text"]}, tokenizer, args.max_length)
        item = {
            "input_ids": tokenized["input_ids"][0],
            "attention_mask": tokenized["attention_mask"][0],
            "labels": tokenized["labels"][0]
        }
        assert item["input_ids"].shape[0] == args.max_length
        val_tokenized.append(item)

    print("\nVerifying tokenized data...")
    sample = train_tokenized[0]
    print(f"input_ids: {sample['input_ids'].shape}")
    print(f"All tensors are 1D with length {args.max_length}")

    train_dataset = SimpleDataset(train_tokenized, args.max_length)
    val_dataset = SimpleDataset(val_tokenized, args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
        generator=torch.Generator().manual_seed(42)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False
    )

    print("\nTesting dataloader...")
    test_batch = next(iter(train_loader))
    if test_batch is not None:
        print(f"Batch input_ids: {test_batch['input_ids'].shape}")
        print(f"Dataloader working correctly")
    else:
        raise RuntimeError("First batch is None!")

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
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
        grad_sample_mode="hooks",
    )

    print(f"Privacy Engine configured")
    print(f"Max grad norm: {args.max_grad_norm}")
    print(f"Target ε: {args.target_epsilon}")
    print(f"Frozen parameters: {len(frozen_params)}")
    print(f"\nStarting training (this may take a moment)...\n")

    model = train_with_dp(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        privacy_engine=privacy_engine,
        args=args,
        tokenizer=tokenizer
    )

    print(f"Saving model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    if hasattr(model, '_module'):
        model._module.save_pretrained(args.output_dir)
    else:
        model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nDone!\n")


if __name__ == "__main__":
    main()
