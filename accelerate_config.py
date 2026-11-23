#!/usr/bin/env python3
"""
Accelerate configuration helper for fine-tuning
"""

import subprocess
import sys
import os
from pathlib import Path

def check_accelerate_installed():
    try:
        import accelerate
        return True
    except ImportError:
        return False

def get_gpu_count():
    try:
        import torch
        return torch.cuda.device_count()
    except:
        return 0

def create_accelerate_config():
    gpu_count = get_gpu_count()

    if gpu_count == 0:
        print("No GPUs detected. Creating CPU config...")
        config_content = """compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: NO
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: true
"""
    elif gpu_count == 1:
        print("Single GPU detected. Creating single GPU config...")
        config_content = """compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: NO
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""
    else:
        print(f"Multiple GPUs detected ({gpu_count}). Creating multi-GPU config...")
        config_content = f"""compute_environment: LOCAL_MACHINE
deepspeed_config: {{}}
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: {gpu_count}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""

    config_dir = Path.home() / ".cache" / "huggingface" / "accelerate"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_file = config_dir / "default_config.yaml"
    with open(config_file, 'w') as f:
        f.write(config_content)

    print(f"Accelerate config written to: {config_file}")
    return config_file

def main():
    print("Accelerate Setup for MEDIQA-QA Fine-tuning")
    print("=" * 50)

    if not check_accelerate_installed():
        print("ERROR: accelerate is not installed!")
        print("Install with: pip install accelerate")
        sys.exit(1)

    gpu_count = get_gpu_count()
    print(f"Detected {gpu_count} GPU(s)")

    config_file = Path.home() / ".cache" / "huggingface" / "accelerate" / "default_config.yaml"

    if config_file.exists():
        print(f"Existing accelerate config found: {config_file}")
        response = input("Do you want to recreate it? (y/n): ").lower()
        if response != 'y':
            print("Using existing config.")
            return

    create_accelerate_config()

    print("\nAccelerate setup complete!")
    print("\nUsage examples:")
    print("=" * 30)

    if gpu_count == 0:
        print("# CPU training")
        print("python run_mediqa_pipeline.py --single_gpu --format simple")
    elif gpu_count == 1:
        print("# Single GPU training")
        print("python run_mediqa_pipeline.py --single_gpu --format simple")
        print("\n# Or with accelerate (same result for 1 GPU)")
        print("python run_mediqa_pipeline.py --format simple")
    else:
        print("# Multi-GPU training (recommended)")
        print("python run_mediqa_pipeline.py --format simple")
        print("\n# Manual accelerate launch")
        print("accelerate launch --main_process_port 29500 gpt2_finetune.py --model_name gpt2")
        print("\n# Single GPU only")
        print("python run_mediqa_pipeline.py --single_gpu --format simple")

if __name__ == "__main__":
    main()
