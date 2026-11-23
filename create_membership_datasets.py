"""
Create membership datasets for running MIA
"""

from datasets import load_from_disk, Dataset
import json
import os
import argparse
from typing import List, Dict, Tuple
import pandas as pd

def get_available_datasets(base_dir: str) -> List[str]:
    datasets = []
    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                datasets.append(item)
    return sorted(datasets)

def load_processed_datasets(data_dir: str, dataset_name: str, format_type: str = "simple") -> Tuple[Dataset, Dataset]:
    dataset_path = os.path.join(data_dir, dataset_name)
    train_path = os.path.join(dataset_path, f"train_{format_type}")
    val_path = os.path.join(dataset_path, f"val_{format_type}")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training dataset not found at {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation dataset not found at {val_path}")
    
    print(f"Loading training data from {train_path}...")
    train_dataset = load_from_disk(train_path)
    
    print(f"Loading validation data from {val_path}...")
    val_dataset = load_from_disk(val_path)
    
    return train_dataset, val_dataset

def create_membership_dataset(train_dataset: Dataset, val_dataset: Dataset, 
                                 balance_samples: bool = True,
                                 max_samples_per_class: int = None) -> Dataset:
    """
    Create the formatted membership dataset directly
    
    Args:
        train_dataset: Training dataset (will be labeled as label=1)
        val_dataset: Validation dataset (will be labeled as label=0)
        balance_samples: Whether to balance the number of member/non-member samples
        max_samples_per_class: Maximum samples per class (None = use all)
    
    Returns:
        formatted dataset with 'label' field
    """

    member_samples = []
    train_size = len(train_dataset)
    
    if max_samples_per_class:
        train_size = min(train_size, max_samples_per_class)
    
    for i in range(train_size):
        sample = dict(train_dataset[i])
        sample['label'] = 1
        sample['sample_type'] = 'train'
        member_samples.append(sample)
    

    non_member_samples = []
    val_size = len(val_dataset)
    
    if balance_samples:
        val_size = min(val_size, len(member_samples))
    
    if max_samples_per_class:
        val_size = min(val_size, max_samples_per_class)
    
    for i in range(val_size):
        sample = dict(val_dataset[i])
        sample['label'] = 0
        sample['sample_type'] = 'val'
        non_member_samples.append(sample)

    all_samples = member_samples + non_member_samples
    
    membership_dataset = Dataset.from_list(all_samples)
    membership_dataset = membership_dataset.shuffle(seed=42)
    
    return membership_dataset, len(member_samples), len(non_member_samples)

def save_membership_formats(dataset: Dataset, output_base: str, dataset_name: str, format_type: str):
    base_filename = f"{dataset_name}_{format_type}_mia"

    dataset_path = os.path.join(output_base, base_filename)
    dataset.save_to_disk(dataset_path)
    print(f"Membership dataset saved to: {dataset_path}")

    jsonl_path = os.path.join(output_base, f"{base_filename}.jsonl")
    with open(jsonl_path, 'w') as f:
        for item in dataset:
            record = {
                'text': item['text'],
                'question': item.get('question', ''),
                'answer': item.get('answer', ''),
                'label': item['label'],
                'sample_type': item['sample_type']
            }
            f.write(json.dumps(record) + '\n')
    print(f"    JSONL saved to: {jsonl_path}")

def print_dataset_stats(dataset: Dataset, dataset_name: str):
    member_count = sum(1 for item in dataset if item['label'] == 1)
    non_member_count = len(dataset) - member_count
    
    print(f"Members (label=1): {member_count}")
    print(f"Non-members (label=0): {non_member_count}")
    print(f"Total samples: {len(dataset)}")
    print(f"Balance: {member_count/len(dataset)*100:.1f}% members")

def process_single_dataset(data_dir: str, dataset_name: str, output_dir: str, 
                          format_type: str, balance: bool, max_samples: int):
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*60}")
    
    try:
        train_dataset, val_dataset = load_processed_datasets(data_dir, dataset_name, format_type)
        print(f"Original train size: {len(train_dataset)}")
        print(f"Original val size: {len(val_dataset)}")
        
        membership_dataset, member_count, non_member_count = create_membership_dataset(
            train_dataset, val_dataset, balance, max_samples
        )

        save_membership_formats(membership_dataset, output_dir, dataset_name, format_type)

        print(f"Final dataset statistics:")
        print_dataset_stats(membership_dataset, dataset_name)
        
        return True, len(membership_dataset)
        
    except Exception as e:
        print(f"ERROR processing {dataset_name}: {str(e)}")
        return False, 0

def main():
    parser = argparse.ArgumentParser(description="Create formatted membership datasets")
    parser.add_argument("--data_dir", default="./medical_processed", 
                       help="Directory containing processed medical datasets")
    parser.add_argument("--datasets", nargs='+', default=None,
                       help="Specific datasets to process (default: all available)")
    parser.add_argument("--format", choices=["simple", "instruction", "chat"], 
                       default="simple", help="Dataset format type")
    parser.add_argument("--output_dir", default="./membership_data", 
                       help="Output directory for membership datasets")
    parser.add_argument("--balance", action="store_true", default=True,
                       help="Balance member/non-member samples")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples per class per dataset")
    parser.add_argument("--list_datasets", action="store_true",
                       help="List available datasets and exit")
    
    args = parser.parse_args()

    if args.list_datasets:
        available = get_available_datasets(args.data_dir)
        print("Available datasets:")
        for dataset in available:
            print(f"  - {dataset}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)

    if args.datasets:
        datasets_to_process = args.datasets
    else:
        datasets_to_process = get_available_datasets(args.data_dir)
    
    if not datasets_to_process:
        print(f"No datasets found in {args.data_dir}")
        return
    
    print(f"Will process {len(datasets_to_process)} datasets:")
    for dataset in datasets_to_process:
        print(f"  - {dataset}")
    
    successful = 0
    total_samples = 0
    
    for dataset_name in datasets_to_process:
        success, samples = process_single_dataset(
            args.data_dir, dataset_name, args.output_dir, 
            args.format, args.balance, args.max_samples
        )
        if success:
            successful += 1
            total_samples += samples
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully processed: {successful}/{len(datasets_to_process)} datasets")
    print(f"Total samples created: {total_samples}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nformatted datasets are ready for membership inference evaluation!")


if __name__ == "__main__":
    main()
