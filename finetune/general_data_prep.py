"""
General Dataset Preparation for Fine-tuning
"""

from datasets import load_dataset, Dataset
import json
import os
from typing import List, Dict, Tuple, Optional
import argparse
import pandas as pd
from pathlib import Path

class GeneralDataProcessor:
    def __init__(self, max_answer_length: int = 1024):
        self.max_answer_length = max_answer_length
        self.supported_datasets = {
            'imdb': self.process_imdb,
            'wikipedia': self.process_wikipedia,
        }
    
    def load_dataset_by_name(self, dataset_name: str):
        print(f"Loading {dataset_name} dataset...")

        if dataset_name == 'imdb':
            return load_dataset("imdb", trust_remote_code=True)
        elif dataset_name == 'wikipedia':
            return load_dataset("wikipedia", "20220301.en", streaming=False)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Available: {list(self.supported_datasets.keys())}")
    
    def process_wikipedia(self, dataset, use_all_splits: bool = True) -> List[Dict]:
        qa_pairs = []
        
        print("Processing Wikipedia...")
        split_data = dataset['train'] if 'train' in dataset else dataset

        max_examples = 50000
        actual_examples = min(len(split_data), max_examples)
        print(f"Processing {actual_examples} examples from {len(split_data)} total")
        
        for idx in range(actual_examples):
            example = split_data[idx]
            
            if idx < 3:
                print(f"  Example {idx} fields: {list(example.keys())}")
            
            title = example.get('title', '').strip()
            text = example.get('text', '').strip()
            
            if title and text and len(text) > 100:
                paragraphs = text.split('\n\n')
                if len(paragraphs) >= 2:
                    question = f"Continue this Wikipedia article about '{title}':\n\n{paragraphs[0]}"
                    answer = paragraphs[1] if len(paragraphs[1]) > 50 else text[len(paragraphs[0]):len(paragraphs[0])+500]
                else:
                    question = f"Summarize this Wikipedia article about '{title}':"
                    answer = paragraphs[0] if paragraphs and len(paragraphs[0]) > 50 else text[:500]
                
                if len(answer) > self.max_answer_length:
                    answer = answer[:self.max_answer_length] + "..."
                
                qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'metadata': {
                        'dataset': 'wikipedia',
                        'split': 'train',
                        'example_id': idx,
                        'title': title,
                        'article_length': len(text)
                    }
                })
            
            if idx > 0 and idx % 1000 == 0:
                print(f"Processed {idx}/{actual_examples} examples...")
        
        return qa_pairs
    

    
    def process_imdb(self, dataset, use_all_splits: bool = True) -> List[Dict]:
        qa_pairs = []
        splits_to_process = ['train', 'test', 'unsupervised'] if use_all_splits else ['train']
        
        for split_name in splits_to_process:
            if split_name not in dataset:
                continue
                
            print(f"Processing IMDB {split_name}...")
            split_data = dataset[split_name]
            
            max_examples = 15000 if split_name in ['train', 'unsupervised'] else len(split_data)
            actual_examples = min(len(split_data), max_examples)
            print(f"  Processing {actual_examples} examples from {len(split_data)} total")
            
            for idx in range(actual_examples):
                example = split_data[idx]
                
                if idx < 3:
                    print(f"  Example {idx} fields: {list(example.keys())}")
                
                text = example.get('text', '').strip()
                label = example.get('label', -1)
                
                if text:
                    review_preview = text[:300] + "..." if len(text) > 300 else text
                    question = f"What is the sentiment of this movie review? '{review_preview}'"
                    
                    if label == 1:
                        answer = "This review has a positive sentiment."
                    elif label == 0:
                        answer = "This review has a negative sentiment."
                    else:
                        answer = "Unable to determine sentiment (unlabeled data)."
                    
                    qa_pairs.append({
                        'question': question,
                        'answer': answer,
                        'metadata': {
                            'dataset': 'imdb',
                            'split': split_name,
                            'example_id': idx,
                            'label': label,
                            'review_length': len(text)
                        }
                    })
                
                if idx > 0 and idx % 2000 == 0:
                    print(f"Processed {idx}/{actual_examples} examples...")
        
        return qa_pairs
    

    
    def process_dataset(self, dataset_name: str, use_all_splits: bool = True) -> List[Dict]:
        if dataset_name not in self.supported_datasets:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        dataset = self.load_dataset_by_name(dataset_name)
        return self.supported_datasets[dataset_name](dataset, use_all_splits)

def format_simple(question: str, answer: str) -> str:
    return f"Question: {question.strip()}\nAnswer: {answer.strip()}"

def format_instruction(question: str, answer: str) -> str:
    return f"""Below is a question that requires analysis and reasoning. Provide a helpful and accurate response.

### Question:
{question.strip()}

### Answer:
{answer.strip()}"""

def format_chat(question: str, answer: str) -> str:
    return f"Human: {question.strip()}\n\nAssistant: {answer.strip()}"

def format_conversation(question: str, answer: str) -> str:
    return f"<|im_start|>user\n{question.strip()}<|im_end|>\n<|im_start|>assistant\n{answer.strip()}<|im_end|>"

def create_formatted_dataset(qa_pairs: List[Dict], 
                           format_type: str = "simple",
                           train_split_ratio: float = 0.5) -> Tuple[Dataset, Dataset]:
    
    format_functions = {
        "simple": format_simple,
        "instruction": format_instruction,
        "chat": format_chat,
        "conversation": format_conversation
    }
    
    if format_type not in format_functions:
        raise ValueError(f"format_type must be one of {list(format_functions.keys())}")
    
    format_fn = format_functions[format_type]

    formatted_texts = []
    for pair in qa_pairs:
        formatted_text = format_fn(pair['question'], pair['answer'])
        formatted_texts.append({
            'text': formatted_text,
            'question': pair['question'],
            'answer': pair['answer'],
            'metadata': pair['metadata']
        })

    split_idx = int(len(formatted_texts) * train_split_ratio)
    train_data = formatted_texts[:split_idx]
    val_data = formatted_texts[split_idx:]
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    return train_dataset, val_dataset

def save_dataset(dataset: Dataset, filepath: str):
    dataset.save_to_disk(filepath)
    print(f"Dataset saved to {filepath}")

def save_jsonl(dataset: Dataset, filepath: str):
    with open(filepath, 'w') as f:
        for item in dataset:
            f.write(json.dumps(item) + '\n')
    print(f"JSONL saved to {filepath}")

def main():
    parser = argparse.ArgumentParser(description="Prepare general datasets for fine-tuning")
    parser.add_argument("--dataset", 
                       choices=['imdb', 'wikipedia', 'all'],
                       default="imdb", 
                       help="Dataset to process ('all' for all available datasets)")
    parser.add_argument("--format", 
                       choices=["simple", "instruction", "chat", "conversation"], 
                       default="simple", 
                       help="Text formatting style")
    parser.add_argument("--output_dir", 
                       default="./general_processed", 
                       help="Output directory for processed datasets")
    parser.add_argument("--max_answer_length", 
                       type=int, default=1024, 
                       help="Maximum answer length in characters")
    parser.add_argument("--train_ratio", 
                       type=float, default=0.5,
                       help="Train/validation split ratio (default: 0.5 for 50-50 split)")
    parser.add_argument("--train_only", 
                       action="store_true", default=False,
                       help="Use only train split (not validation/test)")
    
    args = parser.parse_args()
    

    os.makedirs(args.output_dir, exist_ok=True)
    processor = GeneralDataProcessor(max_answer_length=args.max_answer_length)
    if args.dataset == 'all':
        datasets_to_process = ['imdb', 'wikipedia']
    else:
        datasets_to_process = [args.dataset]
    
    for dataset_name in datasets_to_process:
        try:
            print(f"\n{'='*50}")
            print(f"Processing {dataset_name.upper()}")
            print(f"{'='*50}")
            
            qa_pairs = processor.process_dataset(
                dataset_name,
                use_all_splits=not args.train_only
            )
            
            print(f"Extracted {len(qa_pairs)} pairs from {dataset_name}")
            
            if not qa_pairs:
                print(f"No data extracted from {dataset_name}, skipping...")
                continue

            train_dataset, val_dataset = create_formatted_dataset(
                qa_pairs, 
                format_type=args.format,
                train_split_ratio=args.train_ratio
            )
            

            dataset_output_dir = os.path.join(args.output_dir, dataset_name)
            os.makedirs(dataset_output_dir, exist_ok=True)
            
            train_path = os.path.join(dataset_output_dir, f"train_{args.format}")
            val_path = os.path.join(dataset_output_dir, f"val_{args.format}")
            
            save_dataset(train_dataset, train_path)
            save_dataset(val_dataset, val_path)
            
            sample_size = min(10, len(train_dataset))
            sample_train = train_dataset.select(range(sample_size))
            sample_val = val_dataset.select(range(min(5, len(val_dataset))))
            
            save_jsonl(sample_train, os.path.join(dataset_output_dir, f"sample_train_{args.format}.jsonl"))
            save_jsonl(sample_val, os.path.join(dataset_output_dir, f"sample_val_{args.format}.jsonl"))
            
            print(f"\nSample formatted text ({args.format} format) for {dataset_name}:")
            print("="*60)
            sample_text = train_dataset[0]['text']
            print(sample_text[:500] + "..." if len(sample_text) > 500 else sample_text)
            print("="*60)
            
            print(f"\n{dataset_name} dataset preparation complete!")
            print(f"Files saved in: {dataset_output_dir}")
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            print(f"Skipping {dataset_name}...")
            continue
    
    print(f"\n{'='*50}")
    print("ALL DATASETS PROCESSING COMPLETE!")
    print(f"{'='*50}")
    print(f"Output directory structure:")
    for dataset_name in datasets_to_process:
        dataset_dir = os.path.join(args.output_dir, dataset_name)
        if os.path.exists(dataset_dir):
            print(f"  {args.output_dir}/{dataset_name}/")
            print(f"    ├── train_{args.format}/")
            print(f"    ├── val_{args.format}/")
            print(f"    ├── sample_train_{args.format}.jsonl")
            print(f"    └── sample_val_{args.format}.jsonl")


if __name__ == "__main__":
    main()
