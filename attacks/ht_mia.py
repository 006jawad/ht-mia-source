"""
HT-MIA Attack
"""

import torch
import numpy as np
import torch.nn.functional as F
from attacks import AbstractAttack
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


class HTScoreAttack(AbstractAttack):
    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)

        self.base_model_name = config['base_model_name']
        self.base_device = config.get('base_device', self.device)

        print(f"Loading base model: {self.base_model_name}")

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.bfloat16
        ).to(self.base_device)

        self.base_tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)


        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token


        if self.tokenizer.vocab_size != self.base_tokenizer.vocab_size:
            print(
                f"WARNING: Different vocab sizes - "
                f"Target: {self.tokenizer.vocab_size}, Base: {self.base_tokenizer.vocab_size}"
            )

        self.base_model.eval()


        self.min_k = config.get('min_k', 10)
        self.max_k = config.get('max_k', 20)
        self.threshold = config.get('threshold', 0.6)
        self.max_length = config.get('max_length', 512)

        print("Attack initialized")
        print(f"Target model: {getattr(model, 'name_or_path', 'Unknown')}")
        print(f"Base model: {self.base_model_name}")
        print(f"Threshold: {self.threshold}")
        print(f"Device allocation: Target={self.device}, Base={self.base_device}")

    def run(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(
            lambda x: self.score_batch(x),
            batched=True,
            batch_size=self.config['batch_size'],
            new_fingerprint=f"{self.signature(dataset)}_v1",
        )
        return dataset

    def score_batch(self, batch):
        scores = []
        for text in batch['text']:
            score = self.calculate_score(text)
            scores.append(score)
        return {self.name: scores}

    def calculate_score(self, text):
        try:
            text = str(text)[:800]

            inputs = self.tokenizer.encode(
                text, return_tensors="pt", truncation=True, max_length=self.max_length
            )
            target_inputs = inputs.to(self.device)
            base_inputs = inputs.to(self.base_device)

            if inputs.shape[1] <= 2:
                print("data format not suited!!")
                return 0.5

            with torch.no_grad():
                target_outputs = self.model(target_inputs)
                target_logits = target_outputs.logits[0]

                base_outputs = self.base_model(base_inputs)
                base_logits = base_outputs.logits[0]

                target_probs = F.softmax(target_logits, dim=-1)
                base_probs = F.softmax(base_logits, dim=-1)

                tokens = inputs[0]
                token_probs_target = []
                token_probs_base = []

                for i in range(len(tokens) - 1):
                    next_token = tokens[i + 1].item()
                    target_prob = target_probs[i, next_token].item()
                    base_prob = base_probs[i, next_token].item()

                    token_probs_target.append(target_prob)
                    token_probs_base.append(base_prob)

                if len(token_probs_target) == 0:
                    print("We have an issue!!")
                    return 0.5

                seq_length = len(token_probs_target)
                adaptive_k = max(self.min_k, min(self.max_k, seq_length // 5))
                adaptive_k = min(adaptive_k, seq_length)

                if adaptive_k == 0:
                    return 0.5

                target_probs_array = np.array(token_probs_target)
                min_k_indices = np.argsort(target_probs_array)[:adaptive_k]

                improvements = []
                for idx in min_k_indices:
                    target_prob = token_probs_target[idx]
                    base_prob = token_probs_base[idx]
                    improvement = target_prob - base_prob
                    improvements.append(improvement)

                improvements = np.array(improvements)

                positive_improvements = np.sum(improvements > 0)
                ht_score = positive_improvements / len(improvements)

                return ht_score

        except Exception as e:
            print(f"Error in score calculation: {e}")
            return 0.5

    def __del__(self):
        try:
            if hasattr(self, 'base_model') and self.base_model is not None:
                del self.base_model
            if hasattr(self, 'base_tokenizer') and self.base_tokenizer is not None:
                del self.base_tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
