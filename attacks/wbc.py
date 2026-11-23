"""WBC attack. Adapted from https://github.com/Stry233/WBC (arXiv:2601.02751)."""
import logging

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from attacks import AbstractAttack
from attacks.utils import compute_per_token_losses


class WbcAttack(AbstractAttack):

    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)

        lengths = config.get('context_window_lengths', config.get('context_window_length', None))
        if lengths is None:
            self.window_sizes = [2, 3, 4, 6, 9, 13, 18, 25, 32, 40]
        elif isinstance(lengths, int):
            self.window_sizes = [lengths]
        else:
            self.window_sizes = list(lengths)

        self.max_length = config.get('max_length', 512)
        self.ref_device = config.get('reference_device', str(self.device))

        self.ref_model, self.ref_tokenizer = self._load_reference()
        logging.info(
            f"WbcAttack '{name}' ready | ref={config['reference_model_path']} "
            f"| windows={self.window_sizes}"
        )

    def _load_reference(self):
        ref_path = self.config['reference_model_path']
        tok_path = self.config.get('reference_tokenizer_path', ref_path)
        ref_model = AutoModelForCausalLM.from_pretrained(
            ref_path, torch_dtype=torch.float16
        ).to(self.ref_device)
        ref_tok = AutoTokenizer.from_pretrained(tok_path)
        ref_tok.pad_token = ref_tok.eos_token
        return ref_model, ref_tok

    @staticmethod
    def _window_score(target_losses: np.ndarray, ref_losses: np.ndarray, window_size: int) -> float:
        n = min(len(target_losses), len(ref_losses))
        if n == 0:
            return 0.0
        w = min(window_size, n)
        kernel = np.ones(w)
        t_sums = np.convolve(target_losses[:n], kernel, mode='valid')
        r_sums = np.convolve(ref_losses[:n], kernel, mode='valid')
        return float(np.mean(r_sums > t_sums))

    def _score_batch(self, texts):
        target_losses, target_mask = compute_per_token_losses(
            self.model, self.tokenizer, texts, self.device, self.max_length
        )
        ref_losses, ref_mask = compute_per_token_losses(
            self.ref_model, self.ref_tokenizer, texts, self.ref_device, self.max_length
        )

        scores = []
        for i in range(len(texts)):
            t_valid = target_losses[i][target_mask[i]]
            r_valid = ref_losses[i][ref_mask[i]]
            per_window = [self._window_score(t_valid, r_valid, w) for w in self.window_sizes]
            scores.append(float(np.mean(per_window)))
        return scores

    def run(self, dataset: Dataset) -> Dataset:
        batch_size = self.config.get('batch_size', 1)

        dataset = dataset.map(
            lambda batch: {self.name: self._score_batch(batch['text'])},
            batched=True,
            batch_size=batch_size,
            new_fingerprint=f"{self.signature(dataset)}_wbc_v1",
        )
        return dataset
