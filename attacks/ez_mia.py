"""EZ-MIA attack (our implementation)."""
import numpy as np
import torch
import torch.nn.functional as F

from attacks import AbstractAttack
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def ez_score(target_logits, ref_logits, target_ids, mask, ignore_bos=True, min_tokens=2):
    pred_logits_tgt = target_logits[:-1]
    pred_logits_ref = ref_logits[:-1]
    targets = target_ids[1:]
    m = mask[1:].float()

    if ignore_bos and m.shape[0] > 0:
        m[0] = 0.0

    tgt_log_probs = F.log_softmax(pred_logits_tgt, dim=-1)
    ref_log_probs = F.log_softmax(pred_logits_ref, dim=-1)

    tgt_correct_lp = tgt_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    ref_correct_lp = ref_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    top1_tgt = pred_logits_tgt.argmax(dim=-1)

    valid_mask = m > 0.5
    error_mask = (top1_tgt != targets) & valid_mask

    if error_mask.sum().item() < min_tokens:
        return 0.0

    deltas = (tgt_correct_lp - ref_correct_lp)[error_mask]

    positive_sum = deltas[deltas > 0].sum().item()
    negative_sum = deltas[deltas < 0].abs().sum().item()

    if negative_sum == 0:
        return 0.0

    value = positive_sum / (negative_sum + 1e-12)

    if not (value == value) or value < 0:
        return 0.0

    return value


class EZMIAAttack(AbstractAttack):

    def __init__(self, name, model, tokenizer, config):
        super().__init__(name, model, tokenizer, config)
        self.reference_model, self.reference_tokenizer = self._load_reference(config['device'])
        self.reference_device = config['device']
        self.ignore_bos = config.get('ignore_bos', True)
        self.min_tokens = config.get('min_tokens', 2)

    def _load_reference(self, device):
        reference_model = AutoModelForCausalLM.from_pretrained(
            self.config['reference_model_path'],
            torch_dtype=torch.float16,
        ).to(device)
        reference_tokenizer = AutoTokenizer.from_pretrained(self.config['reference_tokenizer_path'])
        reference_tokenizer.pad_token = reference_tokenizer.eos_token
        return reference_model, reference_tokenizer

    def run(self, dataset: Dataset) -> Dataset:
        dataset = dataset.map(
            lambda x: self.score(x),
            batched=True,
            batch_size=self.config['batch_size'],
            new_fingerprint=f"{self.signature(dataset)}_v1",
        )
        return dataset

    def score(self, batch):
        texts = batch['text']

        tgt_tokenized = self.tokenizer(
            texts, padding="longest", return_tensors="pt", truncation=True,
        )
        tgt_ids = tgt_tokenized['input_ids'].to(self.device)
        tgt_mask = tgt_tokenized['attention_mask'].to(self.device)

        ref_tokenized = self.reference_tokenizer(
            texts, padding="longest", return_tensors="pt", truncation=True,
        )
        ref_ids = ref_tokenized['input_ids'].to(self.reference_device)
        ref_mask = ref_tokenized['attention_mask'].to(self.reference_device)

        with torch.no_grad():
            tgt_logits = self.model(tgt_ids, attention_mask=tgt_mask).logits
            ref_logits = self.reference_model(ref_ids, attention_mask=ref_mask).logits

        tgt_logits = tgt_logits.cpu().float()
        ref_logits = ref_logits.cpu().float()
        tgt_ids = tgt_ids.cpu()
        tgt_mask = tgt_mask.cpu()

        scores = []
        for i in range(len(texts)):
            tgt_len = tgt_mask[i].sum().item()
            ref_len = ref_mask[i].sum().item()

            s = ez_score(
                tgt_logits[i, :tgt_len],
                ref_logits[i, :ref_len],
                tgt_ids[i, :tgt_len],
                tgt_mask[i, :tgt_len],
                ignore_bos=self.ignore_bos,
                min_tokens=self.min_tokens,
            )
            scores.append(s)

        return {self.name: scores}
