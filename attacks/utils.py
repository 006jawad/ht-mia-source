from typing import Optional

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel


def compute_nlloss(
    model: PreTrainedModel,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    ignore_prefix: Optional[int] = None,
):
    with torch.no_grad():
        labels = token_ids.clone()

        outputs = model(token_ids, attention_mask=attention_mask)

        shift_logits = outputs.logits[..., :-1, :].contiguous().view(-1, model.config.vocab_size)
        shift_attention_mask = attention_mask[..., :-1]
        shift_targets = labels[..., 1:]

        shift_targets[shift_attention_mask == 0] = -100

        loss = F.cross_entropy(shift_logits, shift_targets.contiguous().view(-1), reduction="none")
        loss = loss.view(token_ids.shape[0], -1)

        if ignore_prefix:
            loss = loss[:, ignore_prefix:]
            shift_attention_mask = shift_attention_mask[:, ignore_prefix:]

        loss = loss.sum(axis=1) / shift_attention_mask.sum(axis=1)

        return loss.detach().cpu().float().numpy()


def batch_nlloss(batch, model, tokenizer, device, key='nlloss'):
    tokenized = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
    token_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    losses = compute_nlloss(model, token_ids, attention_mask)
    return {key: losses}


def compute_per_token_losses(model, tokenizer, texts, device, max_length=512):
    """Compute per-token losses for a list of texts.

    Returns:
        loss_per_token: np.ndarray of shape (batch, seq_len-1)
        valid_mask:     bool np.ndarray, True where loss is valid (non-padded)
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenized = tokenizer(
        texts,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_attention = attention_mask[..., 1:].contiguous()

        shift_labels[shift_attention == 0] = -100

        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, model.config.vocab_size),
            shift_labels.view(-1),
            reduction='none',
            ignore_index=-100,
        ).view(input_ids.shape[0], -1)

        valid_mask = (shift_labels != -100)

    return loss_per_token.cpu().float().numpy(), valid_mask.cpu().numpy()
