"""Tokenize a batch (harmful/harmless/pseudo) into a padded forward-ready tensor pack."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from main.data import Sample
from main.directions import format_prompt, post_inst_token_count


@dataclass
class HARCBatch:
    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor
    response_mask: Tensor
    t_inst: Tensor
    t_post: Tensor
    is_harmful: Tensor
    is_retain: Tensor
    categories: list[str]


def _tokenize_one(tokenizer, prompt_text: str, response_text: str,
                  max_prompt_len: int, max_response_len: int):
    prompt_str = format_prompt(tokenizer, prompt_text)
    saved_side = tokenizer.truncation_side
    try:
        tokenizer.truncation_side = "left"
        prompt_ids = tokenizer(
            prompt_str, add_special_tokens=False, truncation=True, max_length=max_prompt_len
        ).input_ids
    finally:
        tokenizer.truncation_side = saved_side
    response_ids = tokenizer(
        response_text, add_special_tokens=False, truncation=True, max_length=max_response_len
    ).input_ids
    eos_id = tokenizer.eos_token_id
    response_ids = response_ids + [eos_id]
    return prompt_ids, response_ids


def collate(
    batch: list[Sample],
    tokenizer,
    max_prompt_len: int = 256,
    max_response_len: int = 256,
) -> HARCBatch:
    P = post_inst_token_count(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    encoded: list[tuple[list[int], list[int], str]] = []
    for s in batch:
        p_ids, r_ids = _tokenize_one(tokenizer, s.prompt, s.response, max_prompt_len, max_response_len)
        encoded.append((p_ids, r_ids, s.category))

    seq_lens = [len(p) + len(r) for (p, r, _) in encoded]
    T = max(seq_lens)

    B = len(encoded)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, T), dtype=torch.long)
    labels = torch.full((B, T), -100, dtype=torch.long)
    response_mask = torch.zeros((B, T), dtype=torch.long)
    t_inst = torch.zeros(B, dtype=torch.long)
    t_post = torch.zeros(B, dtype=torch.long)
    is_harmful = torch.zeros(B, dtype=torch.bool)
    is_retain = torch.zeros(B, dtype=torch.bool)
    categories = []

    for i, (p_ids, r_ids, cat) in enumerate(encoded):
        plen = len(p_ids)
        rlen = len(r_ids)
        seq = p_ids + r_ids
        L_seq = plen + rlen
        input_ids[i, :L_seq] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, :L_seq] = 1
        response_mask[i, plen:plen + rlen] = 1
        if cat == "harmful":
            labels[i, plen:plen + rlen] = torch.tensor(r_ids, dtype=torch.long)
        # t_inst is the last user-content token; t_post the last template token after it.
        t_post[i] = plen - 1
        t_inst[i] = plen - 1 - P
        if t_inst[i] < 0:
            raise ValueError(f"prompt too short for post-inst offsets: plen={plen}, P={P}")
        is_harmful[i] = (cat == "harmful")
        is_retain[i] = cat in ("harmless", "pseudo")
        categories.append(cat)

    return HARCBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        response_mask=response_mask,
        t_inst=t_inst,
        t_post=t_post,
        is_harmful=is_harmful,
        is_retain=is_retain,
        categories=categories,
    )
