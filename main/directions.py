"""Prompt-side direction extraction: per layer, v_harm = norm(mean harm-minus-safe residual at t_inst), v_ref = same at t_post (assistant header)."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from main.data import Sample

LLAMA31_POST_INST = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


@dataclass
class Directions:
    v_ref: Tensor
    v_harm: Tensor
    norm_pre_ref: Tensor
    norm_pre_harm: Tensor


def post_inst_token_count(tokenizer) -> int:
    """Number of template tokens after the last user-content token, computed dynamically for any chat template."""
    dummy = "USERPLACEHOLDER1234XYZ"
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": dummy}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_ids = tokenizer(formatted, add_special_tokens=False).input_ids
    dummy_ids = tokenizer(dummy, add_special_tokens=False).input_ids
    for start in range(len(full_ids) - len(dummy_ids) + 1):
        if full_ids[start: start + len(dummy_ids)] == dummy_ids:
            return len(full_ids) - (start + len(dummy_ids))
    return len(tokenizer(LLAMA31_POST_INST, add_special_tokens=False).input_ids)


def format_prompt(tokenizer, prompt: str) -> str:
    msgs = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@contextlib.contextmanager
def _hooks(handles_spec: list[tuple[torch.nn.Module, Callable]]):
    handles = [m.register_forward_pre_hook(h) for m, h in handles_spec]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _capture_pre_hook(layer_idx: int, store: list[list[Tensor]],
                      t_inst: int, t_post: int):
    """Capture the residual entering this layer at the t_inst and t_post positions."""
    def hook(module, inputs):
        x = inputs[0]
        h_inst = x[:, t_inst, :].detach().to(torch.float32).cpu()
        h_post = x[:, t_post, :].detach().to(torch.float32).cpu()
        store[layer_idx].append(torch.stack([h_inst, h_post], dim=1))
    return hook


def _capture_post_hook_last(layer_idx: int, store: list[list[Tensor]],
                            t_inst: int, t_post: int):
    """Capture the residual leaving the final block (post-final-layer slot)."""
    def hook(module, inputs, output):
        x = output[0] if isinstance(output, tuple) else output
        h_inst = x[:, t_inst, :].detach().to(torch.float32).cpu()
        h_post = x[:, t_post, :].detach().to(torch.float32).cpu()
        store[layer_idx].append(torch.stack([h_inst, h_post], dim=1))
    return hook


@torch.no_grad()
def collect_activations(
    model,
    tokenizer,
    samples: list[Sample],
    batch_size: int = 8,
    max_length: int = 256,
) -> Tensor:
    """Forward passes collecting (N, L+1, 2, H) residuals at t_inst & t_post; slot L is the residual leaving the last block."""
    device = next(model.parameters()).device
    # PEFT proxies peft_config onto submodules, so check the class name, not hasattr.
    is_peft = type(model).__name__.startswith("Peft")
    blocks = (model.base_model.model.model.layers if is_peft
              else model.model.layers)
    n_layers = len(blocks)

    P = post_inst_token_count(tokenizer)
    t_inst = -P - 1
    t_post = -1

    store: list[list[Tensor]] = [[] for _ in range(n_layers + 1)]

    pre_specs = [
        (blocks[L], _capture_pre_hook(L, store, t_inst, t_post))
        for L in range(n_layers)
    ]
    last_handle = blocks[n_layers - 1].register_forward_hook(
        _capture_post_hook_last(n_layers, store, t_inst, t_post)
    )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        with _hooks(pre_specs):
            for i in range(0, len(samples), batch_size):
                texts = [format_prompt(tokenizer, s.prompt) for s in samples[i : i + batch_size]]
                enc = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=False,
                )
                model(
                    input_ids=enc.input_ids.to(device),
                    attention_mask=enc.attention_mask.to(device),
                    use_cache=False,
                )
    finally:
        last_handle.remove()

    per_layer = [torch.cat(chunks, dim=0) for chunks in store]
    activations = torch.stack(per_layer, dim=0)
    activations = activations.transpose(0, 1)
    return activations


def compute_directions(
    act_harm: Tensor,
    act_safe: Tensor,
) -> Directions:
    """Difference-of-means directions; activation index 0 (t_inst) -> v_harm, index 1 (t_post) -> v_ref."""
    mean_h = act_harm.mean(dim=0)
    mean_s = act_safe.mean(dim=0)
    diff = mean_h - mean_s

    raw_harm = diff[:, 0, :]
    raw_ref = diff[:, 1, :]

    norm_harm = raw_harm.norm(dim=-1)
    norm_ref = raw_ref.norm(dim=-1)
    eps = 1e-8
    v_harm = raw_harm / (norm_harm.unsqueeze(-1) + eps)
    v_ref = raw_ref / (norm_ref.unsqueeze(-1) + eps)
    return Directions(v_ref=v_ref, v_harm=v_harm,
                      norm_pre_ref=norm_ref, norm_pre_harm=norm_harm)


def extract_directions(
    model,
    tokenizer,
    extract_harmful: list[Sample],
    extract_harmless: list[Sample],
    batch_size: int = 8,
    max_length: int = 256,
    return_activations: bool = False,
):
    act_h = collect_activations(model, tokenizer, extract_harmful, batch_size, max_length)
    act_s = collect_activations(model, tokenizer, extract_harmless, batch_size, max_length)
    dirs = compute_directions(act_h, act_s)
    if return_activations:
        return dirs, act_h, act_s
    return dirs


def bootstrap_cos_ref_harm(act_harm: Tensor, act_safe: Tensor,
                           K: int = 100, seed: int = 0) -> Tensor:
    """Bootstrap-resample examples K times and recompute cos(v_ref, v_harm) per layer; returns (L+1, K)."""
    N_h, N_s = act_harm.shape[0], act_safe.shape[0]
    g = torch.Generator().manual_seed(seed)
    out = []
    for k in range(K):
        idx_h = torch.randint(0, N_h, (N_h,), generator=g)
        idx_s = torch.randint(0, N_s, (N_s,), generator=g)
        sub_h = act_harm[idx_h].mean(dim=0)
        sub_s = act_safe[idx_s].mean(dim=0)
        diff = sub_h - sub_s
        raw_harm = diff[:, 0, :]
        raw_ref = diff[:, 1, :]
        v_harm_b = raw_harm / raw_harm.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        v_ref_b = raw_ref / raw_ref.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        cos = (v_harm_b * v_ref_b).sum(dim=-1)
        out.append(cos)
    return torch.stack(out, dim=-1)


RESPONSE_NUM_TOKENS = 32


@dataclass
class ResponseDirections:
    v_ref_resp: Tensor
    v_harm_resp: Tensor
    norm_pre_ref: Tensor
    norm_pre_harm: Tensor


def _capture_response_pre_hook(layer: int, store: list[list[Tensor]],
                               response_starts: Tensor, response_lens: Tensor,
                               n_tokens: int):
    """Capture the mean residual over the first n_tokens response tokens per example."""
    def hook(module, inputs):
        x = inputs[0]
        B, T, H = x.shape
        out = []
        for i in range(B):
            s = int(response_starts[i].item())
            e = min(s + n_tokens, T, s + int(response_lens[i].item()))
            if e > s:
                v = x[i, s:e, :].mean(dim=0)
            else:
                v = torch.zeros(H, dtype=x.dtype, device=x.device)
            out.append(v.detach().to(torch.float32).cpu())
        store[layer].append(torch.stack(out, dim=0))
    return hook


@torch.no_grad()
def collect_response_activations(
    model, tokenizer,
    pairs: list[tuple[str, str]],
    batch_size: int = 4,
    max_prompt_len: int = 512,
    max_resp_len: int = 256,
    n_response_tokens: int = RESPONSE_NUM_TOKENS,
) -> Tensor:
    """Returns (N, L+1, H) mean-pooled residual over the first n_response_tokens of each (prompt, response) pair."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_peft = type(model).__name__.startswith("Peft")
    blocks = (model.base_model.model.model.layers if is_peft
              else model.model.layers)
    n_layers = len(blocks)
    store: list[list[Tensor]] = [[] for _ in range(n_layers + 1)]
    device = next(model.parameters()).device
    saved_trunc = tokenizer.truncation_side

    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        tokenizer.truncation_side = "left"
        prompt_texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True
            ) for (p, _) in chunk
        ]
        prompt_enc = [tokenizer(t, add_special_tokens=False, truncation=True,
                                max_length=max_prompt_len).input_ids
                      for t in prompt_texts]
        tokenizer.truncation_side = "right"
        resp_enc = [tokenizer(r, add_special_tokens=False, truncation=True,
                              max_length=max_resp_len).input_ids
                    for (_, r) in chunk]
        tokenizer.truncation_side = "left"

        seqs = [p + r for p, r in zip(prompt_enc, resp_enc)]
        T = max(len(s) for s in seqs)
        pad = tokenizer.pad_token_id
        input_ids = torch.full((len(seqs), T), pad, dtype=torch.long)
        attn = torch.zeros((len(seqs), T), dtype=torch.long)
        resp_starts = torch.zeros(len(seqs), dtype=torch.long)
        resp_lens = torch.zeros(len(seqs), dtype=torch.long)
        for j, s in enumerate(seqs):
            input_ids[j, :len(s)] = torch.tensor(s, dtype=torch.long)
            attn[j, :len(s)] = 1
            resp_starts[j] = len(prompt_enc[j])
            resp_lens[j] = len(resp_enc[j])

        specs = [
            (blocks[L], _capture_response_pre_hook(
                L, store, resp_starts, resp_lens, n_response_tokens))
            for L in range(n_layers)
        ]
        with _hooks(specs):
            model(input_ids=input_ids.to(device),
                  attention_mask=attn.to(device), use_cache=False)

    tokenizer.truncation_side = saved_trunc
    per_layer = [torch.cat(chunks, dim=0) for chunks in store[:n_layers]]
    if per_layer:
        per_layer.append(torch.zeros_like(per_layer[0]))
    return torch.stack(per_layer, dim=0).transpose(0, 1)


def extract_response_directions(
    model, tokenizer,
    harm_pairs: list[tuple[str, str]],
    refuse_pairs: list[tuple[str, str]],
    helpful_pairs: list[tuple[str, str]],
    batch_size: int = 4,
    max_prompt_len: int = 512,
    max_resp_len: int = 256,
) -> ResponseDirections:
    """Response-side directions: v_harm_resp = norm(mean harm - mean helpful), v_ref_resp = norm(mean refuse - mean helpful)."""
    act_h = collect_response_activations(model, tokenizer, harm_pairs,
                                         batch_size, max_prompt_len, max_resp_len)
    act_r = collect_response_activations(model, tokenizer, refuse_pairs,
                                         batch_size, max_prompt_len, max_resp_len)
    act_p = collect_response_activations(model, tokenizer, helpful_pairs,
                                         batch_size, max_prompt_len, max_resp_len)

    diff_harm = act_h.mean(dim=0) - act_p.mean(dim=0)
    diff_ref = act_r.mean(dim=0) - act_p.mean(dim=0)
    norm_harm = diff_harm.norm(dim=-1)
    norm_ref = diff_ref.norm(dim=-1)
    eps = 1e-8
    v_harm = diff_harm / (norm_harm.unsqueeze(-1) + eps)
    v_ref = diff_ref / (norm_ref.unsqueeze(-1) + eps)
    return ResponseDirections(
        v_ref_resp=v_ref, v_harm_resp=v_harm,
        norm_pre_ref=norm_ref, norm_pre_harm=norm_harm,
    )


@torch.no_grad()
def project_validation(
    model,
    tokenizer,
    validate_harmful: list[Sample],
    validate_harmless: list[Sample],
    v_ref: Tensor,
    v_harm: Tensor,
    layer_indices: list[int],
    batch_size: int = 8,
    max_length: int = 256,
) -> dict:
    """Mean cos(h_post, v_ref) and cos(h_inst, v_harm) on held-out data — an independent check that coupling is engaging."""
    act_h = collect_activations(model, tokenizer, validate_harmful, batch_size, max_length)
    act_s = collect_activations(model, tokenizer, validate_harmless, batch_size, max_length)

    out: dict[str, dict[int, float]] = {
        "proj_ref_harmful": {}, "proj_harm_harmful": {},
        "proj_ref_harmless": {}, "proj_harm_harmless": {},
    }
    for L in layer_indices:
        h_inst_h = act_h[:, L, 0, :]
        h_post_h = act_h[:, L, 1, :]
        h_inst_s = act_s[:, L, 0, :]
        h_post_s = act_s[:, L, 1, :]

        v_r = v_ref[L].to(h_inst_h.dtype)
        v_h = v_harm[L].to(h_inst_h.dtype)

        def _cosmean(x: Tensor, v: Tensor) -> float:
            n = x.norm(dim=-1).clamp_min(1e-8)
            return float(((x @ v) / n).mean().item())

        out["proj_ref_harmful"][L] = _cosmean(h_post_h, v_r)
        out["proj_harm_harmful"][L] = _cosmean(h_inst_h, v_h)
        out["proj_ref_harmless"][L] = _cosmean(h_post_s, v_r)
        out["proj_harm_harmless"][L] = _cosmean(h_inst_s, v_h)
    return out
