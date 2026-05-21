"""Direction extraction with the Zhao et al. (2025) raw-template + last-instruction-token method on our datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from main.data import build_splits, CB_TRAIN_PATH
from main.directions import (extract_response_directions,
                            bootstrap_cos_ref_harm)


def template_for(model_id: str) -> str:
    """Raw prompt template (no system prompt) per Zhao et al., for Llama or Qwen."""
    mid = model_id.lower()
    if "llama" in mid:
        return ("<|start_header_id|>user<|end_header_id|>\n{}"
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n")
    if "qwen" in mid:
        return "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant"
    raise ValueError(f"unsupported model_id: {model_id}")


def inst_token_for(model_id: str) -> str:
    """Trailing block whose last token is t_post and one-before is t_inst."""
    mid = model_id.lower()
    if "llama" in mid:
        return "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    if "qwen" in mid:
        return "<|im_end|>\n<|im_start|>assistant"
    raise ValueError(f"unsupported model_id: {model_id}")


def extract_directions_paper(
    model,
    tokenizer,
    harmful_prompts: list[str],
    harmless_prompts: list[str],
    template: str,
    inst_token: str,
    batch_size: int = 8,
    max_length: int = 256,
    return_activations: bool = False,
):
    """Drop-in for extract_directions using the paper method; act_h/act_s are (N, L+1, 2, H), slot 0=t_inst, 1=t_post."""
    from main.directions import Directions

    act_h = collect_paper_acts(model, tokenizer, harmful_prompts,
                                template, inst_token,
                                batch_size, max_length)
    act_s = collect_paper_acts(model, tokenizer, harmless_prompts,
                                template, inst_token,
                                batch_size, max_length)
    d = compute_directions_from_acts(act_h, act_s)
    dirs = Directions(
        v_ref=d["v_ref"], v_harm=d["v_harm"],
        norm_pre_ref=d["norm_pre_ref"], norm_pre_harm=d["norm_pre_harm"],
    )
    if return_activations:
        return dirs, act_h, act_s
    return dirs


@torch.no_grad()
def collect_paper_acts(model, tokenizer, prompts: list[str],
                       template: str, inst_token: str,
                       batch_size: int, max_length: int,
                       device: str = "cuda:0") -> Tensor:
    """Returns (N, L+1, 2, H) pre-block residuals, slot 0 = t_inst (-P-1), slot 1 = t_post (-1)."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    n_layers = model.config.num_hidden_layers
    H = model.config.hidden_size
    inst_ids = tokenizer(inst_token, add_special_tokens=False).input_ids
    P = len(inst_ids)

    assert template.endswith(inst_token), (
        f"template must end with inst_token; got template ending "
        f"'{template[-len(inst_token):]!r}', expected '{inst_token!r}'"
    )

    out = torch.zeros(len(prompts), n_layers + 1, 2, H, dtype=torch.float)
    # PEFT proxies peft_config onto submodules, so check the class name, not hasattr.
    is_peft = type(model).__name__.startswith("Peft")
    blocks = (model.base_model.model.model.layers if is_peft
              else model.model.layers)

    cache: dict = {}

    def make_pre_hook(L_idx: int):
        def fn(_module, inputs):
            x = inputs[0]
            cache[L_idx] = x.detach()
        return fn

    def make_post_hook(L_idx: int):
        def fn(_module, _inputs, output):
            x = output[0] if isinstance(output, tuple) else output
            cache[L_idx] = x.detach()
        return fn

    handles = []
    for i, blk in enumerate(blocks):
        handles.append(blk.register_forward_pre_hook(make_pre_hook(i)))
    handles.append(blocks[-1].register_forward_hook(make_post_hook(n_layers)))

    try:
        for start in range(0, len(prompts), batch_size):
            cache.clear()
            batch = prompts[start: start + batch_size]
            texts = [template.format(p) for p in batch]
            enc = tokenizer(texts, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt",
                            add_special_tokens=False).to(device)
            _ = model(**enc)

            for L_idx in range(n_layers + 1):
                x = cache[L_idx]
                t_post = x[:, -1, :]
                t_inst = x[:, -P - 1, :]
                out[start: start + len(batch), L_idx, 0, :] = t_inst.float().cpu()
                out[start: start + len(batch), L_idx, 1, :] = t_post.float().cpu()
    finally:
        for h in handles:
            h.remove()

    print(f"[acts] collected ({len(prompts)}, {n_layers + 1}, 2, {H})  P={P}")
    return out


def compute_directions_from_acts(act_h: Tensor, act_s: Tensor) -> dict:
    """Difference-of-means: slot 0 (t_inst) -> v_harm, slot 1 (t_post) -> v_ref, unit-normed."""
    mean_h = act_h.mean(dim=0)
    mean_s = act_s.mean(dim=0)
    diff = mean_h - mean_s
    v_harm_raw = diff[:, 0, :]
    v_ref_raw = diff[:, 1, :]
    norm_pre_harm = v_harm_raw.norm(dim=-1)
    norm_pre_ref = v_ref_raw.norm(dim=-1)
    eps = 1e-8
    v_harm = v_harm_raw / norm_pre_harm.unsqueeze(-1).clamp_min(eps)
    v_ref = v_ref_raw / norm_pre_ref.unsqueeze(-1).clamp_min(eps)
    return {
        "v_ref": v_ref, "v_harm": v_harm,
        "norm_pre_ref": norm_pre_ref, "norm_pre_harm": norm_pre_harm,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--with_response", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.model_id}")
    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map={"": 0},
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    template = template_for(args.model_id)
    inst_tok = inst_token_for(args.model_id)
    print(f"[template] {template!r}")
    print(f"[inst_token] {inst_tok!r}")

    splits = build_splits()
    extract = splits["extract"]
    harmful_prompts = [s.prompt for s in extract["harmful"]]
    harmless_prompts = [s.prompt for s in extract["harmless"]]
    print(f"[data] harmful={len(harmful_prompts)} harmless={len(harmless_prompts)}")

    print("[extract] harmful activations")
    act_h = collect_paper_acts(model, tok, harmful_prompts, template, inst_tok,
                                args.batch_size, args.max_length)
    print("[extract] harmless activations")
    act_s = collect_paper_acts(model, tok, harmless_prompts, template, inst_tok,
                                args.batch_size, args.max_length)

    dirs = compute_directions_from_acts(act_h, act_s)

    print("[bootstrap] cos(v_harm, v_ref)")
    cos_boot = bootstrap_cos_ref_harm(act_h, act_s, K=200, seed=12345)
    cos_mean = cos_boot.mean(dim=-1)
    cos_lo = cos_boot.quantile(0.025, dim=-1)
    cos_hi = cos_boot.quantile(0.975, dim=-1)

    out = args.out_dir / "directions_base.pt"
    torch.save({
        "v_ref": dirs["v_ref"].cpu(),
        "v_harm": dirs["v_harm"].cpu(),
        "norm_pre_ref": dirs["norm_pre_ref"].cpu(),
        "norm_pre_harm": dirs["norm_pre_harm"].cpu(),
        "cos_boot_mean": cos_mean.cpu(),
        "cos_boot_lo": cos_lo.cpu(),
        "cos_boot_hi": cos_hi.cpu(),
    }, out)
    print(f"[save] {out}")

    n = dirs["v_ref"].shape[0] - 1
    print(f"\n=== cos(v_harm, v_ref) per layer (n_layers={n}) ===")
    print(f"{'L':>3s} | {'cos':>8s}  {'95% CI':>22s}")
    for L in range(1, n + 1):
        m = cos_mean[L].item()
        lo = cos_lo[L].item()
        hi = cos_hi[L].item()
        print(f"{L:3d} | {m:+8.3f}  [{lo:+6.3f}, {hi:+6.3f}]")

    if args.with_response:
        print("\n[extract] response-side directions (shared-helpful baseline)")
        cb_full = json.loads(CB_TRAIN_PATH.read_text())
        import random as _r
        idx = list(range(len(cb_full)))
        _r.Random(0).shuffle(idx)
        ext_h_idx = idx[:300]
        harm_pairs = [(cb_full[i]["prompt"], cb_full[i]["output"]) for i in ext_h_idx]
        refuse_pairs = [(cb_full[i]["prompt"], cb_full[i]["llama3_output"]) for i in ext_h_idx]
        helpful_pairs = [(s.prompt, s.response) for s in extract["harmless"]]
        resp_dirs = extract_response_directions(
            model, tok, harm_pairs, refuse_pairs, helpful_pairs,
            batch_size=args.batch_size,
            max_prompt_len=args.max_length,
            max_resp_len=args.max_length,
        )
        out_resp = args.out_dir / "response_directions_base.pt"
        torch.save({
            "v_ref_resp": resp_dirs.v_ref_resp.cpu(),
            "v_harm_resp": resp_dirs.v_harm_resp.cpu(),
            "norm_pre_ref": resp_dirs.norm_pre_ref.cpu(),
            "norm_pre_harm": resp_dirs.norm_pre_harm.cpu(),
        }, out_resp)
        print(f"[save] {out_resp}")


if __name__ == "__main__":
    main()
