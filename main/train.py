"""HARC fine-tuning loop: per step, couple prompt/response residuals to v_ref/v_harm plus KL retention and refusal CE, with periodic EMA direction recompute; runs to max_steps."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import LoraConfig, get_peft_model
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

from main.collate import HARCBatch, collate
from main.data import Sample, build_splits, category_iterator, sample_batch
from main.directions import (Directions, ResponseDirections,
                            bootstrap_cos_ref_harm,
                            extract_directions, extract_response_directions,
                            project_validation)
from main.layers import select_layers
from main.losses import (ce_refusal_loss, coupling_loss, kl_retain_loss,
                        response_coupling_loss)


def _select(cfg, dirs, k, resp_dirs=None):
    if resp_dirs is None:
        raise ValueError(
            "variant-D layer selection needs response-side directions; "
            "enable response coupling so resp_dirs is available."
        )
    return select_layers(dirs, resp_dirs, k)


@dataclass
class Config:
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    out_dir: str = "runs/harc"
    seed: int = 0

    micro_batch_harmful: int = 6
    micro_batch_harmless: int = 4
    micro_batch_pseudo: int = 2
    grad_accum: int = 2

    lr: float = 1e-4
    warmup: int = 100
    max_steps: int = 4000

    lora_r: int = 32
    lora_alpha: int = 64
    lora_targets: tuple = ("q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj")
    lora_dropout: float = 0.0

    lambda_couple: float = 1.0
    lambda_couple_resp: float = 1.0
    lambda_kl: float = 10.0
    lambda_ce: float = 1.0
    coupling_margin: float = 0.3
    response_n_tokens: int = 32
    enable_response_coupling: bool = True

    K_recompute: int = 200
    beta_ema: float = 0.3
    K_layers_initial: int = 2
    K_layers_final: int = 4
    K_ramp_step: int = 1000
    extract_batch_size: int = 8
    extract_max_len: int = 256
    extraction_method: str = "default"


def set_seed(s: int):
    import random as _r
    import numpy as _np
    _r.seed(s)
    _np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def attach_lora(model, cfg: Config):
    lc = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=list(cfg.lora_targets),
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lc)


class HookedForward:
    """Forward pass that hooks the selected transformer blocks to capture residuals at t_inst/t_post (and a response window)."""

    def __init__(self, model, is_peft: bool):
        self.model = model
        if is_peft:
            self.blocks = model.base_model.model.model.layers
        else:
            self.blocks = model.model.layers

    def _hook(self, layer_idx: int, store: dict, t_inst_idx: Tensor, t_post_idx: Tensor,
              resp_start: Tensor | None, resp_len: Tensor | None, n_resp: int):
        def fn(module, inputs):
            x = inputs[0]
            B = x.size(0)
            arange = torch.arange(B, device=x.device)
            h_inst = x[arange, t_inst_idx.to(x.device), :]
            h_post = x[arange, t_post_idx.to(x.device), :]
            h_resp = None
            if resp_start is not None and resp_len is not None:
                T = x.size(1)
                rs = resp_start.to(x.device).unsqueeze(1)
                rl = resp_len.to(x.device).unsqueeze(1)
                window = torch.minimum(rl, torch.full_like(rl, n_resp))
                # Mean-pool the residual over the first n_resp response tokens.
                pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
                mask = (pos >= rs) & (pos < rs + window)
                mask_f = mask.to(x.dtype).unsqueeze(-1)
                denom = mask_f.sum(dim=1).clamp_min(1.0)
                h_resp = (x * mask_f).sum(dim=1) / denom
            store[layer_idx] = (h_inst, h_post, h_resp)
        return fn

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        capture_layers: list[int] | None,
        t_inst_idx: Tensor | None,
        t_post_idx: Tensor | None,
        with_grad: bool,
        resp_start: Tensor | None = None,
        resp_len: Tensor | None = None,
        n_resp: int = 32,
    ):
        store: dict[int, tuple[Tensor, Tensor, Tensor | None]] = {}
        handles = []
        if capture_layers:
            for L in capture_layers:
                h = self.blocks[L].register_forward_pre_hook(
                    self._hook(L, store, t_inst_idx, t_post_idx,
                               resp_start, resp_len, n_resp)
                )
                handles.append(h)
        try:
            ctx = torch.enable_grad() if with_grad else torch.no_grad()
            with ctx:
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=False,
                )
        finally:
            for h in handles:
                h.remove()
        return out, store


def ema_blend(v_base: Tensor, v_lora: Tensor, beta: float) -> Tensor:
    blended = (1 - beta) * v_base + beta * v_lora
    return F.normalize(blended, dim=-1)


def recompute_directions(
    M_lora,
    tokenizer,
    extract_h: list[Sample],
    extract_s: list[Sample],
    base_dirs: Directions,
    prev_dirs: Directions,
    beta: float,
    cfg: Config,
    return_activations: bool = False,
) -> tuple[Directions, dict] | tuple[Directions, dict, Tensor, Tensor]:
    M_lora.eval()
    inner_model = (M_lora.base_model.model
                   if hasattr(M_lora, "base_model") else M_lora)
    if cfg.extraction_method == "advbench":
        from main.extract_paper_method import (extract_directions_paper,
                                               template_for, inst_token_for)
        h_prompts = [s.prompt for s in extract_h]
        s_prompts = [s.prompt for s in extract_s]
        fresh, act_h, act_s = extract_directions_paper(
            inner_model, tokenizer, h_prompts, s_prompts,
            template=template_for(cfg.model_id),
            inst_token=inst_token_for(cfg.model_id),
            batch_size=cfg.extract_batch_size,
            max_length=cfg.extract_max_len,
            return_activations=True,
        )
    else:
        fresh, act_h, act_s = extract_directions(
            inner_model, tokenizer, extract_h, extract_s,
            batch_size=cfg.extract_batch_size,
            max_length=cfg.extract_max_len, return_activations=True,
        )
    M_lora.train()
    blended = Directions(
        v_ref=ema_blend(base_dirs.v_ref, fresh.v_ref, beta),
        v_harm=ema_blend(base_dirs.v_harm, fresh.v_harm, beta),
        norm_pre_ref=base_dirs.norm_pre_ref,
        norm_pre_harm=base_dirs.norm_pre_harm,
    )
    cos_ref_fresh = (prev_dirs.v_ref * fresh.v_ref).sum(-1)
    cos_harm_fresh = (prev_dirs.v_harm * fresh.v_harm).sum(-1)
    cos_ref_blended = (prev_dirs.v_ref * blended.v_ref).sum(-1)
    cos_harm_blended = (prev_dirs.v_harm * blended.v_harm).sum(-1)
    drift = {
        "cos_ref_fresh_min": float(cos_ref_fresh.min()),
        "cos_ref_fresh_mean": float(cos_ref_fresh.mean()),
        "cos_harm_fresh_min": float(cos_harm_fresh.min()),
        "cos_harm_fresh_mean": float(cos_harm_fresh.mean()),
        "cos_ref_blended_min": float(cos_ref_blended.min()),
        "cos_harm_blended_min": float(cos_harm_blended.min()),
    }
    if return_activations:
        return blended, drift, act_h, act_s
    return blended, drift


def step_loss(
    M_lora_h: HookedForward,
    M_base_h: HookedForward,
    batch: HARCBatch,
    selected: list[int],
    v_ref: Tensor,
    v_harm: Tensor,
    cfg: Config,
    device,
    v_ref_resp: Tensor | None = None,
    v_harm_resp: Tensor | None = None,
) -> dict[str, Tensor]:
    input_ids = batch.input_ids.to(device)
    attention_mask = batch.attention_mask.to(device)
    labels = batch.labels.to(device)
    response_mask = batch.response_mask.to(device)
    t_inst = batch.t_inst.to(device)
    t_post = batch.t_post.to(device)
    is_harmful = batch.is_harmful.to(device)
    is_retain = batch.is_retain.to(device)

    resp_len_per = response_mask.sum(dim=1).long()
    rm_long = response_mask.long()
    has_resp = resp_len_per > 0
    resp_start_per = rm_long.argmax(dim=1).long()
    resp_start_per = torch.where(has_resp, resp_start_per,
                                 torch.zeros_like(resp_start_per))

    out_lora, store = M_lora_h.forward(
        input_ids, attention_mask,
        capture_layers=selected,
        t_inst_idx=t_inst, t_post_idx=t_post,
        with_grad=True,
        resp_start=resp_start_per if v_ref_resp is not None else None,
        resp_len=resp_len_per if v_ref_resp is not None else None,
        n_resp=cfg.response_n_tokens,
    )
    lora_logits = out_lora.logits.to(device)

    retain_idx = is_retain.nonzero(as_tuple=True)[0]
    base_logits_retain = None
    if retain_idx.numel() > 0:
        out_base, _ = M_base_h.forward(
            input_ids[retain_idx], attention_mask[retain_idx],
            capture_layers=None, t_inst_idx=None, t_post_idx=None,
            with_grad=False,
        )
        base_logits_retain = out_base.logits.to(device)

    h_inst_dict = {L: store[L][0].to(device) for L in selected}
    h_post_dict = {L: store[L][1].to(device) for L in selected}
    couple = coupling_loss(
        h_inst=h_inst_dict, h_post=h_post_dict,
        v_ref=v_ref.to(device), v_harm=v_harm.to(device),
        is_harmful=is_harmful, margin=cfg.coupling_margin,
    )
    L_couple = couple["total"]

    L_couple_resp = torch.tensor(0.0, device=device)
    couple_resp: dict[str, Tensor] = {}
    if v_ref_resp is not None and v_harm_resp is not None and has_resp.any():
        h_resp_dict = {L: store[L][2].to(device)
                       for L in selected if store[L][2] is not None}
        if h_resp_dict:
            couple_resp = response_coupling_loss(
                h_resp=h_resp_dict,
                v_ref_resp=v_ref_resp.to(device),
                v_harm_resp=v_harm_resp.to(device),
                is_harmful=is_harmful,
                margin=cfg.coupling_margin,
            )
            L_couple_resp = couple_resp["total"]

    if retain_idx.numel() > 0:
        L_kl = kl_retain_loss(
            lora_logits=lora_logits[retain_idx],
            base_logits=base_logits_retain,
            response_mask=response_mask[retain_idx],
        )
    else:
        L_kl = torch.tensor(0.0, device=device)

    harmful_idx = is_harmful.nonzero(as_tuple=True)[0]
    if harmful_idx.numel() > 0:
        L_ce = ce_refusal_loss(
            lora_logits=lora_logits[harmful_idx],
            labels=labels[harmful_idx],
        )
    else:
        L_ce = torch.tensor(0.0, device=device)

    L_total = (cfg.lambda_couple * L_couple
               + cfg.lambda_couple_resp * L_couple_resp
               + cfg.lambda_kl * L_kl
               + cfg.lambda_ce * L_ce)
    out = {
        "total": L_total,
        "couple": L_couple.detach(),
        "couple_harmful": couple["harmful"],
        "couple_retain": couple["retain"],
        "couple_resp": L_couple_resp.detach(),
        "couple_resp_harmful": couple_resp.get("harmful", torch.tensor(0.0)),
        "couple_resp_retain": couple_resp.get("retain", torch.tensor(0.0)),
        "kl": L_kl.detach(),
        "ce": L_ce.detach(),
    }
    for k in ("proj_ref_harmful", "proj_harm_harmful",
              "proj_ref_retain", "proj_harm_retain"):
        if k in couple:
            out[k] = couple[k].detach()
    for k in ("proj_ref_resp_harmful", "proj_harm_resp_harmful",
              "proj_ref_resp_retain", "proj_harm_resp_retain"):
        if k in couple_resp:
            out[k] = couple_resp[k].detach()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--max_steps", type=int, default=None, help="override for smoke runs")
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.config:
        with open(args.config) as f:
            user_cfg = yaml.safe_load(f)
        for k, v in user_cfg.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))

    set_seed(cfg.seed)
    device = torch.device("cuda:0")

    print(f"[load] {cfg.model_id}")
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    M_base = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    M_base.eval()
    for p in M_base.parameters():
        p.requires_grad = False

    print("[load] LoRA model")
    M_lora_inner = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    M_lora = attach_lora(M_lora_inner, cfg)
    M_lora.print_trainable_parameters()

    M_lora_h = HookedForward(M_lora, is_peft=True)
    M_base_h = HookedForward(M_base, is_peft=False)

    print("[data] build splits")
    splits = build_splits()
    train = splits["train"]
    extract = splits["extract"]
    validate = splits["validate"]
    if cfg.extraction_method == "advbench":
        from main.data import DATA_DIR as _DATA_DIR
        ADV = _DATA_DIR / "advbench/advbench.json"
        ALP = _DATA_DIR / "advbench/alpaca_data_instruction.json"

        def _load_jsonl_field(p, field, n):
            out = []
            with p.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    if field in d:
                        out.append(d[field])
                    if len(out) >= n:
                        break
            return out

        adv_h = _load_jsonl_field(ADV, "bad_q", 300)
        alp_s = _load_jsonl_field(ALP, "instruction", 300)
        extract = {
            "harmful": [Sample("harmful", p, "") for p in adv_h],
            "harmless": [Sample("harmless", p, "") for p in alp_s],
        }
        print(f"[data] paper-method extract split: "
              f"AdvBench harmful={len(extract['harmful'])}, "
              f"Alpaca harmless={len(extract['harmless'])}")

    iters = {
        "harmful": category_iterator(train["harmful"], seed=cfg.seed + 1),
        "harmless": category_iterator(train["harmless"], seed=cfg.seed + 2),
        "pseudo": category_iterator(train["pseudo"], seed=cfg.seed + 3),
    }
    counts = {
        "harmful": cfg.micro_batch_harmful,
        "harmless": cfg.micro_batch_harmless,
        "pseudo": cfg.micro_batch_pseudo,
    }

    print(f"[directions] initial extraction on M_base "
          f"(method={cfg.extraction_method}, with bootstrap)")
    if cfg.extraction_method == "advbench":
        from main.extract_paper_method import (extract_directions_paper,
                                               template_for, inst_token_for)
        base_dirs, base_act_h, base_act_s = extract_directions_paper(
            M_base, tok,
            [s.prompt for s in extract["harmful"]],
            [s.prompt for s in extract["harmless"]],
            template=template_for(cfg.model_id),
            inst_token=inst_token_for(cfg.model_id),
            batch_size=cfg.extract_batch_size,
            max_length=cfg.extract_max_len,
            return_activations=True,
        )
    else:
        base_dirs, base_act_h, base_act_s = extract_directions(
            M_base, tok, extract["harmful"], extract["harmless"],
            batch_size=cfg.extract_batch_size, max_length=cfg.extract_max_len,
            return_activations=True,
        )
    base_cos_boot = bootstrap_cos_ref_harm(base_act_h, base_act_s, K=200, seed=12345)
    base_cos_mean = base_cos_boot.mean(dim=-1)
    base_cos_lo = base_cos_boot.quantile(0.025, dim=-1)
    base_cos_hi = base_cos_boot.quantile(0.975, dim=-1)
    torch.save({
        "v_ref": base_dirs.v_ref.cpu(),
        "v_harm": base_dirs.v_harm.cpu(),
        "norm_pre_ref": base_dirs.norm_pre_ref.cpu(),
        "norm_pre_harm": base_dirs.norm_pre_harm.cpu(),
        "cos_boot_mean": base_cos_mean.cpu(),
        "cos_boot_lo": base_cos_lo.cpu(),
        "cos_boot_hi": base_cos_hi.cpu(),
    }, out / "directions_base.pt")
    from main.layers import BAND_HIGH as _BH, BAND_LOW as _BL
    _n = base_dirs.v_ref.shape[0] - 1
    print("[diag-base] cos(v_ref,v_harm) at M_base [95% CI]:")
    for L in range(_BL, _n - _BH + 1):
        print(f"  L{L:>2d}: {base_cos_mean[L].item():+.3f} "
              f"[{base_cos_lo[L].item():+.3f}, {base_cos_hi[L].item():+.3f}]")

    base_resp_dirs: ResponseDirections | None = None
    cur_resp_dirs: ResponseDirections | None = None
    if cfg.enable_response_coupling:
        print("[directions] response-side extraction on M_base (300 + 300 + 300)")
        from main.data import CB_TRAIN_PATH as _CB
        import json as _j, random as _r
        cb_full = _j.loads(_CB.read_text())
        cb_idx = list(range(len(cb_full)))
        _r.Random(0).shuffle(cb_idx)
        ext_h_idx = cb_idx[:300]
        harm_pairs = [(cb_full[i]["prompt"], cb_full[i]["output"]) for i in ext_h_idx]
        refuse_pairs = [(cb_full[i]["prompt"], cb_full[i]["llama3_output"]) for i in ext_h_idx]
        helpful_pairs = [(s.prompt, s.response) for s in extract["harmless"]]
        base_resp_dirs = extract_response_directions(
            M_base, tok, harm_pairs, refuse_pairs, helpful_pairs,
            batch_size=cfg.extract_batch_size,
            max_prompt_len=cfg.extract_max_len,
            max_resp_len=cfg.extract_max_len,
        )
        cur_resp_dirs = base_resp_dirs
        torch.save({
            "v_ref_resp": base_resp_dirs.v_ref_resp.cpu(),
            "v_harm_resp": base_resp_dirs.v_harm_resp.cpu(),
            "norm_pre_ref": base_resp_dirs.norm_pre_ref.cpu(),
            "norm_pre_harm": base_resp_dirs.norm_pre_harm.cpu(),
        }, out / "response_directions_base.pt")
        _resp_pairs = {"harm": harm_pairs, "refuse": refuse_pairs,
                       "helpful": helpful_pairs}

    diag_log_f = (out / "diag_log.jsonl").open("w")

    cur_dirs = base_dirs
    K_layers = cfg.K_layers_initial
    selected = _select(cfg, cur_dirs, K_layers, resp_dirs=cur_resp_dirs)
    print(f"[layers] K={K_layers} selected={selected}")
    (out / "selected_layers.jsonl").write_text(
        json.dumps({"step": 0, "K": K_layers, "selected": selected}) + "\n"
    )

    optimizer = torch.optim.AdamW(
        [p for p in M_lora.parameters() if p.requires_grad],
        lr=cfg.lr,
    )

    def lr_at(step: int) -> float:
        if step < cfg.warmup:
            return cfg.lr * (step + 1) / cfg.warmup
        return cfg.lr

    log_path = out / "train_log.jsonl"
    log_f = log_path.open("w")

    optimizer.zero_grad()
    accum = 0
    t0 = time.time()

    for step in range(cfg.max_steps):
        if step == cfg.K_ramp_step and K_layers != cfg.K_layers_final:
            K_layers = cfg.K_layers_final
            selected = _select(cfg, cur_dirs, K_layers, resp_dirs=cur_resp_dirs)
            print(f"[step {step}] K-ramp -> K={K_layers} selected={selected}")
            with (out / "selected_layers.jsonl").open("a") as f:
                f.write(json.dumps({"step": step, "K": K_layers, "selected": selected}) + "\n")

        if step > 0 and step % cfg.K_recompute == 0:
            print(f"[step {step}] recompute directions (EMA, beta={cfg.beta_ema})")
            new_dirs, drift, act_h_now, act_s_now = recompute_directions(
                M_lora, tok, extract["harmful"], extract["harmless"],
                base_dirs=base_dirs, prev_dirs=cur_dirs,
                beta=cfg.beta_ema, cfg=cfg, return_activations=True,
            )
            cur_dirs = new_dirs

            if cfg.enable_response_coupling and base_resp_dirs is not None:
                M_lora.eval()
                fresh_resp = extract_response_directions(
                    M_lora.base_model.model if hasattr(M_lora, "base_model") else M_lora,
                    tok,
                    _resp_pairs["harm"], _resp_pairs["refuse"], _resp_pairs["helpful"],
                    batch_size=cfg.extract_batch_size,
                    max_prompt_len=cfg.extract_max_len,
                    max_resp_len=cfg.extract_max_len,
                )
                M_lora.train()
                v_ref_blend = ema_blend(base_resp_dirs.v_ref_resp,
                                        fresh_resp.v_ref_resp, cfg.beta_ema)
                v_harm_blend = ema_blend(base_resp_dirs.v_harm_resp,
                                         fresh_resp.v_harm_resp, cfg.beta_ema)
                cur_resp_dirs = ResponseDirections(
                    v_ref_resp=v_ref_blend, v_harm_resp=v_harm_blend,
                    norm_pre_ref=base_resp_dirs.norm_pre_ref,
                    norm_pre_harm=base_resp_dirs.norm_pre_harm,
                )
            selected_new = _select(cfg, cur_dirs, K_layers, resp_dirs=cur_resp_dirs)
            print(f"[recompute drift] cos_ref(prev,fresh): "
                  f"min={drift['cos_ref_fresh_min']:.3f} mean={drift['cos_ref_fresh_mean']:.3f} ; "
                  f"cos_harm(prev,fresh): min={drift['cos_harm_fresh_min']:.3f} "
                  f"mean={drift['cos_harm_fresh_mean']:.3f} ; selected: {selected} -> {selected_new}")
            selected = selected_new
            with (out / "selected_layers.jsonl").open("a") as f:
                f.write(json.dumps({
                    "step": step, "K": K_layers, "selected": selected, **drift,
                }) + "\n")

            cos_ref_harm = (cur_dirs.v_ref * cur_dirs.v_harm).sum(-1)
            cos_ref_base = (cur_dirs.v_ref * base_dirs.v_ref).sum(-1)
            cos_harm_base = (cur_dirs.v_harm * base_dirs.v_harm).sum(-1)

            cos_boot = bootstrap_cos_ref_harm(act_h_now, act_s_now, K=100, seed=step)
            cos_boot_mean = cos_boot.mean(dim=-1)
            cos_boot_lo = cos_boot.quantile(0.025, dim=-1)
            cos_boot_hi = cos_boot.quantile(0.975, dim=-1)

            M_lora.eval()
            proj = project_validation(
                M_lora.base_model.model if hasattr(M_lora, "base_model") else M_lora,
                tok, validate["harmful"], validate["harmless"],
                v_ref=cur_dirs.v_ref, v_harm=cur_dirs.v_harm,
                layer_indices=selected,
                batch_size=cfg.extract_batch_size, max_length=cfg.extract_max_len,
            )
            M_lora.train()

            from main.layers import BAND_HIGH, BAND_LOW
            n_layers_total = cur_dirs.v_ref.shape[0] - 1
            band_layers = list(range(BAND_LOW, n_layers_total - BAND_HIGH + 1))
            full_cos: dict[str, float] = {}
            for L in band_layers:
                full_cos[f"cos_ref_harm_L{L}"] = float(cos_ref_harm[L].item())
                full_cos[f"cos_boot_mean_L{L}"] = float(cos_boot_mean[L].item())
                full_cos[f"cos_boot_lo_L{L}"] = float(cos_boot_lo[L].item())
                full_cos[f"cos_boot_hi_L{L}"] = float(cos_boot_hi[L].item())

            geom_sel = {
                **{f"drift_ref_base_L{L}": float(cos_ref_base[L].item()) for L in selected},
                **{f"drift_harm_base_L{L}": float(cos_harm_base[L].item()) for L in selected},
            }

            diag_rec = {"step": step, "selected": selected,
                        **geom_sel, **full_cos}
            for k, sub in proj.items():
                for L, v in sub.items():
                    diag_rec[f"{k}_L{L}"] = v
            diag_log_f.write(json.dumps(diag_rec) + "\n")
            diag_log_f.flush()

            torch.save({
                "step": step,
                "v_ref": cur_dirs.v_ref.cpu(),
                "v_harm": cur_dirs.v_harm.cpu(),
            }, out / f"directions_step_{step:06d}.pt")

            sel_strs = []
            for L in selected:
                sel_strs.append(
                    f"L{L}={cos_boot_mean[L].item():+.3f}"
                    f"[{cos_boot_lo[L].item():+.3f},{cos_boot_hi[L].item():+.3f}]"
                )
            ph_h = sum(proj["proj_ref_harmful"].values()) / max(1, len(selected))
            ph_b = sum(proj["proj_ref_harmless"].values()) / max(1, len(selected))
            pm_h = sum(proj["proj_harm_harmful"].values()) / max(1, len(selected))
            pm_b = sum(proj["proj_harm_harmless"].values()) / max(1, len(selected))
            print(f"[diag] cos(v_ref,v_harm) {' '.join(sel_strs)}")
            print(f"[diag] val proj_ref  harmful={ph_h:+.3f}  harmless={ph_b:+.3f}"
                  f"  | proj_harm  harmful={pm_h:+.3f}  harmless={pm_b:+.3f}")

        samples = sample_batch(iters, counts)
        batch = collate(samples, tok)

        v_ref_resp_now = cur_resp_dirs.v_ref_resp if cur_resp_dirs is not None else None
        v_harm_resp_now = cur_resp_dirs.v_harm_resp if cur_resp_dirs is not None else None
        losses = step_loss(M_lora_h, M_base_h, batch, selected,
                           cur_dirs.v_ref, cur_dirs.v_harm, cfg, device,
                           v_ref_resp=v_ref_resp_now, v_harm_resp=v_harm_resp_now)
        L = losses["total"] / cfg.grad_accum
        L.backward()
        accum += 1

        if accum == cfg.grad_accum:
            for pg in optimizer.param_groups:
                pg["lr"] = lr_at(step)
            optimizer.step()
            optimizer.zero_grad()
            accum = 0

        if step % 10 == 0:
            elapsed = time.time() - t0
            rec = {
                "step": step, "elapsed": round(elapsed, 1),
                "L_total": float(losses["total"].item()),
                "L_couple": float(losses["couple"].item()),
                "L_couple_harmful": float(losses["couple_harmful"].item()),
                "L_couple_retain": float(losses["couple_retain"].item()),
                "L_couple_resp": float(losses["couple_resp"].item()),
                "L_couple_resp_harmful": float(losses["couple_resp_harmful"].item()),
                "L_couple_resp_retain": float(losses["couple_resp_retain"].item()),
                "L_kl": float(losses["kl"].item()),
                "L_ce": float(losses["ce"].item()),
                "K": K_layers, "selected": selected,
            }
            for k in ("proj_ref_harmful", "proj_harm_harmful",
                      "proj_ref_retain", "proj_harm_retain",
                      "proj_ref_resp_harmful", "proj_harm_resp_harmful",
                      "proj_ref_resp_retain", "proj_harm_resp_retain"):
                if k in losses:
                    rec[k] = float(losses[k].item())
            print(json.dumps(rec))
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()

    log_f.close()
    diag_log_f.close()
    print(f"[done] trained {cfg.max_steps} steps")
    M_lora.save_pretrained(out / "final")
    tok.save_pretrained(out / "final")


if __name__ == "__main__":
    main()
