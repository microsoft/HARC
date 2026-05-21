"""HARC loss components: additive-margin coupling (prompt & response side), KL retention, refusal CE."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

COUPLING_MARGIN = 0.3


def _cos(x: Tensor, v: Tensor) -> Tensor:
    x = x.to(v.dtype)
    return F.cosine_similarity(x, v.unsqueeze(0).expand_as(x), dim=-1)


def coupling_loss(
    h_inst: dict[int, Tensor],
    h_post: dict[int, Tensor],
    v_ref: Tensor,
    v_harm: Tensor,
    is_harmful: Tensor,
    margin: float = COUPLING_MARGIN,
) -> dict[str, Tensor]:
    """Push harmful residuals above margin on v_ref/v_harm and retain residuals below zero, per selected layer."""
    layers = sorted(h_inst.keys())
    n_h = is_harmful.sum().clamp_min(1)
    n_r = (~is_harmful).sum().clamp_min(1)

    per_layer_total: list[Tensor] = []
    per_layer_h: list[Tensor] = []
    per_layer_r: list[Tensor] = []
    pr_h_list: list[Tensor] = []
    ph_h_list: list[Tensor] = []
    pr_r_list: list[Tensor] = []
    ph_r_list: list[Tensor] = []
    for L in layers:
        proj_ref = _cos(h_post[L], v_ref[L].detach())
        proj_harm = _cos(h_inst[L], v_harm[L].detach())

        l_h = torch.relu(margin - proj_ref) + torch.relu(margin - proj_harm)
        l_b = torch.relu(proj_ref) + torch.relu(proj_harm)

        per_sample = torch.where(is_harmful, l_h, l_b)
        per_layer_total.append(per_sample.mean())

        if is_harmful.any():
            per_layer_h.append(l_h[is_harmful].mean())
            pr_h_list.append(proj_ref[is_harmful].mean().detach())
            ph_h_list.append(proj_harm[is_harmful].mean().detach())
        else:
            per_layer_h.append(torch.zeros((), device=proj_ref.device))
        if (~is_harmful).any():
            per_layer_r.append(l_b[~is_harmful].mean())
            pr_r_list.append(proj_ref[~is_harmful].mean().detach())
            ph_r_list.append(proj_harm[~is_harmful].mean().detach())
        else:
            per_layer_r.append(torch.zeros((), device=proj_ref.device))

    out = {
        "total": torch.stack(per_layer_total).mean(),
        "harmful": torch.stack(per_layer_h).mean().detach(),
        "retain": torch.stack(per_layer_r).mean().detach(),
    }
    if pr_h_list:
        out["proj_ref_harmful"] = torch.stack(pr_h_list).mean()
        out["proj_harm_harmful"] = torch.stack(ph_h_list).mean()
    if pr_r_list:
        out["proj_ref_retain"] = torch.stack(pr_r_list).mean()
        out["proj_harm_retain"] = torch.stack(ph_r_list).mean()
    return out


def response_coupling_loss(
    h_resp: dict[int, Tensor],
    v_ref_resp: Tensor,
    v_harm_resp: Tensor,
    is_harmful: Tensor,
    margin: float = COUPLING_MARGIN,
) -> dict[str, Tensor]:
    """Coupling loss on response-side residuals (h_resp[L] = (B,H) residual mean-pooled over the response window)."""
    layers = sorted(h_resp.keys())
    per_layer_total: list[Tensor] = []
    per_layer_h: list[Tensor] = []
    per_layer_r: list[Tensor] = []
    pr_h_list: list[Tensor] = []
    ph_h_list: list[Tensor] = []
    pr_r_list: list[Tensor] = []
    ph_r_list: list[Tensor] = []
    for L in layers:
        proj_ref = _cos(h_resp[L], v_ref_resp[L].detach())
        proj_harm = _cos(h_resp[L], v_harm_resp[L].detach())

        l_h = torch.relu(margin - proj_ref) + torch.relu(margin - proj_harm)
        l_b = torch.relu(proj_ref) + torch.relu(proj_harm)
        per_sample = torch.where(is_harmful, l_h, l_b)
        per_layer_total.append(per_sample.mean())

        if is_harmful.any():
            per_layer_h.append(l_h[is_harmful].mean())
            pr_h_list.append(proj_ref[is_harmful].mean().detach())
            ph_h_list.append(proj_harm[is_harmful].mean().detach())
        else:
            per_layer_h.append(torch.zeros((), device=proj_ref.device))
        if (~is_harmful).any():
            per_layer_r.append(l_b[~is_harmful].mean())
            pr_r_list.append(proj_ref[~is_harmful].mean().detach())
            ph_r_list.append(proj_harm[~is_harmful].mean().detach())
        else:
            per_layer_r.append(torch.zeros((), device=proj_ref.device))

    out = {
        "total": torch.stack(per_layer_total).mean(),
        "harmful": torch.stack(per_layer_h).mean().detach(),
        "retain": torch.stack(per_layer_r).mean().detach(),
    }
    if pr_h_list:
        out["proj_ref_resp_harmful"] = torch.stack(pr_h_list).mean()
        out["proj_harm_resp_harmful"] = torch.stack(ph_h_list).mean()
    if pr_r_list:
        out["proj_ref_resp_retain"] = torch.stack(pr_r_list).mean()
        out["proj_harm_resp_retain"] = torch.stack(ph_r_list).mean()
    return out


def kl_retain_loss(
    lora_logits: Tensor,
    base_logits: Tensor,
    response_mask: Tensor,
) -> Tensor:
    """Token-mean KL(base || lora) over response positions (distillation toward base on benign)."""
    base_logp = F.log_softmax(base_logits.detach().float(), dim=-1)
    lora_logp = F.log_softmax(lora_logits.float(), dim=-1)
    base_p = base_logp.exp()
    kl = (base_p * (base_logp - lora_logp)).sum(dim=-1)

    mask = response_mask.float()
    denom = mask.sum().clamp_min(1.0)
    return (kl * mask).sum() / denom


def ce_refusal_loss(
    lora_logits: Tensor,
    labels: Tensor,
) -> Tensor:
    """CE on response tokens, shifted so position t predicts token t+1."""
    shift_logits = lora_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
    )
