"""Layer selection (variant-D): score = (1-|cos(v_ref,v_harm)|)(1-|cos(v_ref_resp,v_harm_resp)|)|cos(v_harm,v_harm_resp)||cos(v_ref,v_ref_resp)|, top-K within band."""
from __future__ import annotations

import torch
from torch import Tensor

from main.directions import Directions, ResponseDirections


BAND_LOW = 4
BAND_HIGH = 4


def _cos_along(a: Tensor, b: Tensor) -> Tensor:
    num = (a * b).sum(dim=-1)
    den = (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-8)
    return num / den


def layer_scores(dirs: Directions, resp_dirs: ResponseDirections) -> Tensor:
    cos_p = _cos_along(dirs.v_ref,           dirs.v_harm)
    cos_r = _cos_along(resp_dirs.v_ref_resp, resp_dirs.v_harm_resp)
    rho_h = _cos_along(dirs.v_harm,          resp_dirs.v_harm_resp).abs()
    rho_r = _cos_along(dirs.v_ref,           resp_dirs.v_ref_resp).abs()

    sigma_p = 1.0 - cos_p.abs()
    sigma_r = 1.0 - cos_r.abs()
    score = (sigma_p * sigma_r * rho_h * rho_r).clone()

    n = score.numel()
    score[:BAND_LOW] = float("-inf")
    score[n - BAND_HIGH:] = float("-inf")
    return score


def select_layers(dirs: Directions, resp_dirs: ResponseDirections,
                  k: int) -> list[int]:
    score = layer_scores(dirs, resp_dirs)
    top = torch.topk(score, k=k).indices.tolist()
    return sorted(top)
