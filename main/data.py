"""Training data: Circuit-Breakers harmful + UltraChat harmless + XSTest pseudo-harmful, with held-out extract/validate splits."""
from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
from typing import Iterator

from datasets import load_dataset

CB_TRAIN_PATH = DATA_DIR / "circuit_breakers/circuit_breakers_train.json"
XSTEST_PATH = DATA_DIR / "circuit_breakers/xstest_v2_completions_gpt4_gpteval.csv"

EXTRACT_HARMFUL_N = 300
EXTRACT_HARMLESS_N = 300
VALIDATE_HARMFUL_N = 100
VALIDATE_HARMLESS_N = 100
PSEUDO_CAP = 1500
HARMLESS_CAP = 5000
SPLIT_SEED = 0


@dataclass
class Sample:
    category: str
    prompt: str
    response: str


def _load_cb_train() -> list[dict]:
    with CB_TRAIN_PATH.open() as f:
        return json.load(f)


def _load_xstest_compliant() -> list[dict]:
    """XSTest rows whose gold completion is full compliance (i.e. truly benign)."""
    with XSTEST_PATH.open(newline="") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    return [r for r in rows if r["final_label"] == "1_full_compliance"]


def _load_ultrachat(n: int, seed: int) -> list[tuple[str, str]]:
    """Up to n (user_first_turn, assistant_first_turn) pairs from ultrachat_200k test_sft."""
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    out: list[tuple[str, str]] = []
    for i in indices:
        msgs = ds[i]["messages"]
        if len(msgs) < 2 or msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
            continue
        out.append((msgs[0]["content"], msgs[1]["content"]))
        if len(out) >= n:
            break
    return out


def build_splits() -> dict[str, list[Sample] | dict[str, list[Sample]]]:
    """Seeded D_train (3 cats) + D_extract (direction estimation) + D_validate (diagnostics) splits."""
    rng = random.Random(SPLIT_SEED)

    cb = _load_cb_train()
    cb_idx = list(range(len(cb)))
    rng.shuffle(cb_idx)
    a, b = EXTRACT_HARMFUL_N, EXTRACT_HARMFUL_N + VALIDATE_HARMFUL_N
    cb_extract = [cb[i] for i in cb_idx[:a]]
    cb_validate = [cb[i] for i in cb_idx[a:b]]
    cb_train = [cb[i] for i in cb_idx[b:]]

    uc_total = HARMLESS_CAP + EXTRACT_HARMLESS_N + VALIDATE_HARMLESS_N
    uc = _load_ultrachat(n=uc_total, seed=SPLIT_SEED)
    a, b = EXTRACT_HARMLESS_N, EXTRACT_HARMLESS_N + VALIDATE_HARMLESS_N
    uc_extract = uc[:a]
    uc_validate = uc[a:b]
    uc_train = uc[b:]

    xs = _load_xstest_compliant()
    rng.shuffle(xs)
    # Oversample the ~230 unique compliant rows up to PSEUDO_CAP.
    multiplier = (PSEUDO_CAP + len(xs) - 1) // len(xs)
    xs_train = (xs * multiplier)[:PSEUDO_CAP]

    train = {
        "harmful": [Sample("harmful", r["prompt"], r["llama3_output"]) for r in cb_train],
        "harmless": [Sample("harmless", u, a) for (u, a) in uc_train],
        "pseudo": [Sample("pseudo", r["prompt"], r["completion"]) for r in xs_train],
    }
    extract = {
        "harmful": [Sample("harmful", r["prompt"], r["llama3_output"]) for r in cb_extract],
        "harmless": [Sample("harmless", u, a) for (u, a) in uc_extract],
    }
    validate = {
        "harmful": [Sample("harmful", r["prompt"], r["llama3_output"]) for r in cb_validate],
        "harmless": [Sample("harmless", u, a) for (u, a) in uc_validate],
    }
    return {"train": train, "extract": extract, "validate": validate}


def category_iterator(samples: list[Sample], seed: int) -> Iterator[Sample]:
    """Infinite shuffled iterator (re-shuffles each epoch)."""
    rng = random.Random(seed)
    pool = list(samples)
    while True:
        rng.shuffle(pool)
        for s in pool:
            yield s


def sample_batch(
    iters: dict[str, Iterator[Sample]],
    counts: dict[str, int],
) -> list[Sample]:
    batch: list[Sample] = []
    for cat, n in counts.items():
        for _ in range(n):
            batch.append(next(iters[cat]))
    return batch
