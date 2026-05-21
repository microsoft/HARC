"""Prepare the datasets HARC training needs into ./data/.

Training reads (paths resolved relative to this repo via DATA_DIR):
  data/circuit_breakers/circuit_breakers_train.json   harmful prompts + refusal targets (CE)
  data/circuit_breakers/xstest_v2_completions_gpt4_gpteval.csv   pseudo-harmful (KL retain)
  data/advbench/advbench.json                          AdvBench harmful (prompt-side direction extraction)
  data/advbench/alpaca_data_instruction.json           Alpaca harmless (prompt-side direction extraction)

Auto-downloaded from the HF Hub at runtime (only pre-cached here):
  HuggingFaceH4/ultrachat_200k    harmless / KL-retain

Usage:
  # Reliable: copy from an existing local checkout that has these files
  python prepare_data.py --from-local /path/to/source_tree
  # Or attempt downloads from the canonical public sources
  python prepare_data.py --download
  # Pre-cache the HF datasets too
  python prepare_data.py --download --hf-cache
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

# dest (relative to data/) -> (canonical public URL, relative source path under --from-local)
FILES = {
    "circuit_breakers/circuit_breakers_train.json": (
        "https://raw.githubusercontent.com/GraySwanAI/circuit-breakers/main/data/circuit_breakers_train.json",
        "alignment-methods/circuit-breakers/data/circuit_breakers_train.json"),
    "circuit_breakers/xstest_v2_completions_gpt4_gpteval.csv": (
        "https://raw.githubusercontent.com/paul-rottger/exaggerated-safety/main/xstest_v2_completions_gpt4_gpteval.csv",
        "alignment-methods/circuit-breakers/data/xstest_v2_completions_gpt4_gpteval.csv"),
    "advbench/advbench.json": (
        None,  # ships with the paper repo LLMs_Encode_Harmfulness_Refusal_Separately/data/
        "geometric-safety/LLMs_Encode_Harmfulness_Refusal_Separately/data/advbench.json"),
    "advbench/alpaca_data_instruction.json": (
        None,
        "geometric-safety/LLMs_Encode_Harmfulness_Refusal_Separately/data/alpaca_data_instruction.json"),
}


def _get(url: str, dest: Path) -> bool:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  [download] {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
        return dest.exists() and dest.stat().st_size > 0
    except Exception as e:
        print(f"  [download FAILED] {type(e).__name__}: {e}", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-local", type=Path, default=None,
                    help="copy the required files from an existing local source tree")
    ap.add_argument("--download", action="store_true",
                    help="download from canonical public URLs (where available)")
    ap.add_argument("--hf-cache", action="store_true",
                    help="pre-cache the HF datasets (ultrachat_200k)")
    args = ap.parse_args()
    if not (args.from_local or args.download or args.hf_cache):
        ap.error("pass --from-local SRC and/or --download and/or --hf-cache")

    ok, missing = [], []
    for rel, (url, src_rel) in FILES.items():
        dest = DATA / rel
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[have] {rel}"); ok.append(rel); continue
        done = False
        if args.from_local:
            src = args.from_local / src_rel
            if src.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dest); print(f"  [copied] {src} -> {rel}"); done = True
            else:
                print(f"  [from-local miss] {src}", file=sys.stderr)
        if not done and args.download and url:
            done = _get(url, dest)
        (ok if done else missing).append(rel)

    if args.hf_cache:
        import os
        os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", str(DATA / "hf_cache")))
        from datasets import load_dataset
        for name, kw in [("HuggingFaceH4/ultrachat_200k", dict(split="test_sft"))]:
            try:
                load_dataset(name, **kw); print(f"[hf-cache] {name} OK")
            except Exception as e:
                print(f"[hf-cache FAILED] {name}: {e}", file=sys.stderr)

    print(f"\n=== prepared {len(ok)}/{len(FILES)} files into {DATA} ===")
    if missing:
        print("MISSING (provide via --from-local, or fetch manually):")
        for m in missing:
            url = FILES[m][0]
            print(f"  - {m}" + (f"   <- {url}" if url else "   (ships with the paper repo; see README)"))


if __name__ == "__main__":
    main()
