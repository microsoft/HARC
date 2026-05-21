# Data (git-ignored — stage with `prepare_data.py`)

HARC training reads these files from `data/` (paths resolved relative to the
repo). Populate them with `python prepare_data.py --from-local <src>` or
`--download`.

## `data/circuit_breakers/`
- `circuit_breakers_train.json` — Circuit Breakers (Zou et al. 2024) train split:
  harmful prompts + refusal targets (CE supervision).
  Source: https://github.com/GraySwanAI/circuit-breakers (`data/`)
- `xstest_v2_completions_gpt4_gpteval.csv` — XSTest (Röttger et al.):
  pseudo-harmful retain set (KL retain / over-refusal mitigation).
  Source: https://github.com/paul-rottger/exaggerated-safety

## `data/advbench/`  (prompt-side direction extraction)
- `advbench.json` — AdvBench harmful behaviors.
- `alpaca_data_instruction.json` — Alpaca harmless instructions.
  Both ship with the paper repo (LLMs_Encode_Harmfulness_Refusal_Separately/data).

## Pulled from the HF Hub at runtime (not staged here)
- `HuggingFaceH4/ultrachat_200k` — harmless / KL-retain
