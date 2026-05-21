# Training Hyperparameters

Both HARC and the DPO variant use **LoRA** on top of the same Llama-3.1-8B-Instruct
or Qwen2.5-7B-Instruct base model. Common (unless overridden):

- Precision: `bf16`, `tf32`
- Optimizer: AdamW (lr defined per-method)
- Hardware: single H200 144 GB GPU per training run
- Random seed: `0` (HARC) / `42` (DPO) per the source repos

---

## 1. HARC (geometric coupling)

Source script: [`main/train.py`](main/train.py), config: [`main/configs/llama3.1_8b.yaml`](main/configs/llama3.1_8b.yaml) and [`main/configs/qwen2_5_7b.yaml`](main/configs/qwen2_5_7b.yaml)

**LoRA**
| Parameter | Value |
|---|---|
| `r` | 32 |
| `alpha` | 64 |
| `dropout` | 0.0 |
| `target_modules` | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| trainable params | 84M (1.03% of base) |

**Optimization**
| Parameter | Value |
|---|---|
| `learning_rate` | 1e-4 |
| `warmup_steps` | 100 |
| `lr_scheduler` | constant after warmup |
| `max_steps` | 4000 |
| effective batch | 24 (12 harmful + 8 harmless + 4 pseudo, grad-accum 2) |

**Loss weights**
| Term | Weight (`λ`) |
|---|---|
| `L_couple` (prompt-side) | 1.0 |
| `L_couple_resp` (response-side, 32 tokens mean-pooled) | 1.0 |
| `L_kl` (KL retention vs base on harmless+pseudo response tokens) | 10.0 |
| `L_ce` (refusal CE on harmful response tokens) | 1.0 |
| `coupling_margin` (additive margin) | 0.5 |

**Direction extraction**
| Parameter | Value |
|---|---|
| `extraction_method` | Llama: `default` (CB-train + UltraChat); Qwen: `advbench` (Zhao et al. AdvBench + Alpaca) |
| `extract_batch_size` | 8 |
| `extract_max_len` | 256 |
| `K_recompute` (re-extract directions every N steps) | 200 |
| `beta_ema` (EMA blend of old/fresh directions) | 0.3 |
| `K_layers_initial` | 2 |
| `K_layers_final` | 4 |
| `K_ramp_step` (transition from K=2 to K=4) | 1000 |
| layer score | `(1 − \|cos(v_ref, v_harm)\|) × √(\|v_ref\| × \|v_harm\|)` within band [4, n−4] |

Training runs for the full `max_steps` (no in-loop early stopping); the final adapter is saved to `out_dir/final`.

**Training data** (per `main/data.py`):
- harmful (4,594): `circuit-breakers/data/circuit_breakers_train.json` — prompt + `llama3_output` (refusal CE target)
- harmless (5,000 cap): `HuggingFaceH4/ultrachat_200k` (test_sft) — KL retain target
- pseudo (1,500 cap): `xstest_v2_completions_gpt4_gpteval.csv` `final_label==1_full_compliance` — KL retain (over-refusal mitigation)

---

## 2. DPO

Source: [`main/baselines/train_dpo.py`](main/baselines/train_dpo.py) — uses `trl.DPOTrainer` (trl 1.2.0)

**LoRA**
| Parameter | Value |
|---|---|
| `r` | 16 |
| `alpha` | 16 |
| `dropout` | 0.05 |
| `target_modules` | same 7-module list as HARC |

**Optimization**
| Parameter | Value |
|---|---|
| `learning_rate` | 5e-5 |
| `warmup_steps` | 50 |
| `lr_scheduler` | linear |
| `num_epochs` | 1 |
| `per_device_train_batch_size` | 4 |
| `gradient_accumulation_steps` | 4 |
| effective batch | 16 |

**DPO-specific**
| Parameter | Value |
|---|---|
| `beta` | 0.1 (KL strength) |
| `max_length` | 1024 |
| `max_prompt_length` | 512 |
| reference model | shared base (LoRA-only training; ref = frozen base) |
| total steps | ~188 (3000 pairs / 16 effective batch) |

**Training data** (per `build_pairs()` in `train_dpo.py`):
- `PKU-Alignment/PKU-SafeRLHF` (train split)
- Filter: keep rows where `is_response_0_safe XOR is_response_1_safe == True` (one safe + one unsafe per prompt)
- Sample 3,000 pairs (random shuffle, seed 0)
- chosen = the `is_response_X_safe == True` response; rejected = the unsafe one

---

## 3. HARC + DPO (compose)

Same as **DPO** above, with one extra step:
- Before adding the DPO LoRA, load the HARC LoRA and **merge it into the base**
  weights via `PeftModel.merge_and_unload()` (pass `--init_lora_dir`).
- DPO LoRA is then trained on top of the merged base; produces a single LoRA artifact representing "HARC weights baked in + DPO adjustment".

---

## 4. Side-by-side summary

| Method | LoRA r | LR | Steps | Batch (eff) | Loss components | Source data |
|---|---:|---|---:|---:|---|---|
| **HARC** | 32 | 1e-4 | 4000 | 24 | couple + couple_resp + KL + CE | CB-train + UltraChat + XSTest |
| **DPO** | 16 | 5e-5 | ~188 (1 epoch / 3k pairs) | 16 | DPO loss (β=0.1) | PKU-SafeRLHF (3k pairs) |
| **HARC + DPO** | 16 | 5e-5 | ~188 | 16 | DPO loss on merged-HARC base | PKU-SafeRLHF (3k pairs) |
