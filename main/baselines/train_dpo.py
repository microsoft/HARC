"""DPO training (trl + PKU-SafeRLHF, LoRA); standalone, or HARC+DPO via --init_lora_dir."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Shim: stub feature-flag helpers / _VALID_DICT_FIELDS that trl 1.2.0 expects but older transformers lacks, before importing trl.
import transformers as _tr
import transformers.utils as _tru
if not hasattr(_tr.TrainingArguments, "_VALID_DICT_FIELDS"):
    _tr.TrainingArguments._VALID_DICT_FIELDS = []
def _make_stub(name):
    def _stub(): return False
    _stub.__name__ = name
    return _stub
for _name in (
    "is_trackio_available", "is_swanlab_available", "is_dvclive_available",
    "is_clearml_available", "is_codecarbon_available",
):
    if not hasattr(_tr, _name):
        setattr(_tr, _name, _make_stub(_name))
for _name in (
    "is_rich_available", "is_trackio_available", "is_swanlab_available",
    "is_liger_kernel_available", "is_kernels_available",
):
    if not hasattr(_tru, _name):
        setattr(_tru, _name, _make_stub(_name))

from trl import DPOConfig, DPOTrainer


def build_pairs(n_pairs: int = 3000, seed: int = 0):
    """(prompt, chosen, rejected) pairs from PKU-SafeRLHF where exactly one response is safe."""
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
    rows = []
    for r in ds:
        if r["is_response_0_safe"] == r["is_response_1_safe"]:
            continue
        if r["is_response_0_safe"]:
            chosen, rejected = r["response_0"], r["response_1"]
        else:
            chosen, rejected = r["response_1"], r["response_0"]
        rows.append({"prompt": r["prompt"], "chosen": chosen, "rejected": rejected})
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(rows), generator=rng).tolist()[:n_pairs]
    return Dataset.from_list([rows[i] for i in perm])


def format_chat(tok, prompt: str) -> str:
    msgs = [{"role": "user", "content": prompt}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_pairs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--num_epochs", type=float, default=1.0)
    ap.add_argument("--per_device_batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--max_prompt_length", type=int, default=512)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--init_lora_dir", type=str, default=None,
                    help="If provided, load + merge this LoRA into the base "
                         "before adding the DPO LoRA on top (the HARC+DPO setup, "
                         "initialized from a trained HARC LoRA).")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = False

    if args.init_lora_dir:
        from peft import PeftModel
        print(f"[init] loading + merging starting LoRA from {args.init_lora_dir}")
        model = PeftModel.from_pretrained(model, args.init_lora_dir)
        model = model.merge_and_unload()
        print("[init] merged. Will train DPO LoRA on top of merged weights.")

    print(f"[data] building DPO pairs (n={args.n_pairs}) from PKU-SafeRLHF")
    raw = build_pairs(args.n_pairs)
    raw = raw.map(lambda r: {"prompt": format_chat(tok, r["prompt"]),
                              "chosen": r["chosen"], "rejected": r["rejected"]})
    print(f"[data] {len(raw)} pairs")

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )

    cfg = DPOConfig(
        output_dir=str(args.out_dir),
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        remove_unused_columns=False,
        warmup_steps=50,
        lr_scheduler_type="linear",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=cfg,
        train_dataset=raw,
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(args.out_dir / "final"))
    print(f"[save] {args.out_dir / 'final'}")


if __name__ == "__main__":
    main()
