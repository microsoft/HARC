<!--
 * @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @Date: 2026-05-21 12:12:12
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2026-05-21 12:18:32
 * @FilePath: /sheichua/geometric-safety/harc/README.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
<h1 align="center">HARC: Harmfulness And Refusal Coupling</h1>

<p align="center">
  <a href=""><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg" alt="Paper"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
</p>

<p align="center"><em>Official implementation of "HARC: Coupling Harmfulness and Refusal Directions for Robust Safety Alignment".</em></p>

HARC is a LoRA safety-alignment method that **couples a model's prompt- and response-side hidden states to its harmfulness (`v_harm`) and refusal (`v_ref`) directions**. This repo covers (1) prompt- and response-side direction extraction and (2) LoRA fine-tuning with the coupling objective (plus the **HARC + DPO** variant).

## Layout
```
main/                       core training package (run as `python -m main.<module>` from repo root)
  train.py                  HARC training loop (LoRA + coupling + KL + CE)
  directions.py             v_harm / v_ref extraction (prompt & response side)
  extract_paper_method.py   AdvBench/Alpaca direction extraction (paper method)
  losses.py                 coupling / KL / CE losses
  layers.py                 layer selection (variant-D)
  data.py  collate.py       data mix (Circuit-Breakers + UltraChat + XSTest) and batching
  configs/                  training YAMLs (llama3.1_8b.yaml, qwen2_5_7b.yaml)
  baselines/train_dpo.py    DPO trainer (standalone, or HARC+DPO via --init_lora_dir)
prepare_data.py             download/stage the datasets into data/
scripts/train.sh            launcher: train both models in parallel (one GPU each)
HYPERPARAMS.md              training hyperparameters
data/                       datasets (git-ignored; populate with prepare_data.py)
```

## Setup
Run all commands from the repo root.
```bash
pip install -r requirements.txt
python prepare_data.py --download --hf-cache
```

## Train
```bash
python -m main.train --config main/configs/llama3.1_8b.yaml   # Llama-3.1-8B-Instruct
python -m main.train --config main/configs/qwen2_5_7b.yaml    # Qwen2.5-7B-Instruct
# or both in parallel (GPUs 0 and 1 by default):
bash scripts/train.sh
```
Each config reproduces the checkpoint reported in the paper. A run writes the adapter to `out_dir/final/`, the extracted directions (`directions_base.pt`, `response_directions_base.pt`), and logs (`train_log.jsonl`, `diag_log.jsonl`, `selected_layers.jsonl`).

### HARC + DPO
Apply DPO on top of a trained HARC LoRA (merges the HARC adapter into the base first):
```bash
python -m main.baselines.train_dpo \
    --model_id meta-llama/Llama-3.1-8B-Instruct \
    --init_lora_dir runs/harc_llama3.1_8b/final \
    --out_dir runs/harc_dpo
```

## Using a trained adapter
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "meta-llama/Llama-3.1-8B-Instruct"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, "runs/harc_llama3.1_8b/final").eval()

msgs = [{"role": "user", "content": "How do I bake a cake?"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=256)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
```

## Ethical Statement
HARC is a safety-alignment method: its purpose is to make language models more robust to jailbreak attacks and harmful requests while preserving their helpfulness on benign inputs. The datasets used here (Circuit-Breakers, AdvBench, XSTest, etc.) contain harmful prompts solely to train and measure refusal behavior. We release this code to support reproducible safety research and intend it to be used only for defensive purposes, in accordance with the licenses and intended uses of the underlying models and datasets. We do not condone using this work to facilitate harm.

## Citation
If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{harc2026,
  title   = {HARC: Coupling Harmfulness and Refusal Directions for Robust Safety Alignment},
  author  = {TODO},
  journal = {arXiv preprint arXiv:TODO},
  year    = {2026}
}
```
