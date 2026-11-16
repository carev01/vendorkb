"""Merge a LoRA adapter into Qwen3.5-9B bf16 weights (runs ON the instance).

usage: [SRC=/workspace/train/out/best] [DST=/workspace/merged] python merge_lora.py
"""
import json
import os
from pathlib import Path

import torch
from unsloth import FastLanguageModel

SRC = os.environ.get("SRC", "/workspace/train/out/best")
DST = os.environ.get("DST", "/workspace/merged")
model, tok = FastLanguageModel.from_pretrained(model_name=SRC, max_seq_length=8192,
                                               load_in_4bit=False, dtype=torch.bfloat16)
model.save_pretrained_merged(DST, tok, save_method="merged_16bit")
cfg = json.load(open(Path(DST) / "config.json"))
print("architectures", cfg.get("architectures"), "files", sorted(p.name for p in Path(DST).iterdir()))
