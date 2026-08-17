#!/usr/bin/env python3
"""
festo_llama3.2_exp05 — fine-tune script
QLoRA (4-bit) + 8-bit AdamW + TRL SFTTrainer + LoRA adapter.
Reads HF token from .env one level up (HUGGING_FACE_KEY).
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import time
import gc
import random
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    TrainerCallback,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Load HF token from .env
env_path = ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if line.startswith("HUGGING_FACE_KEY"):
            tok = line.split("=", 1)[1].strip().strip('"').strip("'")
            os.environ["HF_TOKEN"] = tok
            os.environ["HUGGINGFACE_HUB_TOKEN"] = tok
            break

MODEL_NAME = os.environ.get("EXP4_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
DATA_PATH = HERE / "dataset_exp05.jsonl"
OUTPUT_DIR = HERE / "festo_llama3.2_finetuned_exp05"
MAX_SEQ_LEN = 512
SEED = 3407

random.seed(SEED)
torch.manual_seed(SEED)

print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
torch.cuda.init()
torch.cuda.empty_cache()
gc.collect()
print(f"Model: {MODEL_NAME}")
print(f"Data: {DATA_PATH}")
print(f"Out:  {OUTPUT_DIR}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.config.use_cache = False
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    use_rslora=True,
)

# ---------- data ----------
print("Loading dataset...")
rows = []
with DATA_PATH.open() as f:
    for line in f:
        item = json.loads(line)
        text = tokenizer.apply_chat_template(
            item["messages"], tokenize=False, add_generation_prompt=False
        )
        rows.append({"text": text})

random.shuffle(rows)
split = int(0.98 * len(rows))
train_rows = rows[:split]
val_rows = rows[split:split + 250]
print(f"Train: {len(train_rows)} | Val: {len(val_rows)}")

train_ds = Dataset.from_list(train_rows)
val_ds = Dataset.from_list(val_rows)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class ClearCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            torch.cuda.empty_cache()
        return control


sft_cfg = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=5,
    learning_rate=2e-4,        # LoRA adapter LR (course note: bump vs full-FT)
    warmup_steps=100,
    fp16=True,
    logging_steps=50,
    optim="adamw_8bit",
    weight_decay=0.01,
    max_grad_norm=0.3,
    lr_scheduler_type="cosine",
    seed=SEED,
    save_steps=400,
    save_total_limit=2,
    eval_strategy="no",
    eval_steps=200,
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    gradient_checkpointing=True,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    max_seq_length=MAX_SEQ_LEN,
    packing=False,
    dataset_text_field="text",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=sft_cfg,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    peft_config=lora_config,
    callbacks=[
        
        ClearCacheCallback(),
    ],
)

print("Trainable parameters:")
trainer.model.print_trainable_parameters()

print("Training...")
start = time.time()

ckpt = None
checkpoints = sorted(p for p in OUTPUT_DIR.glob("checkpoint-*") if p.is_dir())
if checkpoints:
    ckpt = str(checkpoints[-1])
    print(f"Resuming from: {ckpt}")

trainer.train(resume_from_checkpoint=ckpt)

elapsed = (time.time() - start) / 60
best = trainer.state.best_metric
print(f"Done in {elapsed:.1f} min, best eval_loss={best:.4f}")

trainer.model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

(OUTPUT_DIR / "info.json").write_text(json.dumps({
    "model": MODEL_NAME,
    "lora_r": 16,
    "lora_alpha": 32,
    "epochs": 5,
    "lr": 2e-4,
    "best_eval_loss": float(best) if best is not None else None,
    "train_size": len(train_rows),
    "val_size": len(val_rows),
    "elapsed_min": elapsed,
}, indent=2))

print(f"Saved to {OUTPUT_DIR}")
