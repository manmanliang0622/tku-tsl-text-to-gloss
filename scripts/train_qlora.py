#!/usr/bin/env python3
"""Stage B：Gemma 4 E4B QLoRA 監督式微調（計畫第 5 節 Stage B）。

依據：官方 QLoRA 教學路徑（transformers + peft + bitsandbytes）[來源6]；
先例 SCOPE（Q-LoRA on Qwen2）[來源3]、SignAlignLM（LoRA on LLaMA3）[來源1]。

資料：scripts/split_data.py 產出的 data/splits/{train,dev}.jsonl。
提示格式與 Stage A 一致（scripts/prompt_common.py），只在 assistant 段（Gloss）算 loss。

用法（在 VM venv 內）：
  python scripts/train_qlora.py --epochs 3
  python scripts/train_qlora.py --model google/gemma-4-E4B-it --output outputs/qlora_e4b
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

import prompt_common as pc

BASE = Path(__file__).resolve().parent.parent


def load_split(name):
    rows = [json.loads(l) for l in (BASE / "data" / "splits" / f"{name}.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    return Dataset.from_list([
        {"messages": pc.build_messages(r["chinese"], r["gloss_text"])} for r in rows
    ])


def load_model(model_id, bnb_config):
    """Gemma 4 為多模態架構，優先試 CausalLM，失敗改 ImageTextToText。"""
    last_err = None
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config,
            dtype=torch.bfloat16, device_map="auto")
    except Exception as e:
        last_err = e
    try:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(
            model_id, quantization_config=bnb_config,
            dtype=torch.bfloat16, device_map="auto")
    except Exception as e:
        raise RuntimeError(f"CausalLM 與 ImageTextToText 皆失敗：{last_err} / {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--output", default=str(BASE / "outputs" / "qlora_e4b"))
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    args = ap.parse_args()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, bnb)
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    train_ds, dev_ds = load_split("train"), load_split("dev")
    print(f"train={len(train_ds)} dev={len(dev_ds)}")

    cfg = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        max_length=args.max_len,
        assistant_only_loss=True,   # 只在 Gloss（assistant）段算 loss
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=42,
    )
    trainer = SFTTrainer(
        model=model, args=cfg,
        train_dataset=train_ds, eval_dataset=dev_ds,
        processing_class=tokenizer, peft_config=lora,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"DONE. adapter saved to {args.output}")


if __name__ == "__main__":
    main()
