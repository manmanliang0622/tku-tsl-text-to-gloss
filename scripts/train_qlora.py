#!/usr/bin/env python3
"""Stage B：Gemma 4 E4B QLoRA 監督式微調（計畫第 5 節 Stage B）。

依據：官方 QLoRA 教學路徑（transformers + peft + bitsandbytes）[來源6]；
先例 SCOPE（Q-LoRA on Qwen2）[來源3]、SignAlignLM（LoRA on LLaMA3）[來源1]。

資料：scripts/split_data.py 產出的 data/splits/{train,dev}.jsonl。
提示格式與 Stage A 一致（scripts/prompt_common.py）；**只在 assistant 段（Gloss）算 loss**，
prompt（任務描述＋規則＋中文）以 label=-100 遮罩，避免長 prompt 淹沒學習訊號。

記憶體策略：Gemma 4 E4B 的 Per-Layer Embedding（5.6GB bf16，bnb 量化不到）與視覺／
音訊塔 offload 到 CPU，GPU 只留量化後的 transformer 層（常駐約 3.4GB），
以配合本機共用 GPU（RTX 4060 Ti 16GB，另有生產服務佔用）。

用法（VM venv 內，建議設 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True）：
  python scripts/train_qlora.py --epochs 3
  python scripts/train_qlora.py --max-steps 2 --batch 1   # 冒煙測試
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (AutoTokenizer, BitsAndBytesConfig, Trainer,
                          TrainingArguments)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import prompt_common as pc

BASE = Path(__file__).resolve().parent.parent

def load_model(model_id, bnb_config):
    """載入 Gemma 4 E4B（4-bit），並把 5.6GB 的 Per-Layer Embedding 表搬到 CPU。

    PLE（embed_tokens_per_layer）是 nn.Embedding，bnb 量化不到、以 bf16 佔 5.6GB。
    它只是查表：每步輸出僅約 11MB。若用 accelerate device_map offload，前向時
    整張表會被搬上 GPU（且 fp32＝10.5GB）而 OOM。故改為：整個載入 GPU 後，
    把 PLE 搬到 CPU 常駐，並改寫其 forward 讓查表在 CPU 進行、只回傳小輸出到 GPU。
    GPU 常駐從 9.3GB 降到約 3.7GB。
    """
    from transformers import Gemma4ForConditionalGeneration
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_id, quantization_config=bnb_config,
        dtype=torch.bfloat16, device_map={"": 0})

    ple = model.model.language_model.embed_tokens_per_layer
    ple.to("cpu")
    torch.cuda.empty_cache()
    _orig_forward = ple.forward

    def cpu_lookup(input_ids, *a, **k):
        dev = input_ids.device
        return _orig_forward(input_ids.to("cpu"), *a, **k).to(dev)

    ple.forward = cpu_lookup
    return model


def build_dataset(name, tokenizer, max_len):
    """把 {chinese, gloss_text} 轉成 input_ids + labels（prompt 段遮罩為 -100）。"""
    rows = [json.loads(l) for l in (BASE / "data" / "splits" / f"{name}.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]

    def ids(text):
        # 模板文字已含 <bos> 等特殊標記，故 add_special_tokens=False
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    def encode(r):
        prompt_text = tokenizer.apply_chat_template(
            pc.build_messages(r["chinese"]), add_generation_prompt=True,
            tokenize=False)
        full_text = tokenizer.apply_chat_template(
            pc.build_messages(r["chinese"], r["gloss_text"]),
            add_generation_prompt=False, tokenize=False)
        prompt_ids, full_ids = ids(prompt_text), ids(full_text)
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        return {"input_ids": full_ids[:max_len], "labels": labels[:max_len]}

    return Dataset.from_list([encode(r) for r in rows])


class MaskedCollator:
    """動態 padding：input_ids 補 pad、labels 補 -100、建 attention_mask。"""
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        maxlen = max(len(f["input_ids"]) for f in feats)
        input_ids, labels, attn = [], [], []
        for f in feats:
            n = maxlen - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * n)
            labels.append(f["labels"] + [-100] * n)
            attn.append([1] * len(f["input_ids"]) + [0] * n)
        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn),
        }


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
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="限制訓練步數（>0 用於冒煙測試）")
    args = ap.parse_args()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(args.model, bnb)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=r".*language_model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = build_dataset("train", tokenizer, args.max_len)
    dev_ds = build_dataset("dev", tokenizer, args.max_len)
    print(f"train={len(train_ds)} dev={len(dev_ds)}")

    smoke = args.max_steps and args.max_steps > 0
    targs = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="no" if smoke else "epoch",
        save_strategy="no" if smoke else "epoch",
        save_total_limit=2,
        load_best_model_at_end=not smoke,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=42,
    )
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=dev_ds,
        data_collator=MaskedCollator(tokenizer.pad_token_id),
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"DONE. adapter saved to {args.output}")


if __name__ == "__main__":
    main()
