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

# device_map：把 PLE（5.6GB embedding）與視覺/音訊塔放 CPU，其餘量化層放 GPU。
# 用 device_map 而非載入後 .to()，是因為 accelerate 會標記此為 dispatched model，
# HF Trainer 就不會呼叫 model.to(device)（那會把 PLE 搬回 GPU 而 OOM）。
GEMMA4_OFFLOAD_MAP = {
    "model.language_model.embed_tokens_per_layer": "cpu",
    "model.vision_tower": "cpu",
    "model.audio_tower": "cpu",
    "model.embed_vision": "cpu",
    "model.embed_audio": "cpu",
    "model.language_model.embed_tokens": 0,
    "model.language_model.layers": 0,
    "model.language_model.norm": 0,
    "model.language_model.rotary_emb": 0,
    "model.language_model.per_layer_model_projection": 0,
    "model.language_model.per_layer_projection_norm": 0,
    "lm_head": 0,
}


FULL_GPU_BYTES = 9.1 * 1024 ** 3    # 實測全模型放 GPU 所需（PLE 5.25＋層 3.1＋視聽 0.89）
FULL_GPU_MARGIN = 1.0 * 1024 ** 3   # 共用生產機至少保留的餘裕


def can_fit_ple_on_gpu():
    """顯存是否足以把整個模型放 GPU（此機為共用生產機，需保留餘裕）。"""
    if not torch.cuda.is_available():
        return False
    free, _ = torch.cuda.mem_get_info()
    return free > FULL_GPU_BYTES + FULL_GPU_MARGIN


def load_model(model_id, bnb_config, ple_on_gpu=False):
    """載入 Gemma 4 E4B（4-bit）。

    PLE（embed_tokens_per_layer）是 nn.Embedding，bnb 量化不到、bf16 實測 5.25GB。

    - `ple_on_gpu=False`（訓練預設）：PLE 常駐 CPU、查表在 CPU 執行，只回傳小輸出到
      GPU，GPU 常駐約 3.1GB。訓練時顯存吃緊，用這個。
      注意 accelerate 的 offload hook 預設會在前向把整張表搬上 GPU（且 fp32＝10.5GB）
      而 OOM，故需移除該 hook 並改寫 forward。
    - `ple_on_gpu=True`（推論加速）：**整個模型放 GPU（device_map={"":0}），
      完全不用 accelerate offload**。

      ⚠️ 2026-08-06 實測，這是本專案最重要的效能發現：
      瓶頸不是 PLE 放哪裡，而是 **只要 device_map 內有任何 "cpu" 項目，
      accelerate 就會為整個模型掛上 offload hook，使每個 token 慢到約 35 秒**。
      對照數據（同一台機、同一模型）：

        device_map 含 cpu 項目（PLE 在 CPU）→ 36.0 秒/token
        device_map 含 cpu 項目（PLE 移到 GPU）→ 35.0 秒/token（幾乎無改善）
        device_map={"":0} 全部放 GPU        →  0.06 秒/token（快約 580 倍）

      另：載入時間也從約 2 分鐘降到 7 秒。GPU 需約 9GB（含 PLE 5.25GB、
      層 3.1GB、視覺/音訊塔 0.89GB），僅在 `can_fit_ple_on_gpu()` 為真時使用。
    """
    from transformers import Gemma4ForConditionalGeneration
    from accelerate.hooks import remove_hook_from_module

    if ple_on_gpu:
        device_map = {"": 0}          # 全部放 GPU：唯一能避開 offload hook 的做法
    else:
        device_map = dict(GEMMA4_OFFLOAD_MAP)

    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_id, quantization_config=bnb_config,
        dtype=torch.bfloat16, device_map=device_map)

    if not ple_on_gpu:
        ple = model.model.language_model.embed_tokens_per_layer
        remove_hook_from_module(ple, recurse=True)  # 拿掉會把表搬上 GPU 的 hook
        ple.to("cpu")
        _orig_forward = ple.forward

        def cpu_lookup(input_ids, *a, **k):
            dev = input_ids.device
            return _orig_forward(input_ids.to("cpu"), *a, **k).to(dev)

        ple.forward = cpu_lookup
    return model


def build_dataset(name, tokenizer, max_len, target="gloss"):
    """把資料轉成 input_ids + labels（prompt 段遮罩為 -100）。

    target="gloss"：讀 data/splits/，目標為「我/台北/住」純 Gloss。
    target="json" ：讀 data/splits_json/（scripts/build_json_targets.py 產出），
                    目標為含 gloss/question_type/negation/nonmanual 的結構化 JSON，
                    讓下游虛擬人可直接取用非手部標記（計畫第 1 節）。
    """
    if target == "json":
        rows = [json.loads(l) for l in (BASE / "data" / "splits_json" / f"{name}.jsonl")
                .read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = [{"chinese": r["input"], "gloss_text": r["output"],
                 "context": r.get("context", "")} for r in rows]
    else:
        rows = [json.loads(l) for l in (BASE / "data" / "splits" / f"{name}.jsonl")
                .read_text(encoding="utf-8").splitlines() if l.strip()]

    def ids(text):
        # 模板文字已含 <bos> 等特殊標記，故 add_special_tokens=False
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    def encode(r):
        ctx = r.get("context", "")
        prompt_text = tokenizer.apply_chat_template(
            pc.build_messages(r["chinese"], context=ctx), add_generation_prompt=True,
            tokenize=False)
        full_text = tokenizer.apply_chat_template(
            pc.build_messages(r["chinese"], r["gloss_text"], context=ctx),
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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target", choices=["gloss", "json"], default="gloss",
                    help="訓練目標格式：gloss=純 Gloss；json=結構化（含 NMS）")
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

    train_ds = build_dataset("train", tokenizer, args.max_len, args.target)
    dev_ds = build_dataset("dev", tokenizer, args.max_len, args.target)
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
        save_total_limit=None,      # 留下每個 epoch 的 checkpoint，供事後依 dev loss 挑最佳
        # 不用 load_best_model_at_end：在 PLE-offloaded model 上 reload adapter 會觸發
        # accelerate dispatch_model 報錯；改為訓練後由 trainer_state 挑最佳 checkpoint 再評估。
        load_best_model_at_end=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=args.seed,
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
