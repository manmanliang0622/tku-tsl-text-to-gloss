#!/usr/bin/env python3
"""用 Unsloth 訓練 Gemma 4 E4B QLoRA（v12u），超參數對齊 v8/v11 以求可比。

為什麼要這一輪（2026-08-20，教授要求評估 Unsloth）：
  smoke test 已證明能跑、峰值 10.87GB。但「能跑」不等於「品質相同」，
  要換掉自製 train_qlora.py 之前，必須在**同一份切分、同一組超參數**下
  訓出模型並用同一支 eval 比分，確認 Unsloth 沒有讓品質退步。

對齊 v8 的設定（取自 scripts/train_qlora.py 的 argparse 預設）：
  lr 2e-4 / batch 2 / grad_accum 8 / max_len 512 / LoRA r16 alpha32 dropout0.05
  / cosine scheduler / warmup_ratio 0.03 / seed 42 / 每 epoch 存檔與評估

差異（Unsloth 無法完全對齊處，報告需註明）：
  - optimizer 用 adamw_8bit（Unsloth 建議值；v8 用 HF 預設 adamw_torch）
  - LoRA target 由 Unsloth 的 finetune_* 旗標決定，非 v8 的手寫 regex
  - 遮罩改用 Unsloth train_on_responses_only，取代 v8 自製 MaskedCollator

環境（VM 上的獨立 venv，不可混用既有 .venv）：
  python3.10 -m venv --system-site-packages ~/unsloth-venv
  ~/unsloth-venv/bin/pip install unsloth "pillow>=10" "torchao==0.12.0"
  # pillow：系統版太舊缺 Image.Resampling
  # torchao：unsloth 帶的 0.18 需要 torch>=2.8，本機是 2.7.1

執行（訓練需 ~10.9GB 顯存，須先停 v11 服務）：
  setsid nohup ~/unsloth-venv/bin/python ~/train_unsloth.py > ~/unsloth_train.log 2>&1 &
"""
import argparse
import inspect
import json
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(Path.home() / "0813/model_service/base_model"))
ap.add_argument("--data", default=str(Path.home() / "tku-tsl-text-to-gloss/data/splits_json"))
ap.add_argument("--output", default=str(Path.home() / "outputs/qlora_e4b_v12u"))
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--lr", type=float, default=2e-4)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--grad-accum", type=int, default=8)
ap.add_argument("--max-len", type=int, default=512)
ap.add_argument("--lora-r", type=int, default=16)
ap.add_argument("--lora-alpha", type=int, default=32)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

print(f"[train] {json.dumps(vars(args), ensure_ascii=False)}", flush=True)

from unsloth import FastModel  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

t0 = time.time()
model, tokenizer = FastModel.from_pretrained(
    model_name=args.model,
    max_seq_length=args.max_len,
    load_in_4bit=True,
    full_finetuning=False,
)
print(f"[train] model loaded in {time.time()-t0:.1f}s", flush=True)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=0.05,
    bias="none",
    random_state=args.seed,
)


def load_split(name):
    rows = []
    with open(Path(args.data) / f"{name}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            user = r["instruction"] + "\n" + r.get("input", "")
            msgs = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": r["output"]},
            ]
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            if not isinstance(text, str):
                text = str(text)
            if text.startswith("<bos>"):
                text = text[len("<bos>"):]
            rows.append({"text": text})
    print(f"[train] {name}: {len(rows)} examples", flush=True)
    return Dataset.from_list(rows)


train_ds = load_split("train")
dev_ds = load_split("dev")

cfg_want = dict(
    dataset_text_field="text",
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=args.batch,
    gradient_accumulation_steps=args.grad_accum,
    num_train_epochs=args.epochs,
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=25,
    output_dir=args.output,
    optim="adamw_8bit",
    seed=args.seed,
    report_to="none",
    max_length=args.max_len,
    max_seq_length=args.max_len,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=None,
    load_best_model_at_end=False,  # v4 踩過：offloaded model reload 會拋 dispatch 錯
    bf16=True,
)
sig = set(inspect.signature(SFTConfig.__init__).parameters)
dropped = sorted(set(cfg_want) - sig)
if dropped:
    print(f"[train] SFTConfig 不支援而略過: {dropped}", flush=True)
cfg = SFTConfig(**{k: v for k, v in cfg_want.items() if k in sig})

tk = dict(model=model, train_dataset=train_ds, eval_dataset=dev_ds, args=cfg)
tsig = set(inspect.signature(SFTTrainer.__init__).parameters)
tk["processing_class" if "processing_class" in tsig else "tokenizer"] = tokenizer
trainer = SFTTrainer(**tk)

# 只對 assistant 段算 loss（等價於 v8 的 MaskedCollator）
# ⚠️ 標記是 Gemma 4 的 `<|turn>`，不是 Gemma 3 的 `<start_of_turn>`。
# 用錯不會報錯，只會整批遮成 -100 然後退回全序列 loss（與 v8 不等價）。
# 實測模板輸出：'<bos><|turn>user\nUUU<turn|>\n<|turn>model\nAAA<turn|>\n'
from unsloth.chat_templates import train_on_responses_only  # noqa: E402

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|turn>user\n",
    response_part="<|turn>model\n",
)
print("[train] train_on_responses_only 已套用（Gemma 4 <|turn> 標記）", flush=True)

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
stats = trainer.train()
wall = time.time() - t0
print(
    f"TRAIN_RESULT wall={wall/60:.1f}min peak_alloc={torch.cuda.max_memory_allocated()/1e9:.2f}GB "
    f"train_loss={stats.training_loss:.4f}",
    flush=True,
)

hist = trainer.state.log_history
evals = [(h.get("epoch"), h["eval_loss"]) for h in hist if "eval_loss" in h]
for ep, loss in evals:
    print(f"EVAL epoch={ep} eval_loss={loss:.4f}", flush=True)
if evals:
    best = min(evals, key=lambda x: x[1])
    print(f"BEST_EPOCH epoch={best[0]} eval_loss={best[1]:.4f}", flush=True)

with open(Path(args.output) / "unsloth_run.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "args": vars(args),
            "wall_min": wall / 60,
            "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
            "train_loss": stats.training_loss,
            "evals": evals,
            "log_history": hist,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )
print(f"[train] checkpoints in {args.output}", flush=True)
print("[train] DONE", flush=True)
