#!/usr/bin/env python3
"""tsl-script-v1 格式的 QLoRA 微調（unsloth + TRL SFTTrainer）。

**這是重建版，不是原件。** 訓練出 v14–v17 的那支腳本在任何地方都找不到——
不在工作目錄、不在 git 的任何分支（6 個歷史版本全部沒有 script 格式）、
不在四份 __pycache__ 的 .pyc 裡。`model_service/scripts/train_qlora.py` 是更早的
transformers 版本，只支援 --target {gloss,json}，且用 device_map 把 PLE offload 到
CPU（GPU 常駐約 3.1GB）——與 v17 實測 peak 11.8GB 的 unsloth 全 GPU 路線不同。

重建依據（全部來自 v17 留下的紀錄，不是猜的）:
  * outputs/qlora_e4b_v17script_k40sem/unsloth_run.json 的 args 區塊
    → 參數名稱與預設值（format/data/lora_dropout/nr_loss_weight/...）
  * ~/v17_train.log 的 [train] 訊息
    → 每一行輸出格式、max_len 自動提升、train_on_responses_only、遮罩檢查
  * outputs/.../README.md → TRL 0.24.0 + SFT
  * 步數核對：8915 句 ÷ (batch 2 × grad_accum 8) = 557.2 → 558 步/epoch，
    與 checkpoint-558 / checkpoint-1116 相符。

**已驗證忠實（2026-08-27）**：用 --verify-v17 以 v17 的原始設定完整重訓一次，
結果全部落在 1.2% 以內，量級與本專案已知的 bf16 跨行程非決定性相同：

    指標             重建版      v17      相對差
    eval_loss ep1    0.1923    0.1910    +0.69%
    eval_loss ep2    0.2282    0.2279    +0.13%
    train_loss       0.1122    0.1109    +1.17%
    wall_min         157.8     157.0     +0.54%
    peak_alloc_gb    10.99     11.80     -6.86%

BEST_EPOCH 兩邊同為 epoch 1；adapter_config 逐項相同；adapter 權重檔大小
完全一致（146,888,168 bytes）。產物留在 outputs/qlora_e4b_v17script_k40sem_rebuild。

改動這支腳本後請重跑一次 --verify-v17，它會自動核對可訓練參數量（見
TARGET_MODULES 的註解——第一次重建就是栽在這裡）。

用法：
  # 重建 v17（驗證用）
  python3 scripts/train_script_qlora.py --verify-v17

  # v18：無語義通道的可重現基準，epoch 1 即最佳所以只跑 1
  python3 scripts/train_script_qlora.py \\
      --data data/splits_script_v18 --output ~/outputs/qlora_e4b_v18script \\
      --epochs 1

GPU：v17 實測 peak 11.8GB。本機 RTX 4060 Ti 16GB 另有線上服務佔用約 9.4GB，
**必須先停掉服務**，否則會 OOM。
"""
import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_BASE = "/home/b310ai/0813/model_service/base_model"

# Gemma 4 canonical chat template 的段落標記（實測 apply_chat_template 的輸出）。
# 只在 model 段算 loss：prompt 含 40 個候選 ID，不遮罩的話會淹沒學習訊號。
INSTRUCTION_PART = "<|turn>user\n"
RESPONSE_PART = "<|turn>model\n"

# **逐字抄自 v17 的 checkpoint-558/adapter_config.json**，不是自己寫的。
# 這是 unsloth 產生的正規表示式，只打在 language/text 相關模組上。
# 第一次重建時用了明確清單 ["q_proj", "k_proj", ...]，結果連視覺塔與音訊塔
# 一起打進去：可訓練參數 42,401,792 (0.53%) vs v17 的 36,700,160 (0.46%)，
# 多出 5,701,632 個。差異剛好等於 base 總數的差，證實就是多掛的那些 LoRA。
# 這會讓 --verify-v17 失去意義（對不上時分不清是重建問題還是模組範圍不同），
# 也會產生與線上 adapter 結構不同的權重。改動前先確認 v17 的 adapter_config。
TARGET_MODULES = (
    r"(?:.*?(?:language|text).*?(?:self_attn|attention|attn|mixer|mlp|feed_forward|ffn|dense|mixer)"
    r".*?(?:k_proj|q_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|per_layer_input_gate"
    r"|per_layer_projection|linear|embedding_projection|relative_k_proj))"
    r"|(?:\bmodel\.layers\.[\d]{1,}\.(?:self_attn|attention|attn|mixer|mlp|feed_forward|ffn|dense|mixer)"
    r"\.(?:(?:k_proj|q_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|per_layer_input_gate"
    r"|per_layer_projection|linear|embedding_projection|relative_k_proj)))"
)
V17_TRAINABLE = 36_700_160      # v17 的可訓練參數量，--verify-v17 會核對

# v17 的實際設定，--verify-v17 用來重現
V17 = dict(data=str(REPO / "data" / "splits_script_k40sem"),
           output="/home/b310ai/outputs/qlora_e4b_v17script_k40sem_rebuild",
           epochs=2, lr=2e-4, batch=2, grad_accum=8, max_len=768,
           lora_r=16, lora_alpha=32, lora_dropout=0.05, nr_loss_weight=1.0, seed=42)


def log(msg):
    print(f"[train] {msg}", flush=True)


def load_split(data_dir: Path, name: str):
    """讀 build_script_dataset.py 產出的 messages 格式。"""
    path = data_dir / f"{name}.jsonl"
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append({"messages": json.loads(line)["messages"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_BASE)
    ap.add_argument("--data", default=str(REPO / "data" / "splits_script_v18"),
                    help="build_script_dataset.py 的輸出目錄")
    ap.add_argument("--format", choices=["script"], default="script")
    ap.add_argument("--output", default="/home/b310ai/outputs/qlora_e4b_v18script")
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--nr-loss-weight", type=float, default=1.0,
                    help="needs_review token 的 loss 權重；1.0＝不加權（v15 試過加權，已判死）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=-1, help=">0 用於冒煙測試")
    ap.add_argument("--verify-v17", action="store_true",
                    help="用 v17 的原始設定重訓，驗證這支重建版是否忠實")
    args = ap.parse_args()

    if args.verify_v17:
        for k, v in V17.items():
            setattr(args, k, v)
        log("--verify-v17：套用 v17 原始設定，預期 eval_loss ep1≈0.1910 ep2≈0.2279")

    if args.format == "script" and args.max_len < 768:
        log(f"--format script：max_len {args.max_len} → 768（候選清單需要）")
        args.max_len = 768

    if args.nr_loss_weight != 1.0:
        raise SystemExit("nr_loss_weight != 1.0 的加權路徑未重建（v15 實測已判死，"
                         "原件也沒留下）。要用請先自行實作。")

    log(json.dumps({k: v for k, v in vars(args).items()
                    if k not in ("verify_v17", "max_steps")}, ensure_ascii=False))

    # unsloth 必須在 transformers 之前 import（它會 patch）
    from unsloth import FastModel
    from unsloth.chat_templates import train_on_responses_only
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    import torch

    t0 = time.time()
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_len,
        load_in_4bit=True,
        full_finetuning=False,
    )
    log(f"model loaded in {time.time() - t0:.1f}s")

    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=TARGET_MODULES,
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"可訓練參數 {trainable:,}（v17 為 {V17_TRAINABLE:,}）")
    if args.verify_v17 and trainable != V17_TRAINABLE:
        raise SystemExit(
            f"可訓練參數 {trainable:,} != v17 的 {V17_TRAINABLE:,}，"
            f"差 {trainable - V17_TRAINABLE:+,}。target_modules 與 v17 不同，"
            f"對照 eval_loss 會失去意義。先核對 v17 的 adapter_config.json。")

    data_dir = Path(args.data)
    train_rows = load_split(data_dir, "train")
    dev_rows = load_split(data_dir, "dev")
    log(f"train: {len(train_rows)} examples")
    log(f"dev: {len(dev_rows)} examples")

    def to_text(rows):
        return Dataset.from_list([
            {"text": tokenizer.apply_chat_template(r["messages"], tokenize=False)}
            for r in rows])

    train_ds, dev_ds = to_text(train_rows), to_text(dev_rows)

    smoke = args.max_steps and args.max_steps > 0
    cfg = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=25,
        eval_strategy="no" if smoke else "epoch",
        save_strategy="no" if smoke else "epoch",
        save_total_limit=None,      # 留下每個 epoch，事後依 dev loss 挑最佳
        load_best_model_at_end=False,
        bf16=True,
        max_length=args.max_len,
        dataset_text_field="text",
        report_to="none",
        seed=args.seed,
    )
    trainer = SFTTrainer(model=model, args=cfg,
                         train_dataset=train_ds, eval_dataset=dev_ds)

    # 只在 model 段算 loss。prompt 帶 40 個候選 ID，不遮罩會淹沒學習訊號。
    trainer = train_on_responses_only(trainer,
                                      instruction_part=INSTRUCTION_PART,
                                      response_part=RESPONSE_PART)
    log("train_on_responses_only 已套用（Gemma 4 <|turn> 標記）")

    # 遮罩檢查：v17 的紀錄是 92/437 token 參與 loss（21.1%）。
    # 這個比例若明顯不同，代表模板或標記對不上，訓練會學錯東西——寧可現在就發現。
    sample = trainer.train_dataset[0]
    labels = sample["labels"]
    kept = sum(1 for x in labels if x != -100)
    log(f"實際學習內容：{tokenizer.decode([x for x in labels if x != -100])!r}")
    log(f"遮罩檢查：{kept}/{len(labels)} token 參與 loss（{100 * kept / len(labels):.1f}%）")
    if not 0.05 < kept / len(labels) < 0.60:
        raise SystemExit(f"遮罩比例 {100 * kept / len(labels):.1f}% 不合理，"
                         f"對照 v17 的 21.1%。檢查 chat template 的 turn 標記是否改變。")

    result = trainer.train()

    peak = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0
    wall = (time.time() - t0) / 60
    print(f"TRAIN_RESULT wall={wall:.1f}min peak_alloc={peak:.2f}GB "
          f"train_loss={result.training_loss:.4f}", flush=True)

    evals = [(h["epoch"], h["eval_loss"])
             for h in trainer.state.log_history if "eval_loss" in h]
    for ep, loss in evals:
        print(f"EVAL epoch={ep} eval_loss={loss:.4f}", flush=True)
    if evals:
        best = min(evals, key=lambda x: x[1])
        print(f"BEST_EPOCH epoch={best[0]} eval_loss={best[1]:.4f}", flush=True)

    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "unsloth_run.json").write_text(json.dumps({
        "args": {k: v for k, v in vars(args).items()
                 if k not in ("verify_v17", "max_steps")},
        "wall_min": wall,
        "peak_alloc_gb": peak,
        "train_loss": result.training_loss,
        "evals": [list(e) for e in evals],
        "log_history": trainer.state.log_history,
        "_rebuilt_trainer": "scripts/train_script_qlora.py（2026-08-27 重建，非原件）",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"checkpoints in {args.output}")
    log("DONE")


if __name__ == "__main__":
    main()
