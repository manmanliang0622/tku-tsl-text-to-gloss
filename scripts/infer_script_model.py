#!/usr/bin/env python3
"""用訓練好的 adapter 對「手語腳本」格式（tsl-script-v1）產生預測。

為什麼要另寫一支：`eval_json_model.py` 是走 `prompt_common` 組舊格式 prompt 的，
新格式的 messages 本身就是完整的 system/user/assistant，**不能也不該再包一層**。

**prompt 逐字一致**是這支腳本的核心設計：它直接取資料檔裡的 messages[0]（system）
與 messages[1]（user），原封不動套 chat template，不自行組字串。既有教訓——
2026-08-20 因為自組 prompt 與評估腳本不同，33 句 test 的 EM 與 ValidJSON 全部
掛零，看起來像模型壞掉，其實模型是好的。把 prompt 的來源綁死在資料檔就不會再犯。

輸出每行含 `eval_script_format.py` 需要的欄位：
    id / raw（模型原始輸出）/ candidate_ids / ref_sign_ids / ref_needs_review

用法：
    python3 scripts/infer_script_model.py \\
        --adapter ~/outputs/qlora_e4b_v14script/checkpoint-558 \\
        --split test --tag v14script
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "splits_script"
RESULTS = BASE / "results"


def candidate_ids(user_content: str) -> list[str]:
    """取出該題的候選 ID。相容兩種序列化：'TSL_今天'（語義 ID）與 'TSL_01084=今天'。"""
    u = json.loads(user_content)
    out = []
    for c in u.get("candidates", []):
        if isinstance(c, dict):
            out.append(str(c.get("sign_id")))
        else:
            out.append(str(c).split("=", 1)[0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path.home() / "0813/model_service/base_model"))
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-new", type=int, default=256,
                    help="腳本目標比舊 JSON 長（sign_ids 陣列），預設放寬到 256")
    ap.add_argument("--max-len-ctx", type=int, default=768,
                    help="載入時的 max_seq_length，需 >= 訓練時的值")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src = DATA / f"{args.split}.jsonl"
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    import torch
    # 用 Unsloth 的 FastModel 載入，與 train_unsloth.py 同一條路徑。
    # 不要改用 transformers+PeftModel：unsloth venv 裡的 torchao 是 0.12.0，
    # peft 的 torchao 分派器要求 >0.16.0，會直接 ImportError（2026-08-21 實測）。
    # 載入方式與訓練一致本來就比較安全，順便繞開這個版本衝突。
    from unsloth import FastModel

    model, tok = FastModel.from_pretrained(
        model_name=args.adapter,          # adapter 目錄，Unsloth 會自己接回 base
        max_seq_length=args.max_len_ctx,
        load_in_4bit=True,
        full_finetuning=False,
    )
    FastModel.for_inference(model)
    # processor 與 tokenizer 的屬性位置不同，兩種都試
    _tk = getattr(tok, "tokenizer", tok)
    pad_id = _tk.pad_token_id if _tk.pad_token_id is not None else _tk.eos_token_id

    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / f"{args.tag}_{args.split}.jsonl"
    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            msgs = r["messages"]
            # 原封不動用資料檔裡的 system+user，不自行組 prompt
            prompt = tok.apply_chat_template(msgs[:2], tokenize=False,
                                             add_generation_prompt=True)
            # 必須指名 text=：Gemma 4 的是多模態 processor，第一個位置參數是
            # images，直接傳字串會被當成圖片而 text 變成 None（TypeError）。
            enc = tok(text=prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new,
                                     do_sample=False, pad_token_id=pad_id)
            raw = tok.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            ref = json.loads(msgs[2]["content"])
            f.write(json.dumps({
                "id": r.get("metadata", {}).get("id"),
                "raw": raw,
                "candidate_ids": candidate_ids(msgs[1]["content"]),
                "ref_sign_ids": ref.get("sign_ids") or [],
                "ref_needs_review": bool(ref.get("needs_review", False)),
            }, ensure_ascii=False) + "\n")
            if i % 20 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
    print(f"寫出 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
