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


def needs_review_prob(tok, seq, scores):
    """回傳模型在 needs_review 那個位置給 true 的機率（相對 false 正規化）。

    為什麼要這個（2026-08-21）：貪婪解碼只給硬 true/false，而 v14 在 test_corpus
    的表現是 precision 0.941／recall 0.118——它一開口幾乎都對，就是太少開口。
    這是典型的門檻過保守，用機率＋在 **dev** 上調門檻即可校準，不必重訓。
    ⚠️ 門檻只能在 dev 上調，拿 test_corpus 調就是對測試集調參。

    只在 true/false 兩個結果之間正規化，不看其他 token——那些與這個決策無關，
    納入只會讓機率被序列長度稀釋。找不到該位置時回 None（不猜）。
    """
    import torch
    text = ""
    for i, tid in enumerate(seq.tolist()):
        piece = tok.decode([tid])
        low = piece.strip().lower()
        if "needs_review" in text and low[:4] in ("true", "fals"):
            if i >= len(scores):
                return None
            probs = torch.softmax(scores[i][0].float(), dim=-1)
            top = torch.topk(probs, 50)
            pt = pf = 0.0
            for val, idx in zip(top.values.tolist(), top.indices.tolist()):
                w = tok.decode([idx]).strip().lower()
                if w.startswith("true"):
                    pt += val
                elif w.startswith("fals"):
                    pf += val
            return round(pt / (pt + pf), 6) if (pt + pf) > 0 else None
        text += piece
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path.home() / "0813/model_service/base_model"))
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--split", nargs="+", default=["test"],
                    help="可一次給多個切分：同一程序內共用已載入的模型，"
                         "切分之間不釋放 GPU——2026-08-25 教訓：逐切分開新程序"
                         "會在模型重載的空窗被看門狗搶走 GPU，之後整批掉進"
                         "CPU offload 慢 70 倍")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default=str(DATA),
                    help="切分資料夾。⚠️ 必須與訓練用的同一份（候選 k 不同就是"
                         "不同的任務分布），v15 起用 data/splits_script_k60")
    ap.add_argument("--max-new", type=int, default=256,
                    help="腳本目標比舊 JSON 長（sign_ids 陣列），預設放寬到 256")
    ap.add_argument("--max-len-ctx", type=int, default=768,
                    help="載入時的 max_seq_length，需 >= 訓練時的值")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

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
    for split in args.split:
        run_split(split, args, model, tok, pad_id)
    return 0


def run_split(split, args, model, tok, pad_id):
    import torch
    src = Path(args.data) / f"{split}.jsonl"
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    out_path = RESULTS / f"{args.tag}_{split}.jsonl"
    print(f"=== {split} ===", flush=True)
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
                                     do_sample=False, pad_token_id=pad_id,
                                     return_dict_in_generate=True, output_scores=True)
            seq = gen.sequences[0][enc["input_ids"].shape[1]:]
            raw = tok.decode(seq, skip_special_tokens=True)
            p_nr = needs_review_prob(tok, seq, gen.scores)
            ref = json.loads(msgs[2]["content"])
            f.write(json.dumps({
                "id": r.get("metadata", {}).get("id"),
                "raw": raw,
                "candidate_ids": candidate_ids(msgs[1]["content"]),
                "ref_sign_ids": ref.get("sign_ids") or [],
                "ref_needs_review": bool(ref.get("needs_review", False)),
                # needs_review 那個位置的機率，供事後調門檻（見 needs_review_prob）
                "p_needs_review": p_nr,
            }, ensure_ascii=False) + "\n")
            if i % 20 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
    print(f"寫出 {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
