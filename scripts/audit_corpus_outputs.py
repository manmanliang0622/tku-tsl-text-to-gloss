#!/usr/bin/env python3
"""自動用語料庫句子測模型，並分類錯誤型態，找出改進方向。

前提（必須據實理解結果）：文化部語料庫已全數進入訓練集（2026-08-05 政策），
故本測試量的是「模型能否復現學過的內容」，**不是泛化能力**。即便如此，
連訓練資料都復現不了就代表模型欠學習或資料訊號被稀釋，仍是有效的診斷。

錯誤分類（依 CCL24-Eval 的分類法擴充）：
  完全正確 / 詞彙替換 / Gloss 缺失 / Gloss 多增 / 語序錯誤（詞集相同）/ 混合

另量三項本專案關切的系統性問題（皆來自使用者實測回報）：
  - 情態詞遺漏：ref 有「要／能／可以」而 pred 沒有
  - 否定詞形不符：ref 用「沒辦法／禁止」而 pred 用「沒有／不能」等
  - 長度不足：pred 明顯短於 ref（模型傾向少生成）

用法（VM，服務需已啟動）：
  python3 scripts/audit_corpus_outputs.py --n 60 --tag v9_corpus_audit
"""
import argparse
import json
import random
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
import metrics  # noqa: E402

API = "http://127.0.0.1:8018/translate"
MODALS = {"要", "想", "能", "可以", "會", "必須", "需要"}
NEGATIONS = {"不", "沒有", "沒", "沒辦法", "不能", "禁止", "不行", "無法", "不要"}


def ask(text, timeout=300):
    body = json.dumps({"text": text}).encode()
    req = urllib.request.Request(API, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def classify(ref, hyp):
    if ref == hyp:
        return "完全正確"
    if not hyp:
        return "空輸出"
    if sorted(ref) == sorted(hyp):
        return "語序錯誤"
    sr, sh = set(ref), set(hyp)
    if sh < sr:
        return "Gloss缺失"
    if sr < sh:
        return "Gloss多增"
    return "詞彙替換/混合"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="抽樣句數")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="corpus_audit")
    ap.add_argument("--min-len", type=int, default=4,
                    help="只測 Gloss 長度 >= 此值的句子（短句已知表現較好）")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (BASE / "data/tslcorpus/parallel.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    pool = [r for r in rows if len(r["gloss"]) >= args.min_len]
    random.Random(args.seed).shuffle(pool)
    sample = pool[:args.n]
    print(f"語料庫 {len(rows)} 句，長度 >= {args.min_len} 者 {len(pool)} 句，抽 {len(sample)} 句\n")

    recs = []
    for i, r in enumerate(sample):
        try:
            d = ask(r["chinese"])
            pred = d.get("gloss_text", "")
        except Exception as e:
            pred = ""
            d = {"error": str(e)}
        ref_toks = metrics.tokenize(r["gloss_text"])
        hyp_toks = metrics.tokenize(pred)
        cat = classify(ref_toks, hyp_toks)
        recs.append({
            "id": r["id"], "chinese": r["chinese"],
            "ref": r["gloss_text"], "pred": pred, "category": cat,
            "ref_len": len(ref_toks), "pred_len": len(hyp_toks),
            "modal_dropped": bool((set(ref_toks) & MODALS) - set(hyp_toks)),
            "neg_mismatch": bool((set(ref_toks) & NEGATIONS)
                                 and (set(ref_toks) & NEGATIONS) != (set(hyp_toks) & NEGATIONS)),
            "seconds": d.get("seconds"),
        })
        print(f"[{i+1}/{len(sample)}] {cat}  {r['chinese'][:26]}", flush=True)

    out = BASE / "results" / f"{args.tag}.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    vocab = json.load((BASE / "data/vocab/eval_vocab.json").open(encoding="utf-8"))
    m = metrics.evaluate([r["ref"] for r in recs], [r["pred"] for r in recs],
                         set(vocab["renderable"]))
    cats = Counter(r["category"] for r in recs)
    n = len(recs)
    summary = {
        "metrics": m,
        "categories": dict(cats.most_common()),
        "modal_dropped_pct": round(sum(r["modal_dropped"] for r in recs) / n * 100, 1),
        "neg_mismatch_pct": round(sum(r["neg_mismatch"] for r in recs) / n * 100, 1),
        "avg_ref_len": round(sum(r["ref_len"] for r in recs) / n, 2),
        "avg_pred_len": round(sum(r["pred_len"] for r in recs) / n, 2),
        "shorter_than_ref_pct": round(sum(1 for r in recs if r["pred_len"] < r["ref_len"]) / n * 100, 1),
        "note": "語料庫句已全在訓練集，本測量的是復現能力而非泛化",
    }
    (BASE / "results" / f"summary_{args.tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== {args.tag}（n={n}）===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
