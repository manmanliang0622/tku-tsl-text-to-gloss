#!/usr/bin/env python3
"""評估「手語腳本」格式（tsl-script-v1）的模型輸出。

為什麼要另寫一支（2026-08-20）：舊指標量的是「生成的 gloss 對不對」，
新格式的任務變成「從候選清單挑 sign_id」，多出三個舊指標答不了的問題：

  1. **約束違反率** — 模型有沒有生出候選清單以外的 ID？這是新格式存在的理由，
     理想值 0%。舊格式的「詞彙表內率」在此功成身退：候選本來就只從
     有影片的手語裡挑，合法性由清單保證，不必再事後比對詞表。
  2. **needs_review 的 precision/recall** — 該拒答有沒有拒、不該拒有沒有亂拒。
     只看整體準確率會被多數類洗掉（拒答率可能高達五成），必須分開看。
  3. **可播放率** — 輸出的 ID 是否都對得到實際影片。理論上恆為 100%
     （ID 來自動作庫），但模型若幻覺出格式正確卻不存在的 ID 就會破功，
     這是必須實測而非假設的安全網。

Gloss 層的 BLEU/ROUGE/EM 仍沿用 metrics.py，把 sign_id 序列轉回 gloss 後計算，
才能與 v8/v11 等舊模型放在同一張表上比較。

用法：
    python3 scripts/eval_script_format.py --pred results/xxx_test.jsonl
    # pred 檔每行需有 candidates（該題的候選 ID）、ref_sign_ids、pred_raw

輸出：console 報表 + 同名 _scriptmetrics.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
INVENTORY = BASE / "data" / "signs" / "sign_inventory.jsonl"


def load_inventory() -> dict[str, str]:
    """sign_id → gloss。用來把 ID 序列轉回 gloss 以沿用既有指標。"""
    out = {}
    with INVENTORY.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["sign_id"]] = r["gloss"]
    return out


def parse_output(raw: str) -> dict | None:
    """從模型輸出取出 JSON；取不到回 None（記為無效輸出）。"""
    m = re.search(r"\{.*\}", str(raw), re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def evaluate(rows: list[dict]) -> dict:
    id2gloss = load_inventory()

    valid = 0
    viol_rows = 0          # 至少違反一次的題數
    viol_tokens = 0        # 違反的 ID 個數
    total_pred_ids = 0
    unplayable = 0         # 格式像 ID 但總表查無此 ID
    ref_gloss, hyp_gloss = [], []
    groups = []

    # needs_review 混淆矩陣
    tp = fp = tn = fn = 0

    for r in rows:
        obj = parse_output(r.get("raw", ""))
        cands = set(r.get("candidate_ids") or [])
        ref_ids = list(r.get("ref_sign_ids") or [])
        ref_nr = bool(r.get("ref_needs_review", False))

        if obj is None:
            # 無效輸出：Gloss 記空字串（計為全錯），needs_review 視為未預測
            ref_gloss.append("/".join(id2gloss.get(i, i) for i in ref_ids))
            hyp_gloss.append("")
            groups.append(r.get("id"))
            if ref_nr:
                fn += 1
            else:
                tn += 1
            continue

        valid += 1
        pred_ids = [str(x) for x in (obj.get("sign_ids") or [])]
        pred_nr = bool(obj.get("needs_review", False))

        total_pred_ids += len(pred_ids)
        bad = [i for i in pred_ids if cands and i not in cands]
        viol_tokens += len(bad)
        viol_rows += bool(bad)
        unplayable += sum(1 for i in pred_ids if i not in id2gloss)

        ref_gloss.append("/".join(id2gloss.get(i, i) for i in ref_ids))
        hyp_gloss.append("/".join(id2gloss.get(i, i) for i in pred_ids))
        groups.append(r.get("id"))

        if ref_nr and pred_nr:
            tp += 1
        elif not ref_nr and pred_nr:
            fp += 1
        elif not ref_nr and not pred_nr:
            tn += 1
        else:
            fn += 1

    n = len(rows)
    vocab = metrics.load_eval_vocab() if hasattr(metrics, "load_eval_vocab") else set()
    try:
        base = metrics.evaluate(ref_gloss, hyp_gloss, vocab, groups=groups)
    except Exception as e:  # noqa: BLE001  指標細節不該擋住主報表
        base = {"metrics_error": str(e)}

    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None

    # 舊的「詞彙表內率」在新格式下已由候選清單取代（候選只從有影片的手語裡出），
    # 留著會顯示 0.0% 誤導成模型全錯。改由 Playable% 承擔這個角色。
    out = {k: v for k, v in base.items() if not k.startswith("InVocab")}
    out.update({
        "n": n,
        "ValidJSON%": round(100 * valid / n, 2) if n else None,
        # 新格式的核心指標
        "ConstraintViolation%_rows": round(100 * viol_rows / n, 2) if n else None,
        "ConstraintViolation%_ids": (round(100 * viol_tokens / total_pred_ids, 2)
                                     if total_pred_ids else None),
        "UnknownSignID": unplayable,
        "Playable%": (round(100 * (total_pred_ids - unplayable) / total_pred_ids, 2)
                      if total_pred_ids else None),
        "NeedsReview": {
            "precision": round(prec, 4) if prec is not None else None,
            "recall": round(rec, 4) if rec is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "ref_positive_rate": round((tp + fn) / n, 4) if n else None,
        },
    })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True,
                    help="逐句預測 jsonl（需含 candidate_ids / ref_sign_ids / raw）")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            args.pred.read_text(encoding="utf-8").splitlines() if l.strip()]
    res = evaluate(rows)
    print(json.dumps(res, ensure_ascii=False, indent=2))

    out = args.out or args.pred.with_name(args.pred.stem + "_scriptmetrics.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n寫出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
