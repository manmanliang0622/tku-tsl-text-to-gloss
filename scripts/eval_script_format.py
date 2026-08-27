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

2026-08-27 補回兩塊當初改在 VM 上、沒有 commit 的邏輯。v14–v17 的
*_scriptmetrics.json 都含這兩塊，但腳本只 commit 過 8/20 那一版
（commit 5362e3d，在未合併的 origin/video-coverage-eval 上），
導致那些數字一度無法從頭重現：

  * **Full_reference** — 上面那組 BLEU/ROUGE/EM 是對「候選內參考」算的，
    撈不到的詞根本不在參考裡，回答的是「有沒有從候選裡挑對」。Full_reference
    改用 data/splits/<split>.jsonl 的 gloss_text（完整、保留語序的參考），
    才是含檢索缺口的系統整體水準。兩個口徑差很多——2026-08-27 的診斷顯示
    corpus 有 30.2% 的參考詞從未進入候選，所以引用單一數字時務必標明口徑。
  * **NeedsReview_calibrated** — 讀 p_needs_review 配門檻判定，而不是看模型
    自己輸出的硬 true/false（後者 recall 只有 0.15–0.22，幾乎不拒答）。
    門檻選定見 scripts/nr_threshold.py。

回歸測試：tests/test_eval_script_format.py 會拿 results/v17cd_* 重跑，
與既有的 *_scriptmetrics.json 逐欄比對。改動這支腳本後務必跑一次。

用法：
    python3 scripts/eval_script_format.py --pred results/v17cd_dev.jsonl \\
        --threshold 0.039707
    # split 會從檔名推斷（v17cd_dev → data/splits/dev.jsonl），也可 --split 指定
    # 給 --no-full-reference 可退回 8/20 那版的行為

輸出：console 報表 + 同名 _scriptmetrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics  # noqa: E402

BASE = Path(__file__).resolve().parent.parent

# 動作庫不在本 repo（歸 0821 服務管），依序找：環境變數 → 本 repo → 0821。
INVENTORY_CANDIDATES = [
    Path(os.environ["SIGN_INVENTORY"]) if os.environ.get("SIGN_INVENTORY") else None,
    BASE / "data" / "signs" / "sign_inventory.jsonl",
    Path("/home/b310ai/0821/model_service/data/signs/sign_inventory.jsonl"),
]

REF_SCOPE_NOTE = (
    "上方 BLEU-4／ROUGE-L／ExactMatch%% 是對**候選內參考**算的"
    "（撈不到的詞不在參考裡），回答『有沒有從候選裡挑對』；"
    "Full_reference 那組才是含檢索缺口的系統整體水準。"
)


def find_inventory() -> Path:
    for path in INVENTORY_CANDIDATES:
        if path and path.exists():
            return path
    tried = "\n  ".join(str(p) for p in INVENTORY_CANDIDATES if p)
    raise SystemExit(f"找不到 sign_inventory.jsonl，試過：\n  {tried}\n"
                     f"可用 SIGN_INVENTORY=<路徑> 指定。")


def load_inventory() -> dict[str, str]:
    """sign_id → gloss。用來把 ID 序列轉回 gloss 以沿用既有指標。"""
    out = {}
    with find_inventory().open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["sign_id"]] = r["gloss"]
    return out


def guess_split(pred: Path) -> Path | None:
    """v17cd_test_corpus.jsonl → data/splits/test_corpus.jsonl。

    比對已知的 split 名，取最長的匹配（test_corpus 要贏 test）。"""
    stem = pred.stem
    split_dir = BASE / "data" / "splits"
    names = sorted((p.stem for p in split_dir.glob("*.jsonl")), key=len, reverse=True)
    for name in names:
        if stem == name or stem.endswith("_" + name):
            return split_dir / f"{name}.jsonl"
    return None


def load_full_reference(split: Path) -> dict[str, str]:
    """id → 完整參考 gloss_text（保留語序，含候選撈不到的詞）。"""
    out = {}
    with split.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r.get("gloss_text", "")
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


def _prf(tp: int, fp: int, fn: int):
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    return prec, rec, f1


def calibrated_needs_review(rows: list[dict], threshold: float) -> dict:
    """讀 p_needs_review 配門檻；沒有機率的列落回模型自己的 true/false。"""
    tp = fp = tn = fn = 0
    with_prob = fallback = 0
    for r in rows:
        ref_nr = bool(r.get("ref_needs_review", False))
        prob = r.get("p_needs_review")
        if prob is None:
            obj = parse_output(r.get("raw", ""))
            pred_nr = bool(obj.get("needs_review", False)) if obj else False
            fallback += 1
        else:
            pred_nr = prob >= threshold
            with_prob += 1
        if ref_nr and pred_nr:
            tp += 1
        elif not ref_nr and pred_nr:
            fp += 1
        elif not ref_nr and not pred_nr:
            tn += 1
        else:
            fn += 1
    prec, rec, f1 = _prf(tp, fp, fn)
    return {
        "precision": round(prec, 4) if prec is not None else None,
        "recall": round(rec, 4) if rec is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "threshold": threshold,
        "rows_with_prob": with_prob,
        "rows_fallback_to_greedy": fallback,
        "decision": f"p_needs_review >= {threshold}（門檻在 dev 上選定）",
    }


def full_reference_block(rows: list[dict], full_ref: dict[str, str],
                         id2gloss: dict[str, str], vocab: set) -> dict:
    """對完整參考（含候選撈不到的詞）重算一次，這才是系統整體水準。"""
    refs, hyps, groups = [], [], []
    tokens_full = tokens_in_cand = 0
    matched = 0
    for r in rows:
        rid = r.get("id")
        if rid not in full_ref:
            continue
        matched += 1
        gold = full_ref[rid]
        obj = parse_output(r.get("raw", ""))
        pred_ids = [str(x) for x in (obj.get("sign_ids") or [])] if obj else []
        refs.append(gold)
        hyps.append("/".join(id2gloss.get(i, i) for i in pred_ids))
        groups.append(rid)
        tokens_full += len([t for t in gold.split("/") if t])
        tokens_in_cand += len(r.get("ref_sign_ids") or [])

    try:
        base = metrics.evaluate(refs, hyps, vocab, groups=groups)
    except Exception as e:  # noqa: BLE001
        return {"metrics_error": str(e)}

    dropped = tokens_full - tokens_in_cand
    return {
        "BLEU-4": base.get("BLEU-4"),
        "ROUGE-L": base.get("ROUGE-L"),
        "ExactMatch%": base.get("ExactMatch%"),
        "ref_tokens_full": tokens_full,
        "ref_tokens_in_candidates": tokens_in_cand,
        "ref_tokens_dropped%": (round(100 * dropped / tokens_full, 2)
                                if tokens_full else None),
        "rows_matched": matched,
    }


def evaluate(rows: list[dict], threshold: float | None = None,
             full_ref: dict[str, str] | None = None) -> dict:
    id2gloss = load_inventory()

    valid = 0
    viol_rows = 0          # 至少違反一次的題數
    viol_tokens = 0        # 違反的 ID 個數
    total_pred_ids = 0
    unplayable = 0         # 格式像 ID 但總表查無此 ID
    ref_gloss, hyp_gloss = [], []
    groups = []

    # needs_review 混淆矩陣（greedy：模型自己輸出的 true/false）
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

    prec, rec, f1 = _prf(tp, fp, fn)

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
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "ref_positive_rate": round((tp + fn) / n, 4) if n else None,
            "decision": "greedy（模型輸出的 true/false）",
        },
    })

    if threshold is not None:
        out["NeedsReview_calibrated"] = calibrated_needs_review(rows, threshold)

    if full_ref:
        out["_ref_scope"] = REF_SCOPE_NOTE
        out["Full_reference"] = full_reference_block(rows, full_ref, id2gloss, vocab)

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True,
                    help="逐句預測 jsonl（需含 candidate_ids / ref_sign_ids / raw）")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="needs_review 門檻；給了才算 NeedsReview_calibrated")
    ap.add_argument("--split", type=Path, default=None,
                    help="data/splits/<name>.jsonl，用來取完整參考；預設從檔名推斷")
    ap.add_argument("--no-full-reference", action="store_true",
                    help="退回 2026-08-20 那版行為，只算候選內參考")
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            args.pred.read_text(encoding="utf-8").splitlines() if l.strip()]

    full_ref = None
    if not args.no_full_reference:
        split = args.split or guess_split(args.pred)
        if split and split.exists():
            full_ref = load_full_reference(split)
            print(f"[完整參考] {split}", file=sys.stderr)
        else:
            print("[完整參考] 找不到對應的 split，跳過 Full_reference"
                  "（--split 可指定）", file=sys.stderr)

    res = evaluate(rows, threshold=args.threshold, full_ref=full_ref)
    print(json.dumps(res, ensure_ascii=False, indent=2))

    out = args.out or args.pred.with_name(args.pred.stem + "_scriptmetrics.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n寫出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
