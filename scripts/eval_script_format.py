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


# 2026-08-21 在 dev 250 句上選定的門檻。規則事前講定：**recall >= 0.7 下
# 最大化 precision**，選出 0.02（dev P 0.679／R 0.734／F1 0.705）。
# 套到 test_corpus：P 0.952／R 0.728／F1 0.825，對照原本貪婪解碼的
# P 0.941／R 0.118——recall 提升 6.2 倍而 precision 沒有付代價。
#
# ⚠️ 事後診斷發現 test_corpus 上 F1 最佳門檻其實是 0.008（F1 0.921），
# **刻意不採用**：那是拿測試集調參，得到的數字沒有意義。門檻只在 dev 上選。
# 要重選請重跑 dev 推論並沿用同一條規則，不要看 test 的數字調。
NEEDS_REVIEW_THRESHOLD = 0.02


def load_full_refs(path: Path) -> dict[str, str]:
    """讀原始切分的完整參考 Gloss（id → gloss_text）。

    為什麼需要（2026-08-21）：`ref_sign_ids` 只含**候選撈得到**的詞，撈不到的被
    移到 `oov_items` 不在參考裡。test_corpus 詞涵蓋率 69.7%，等於參考答案有
    30.3%（382/1259 個 token）一開始就被拿掉。拿那個當參考算出來的 BLEU
    回答的是「有沒有從候選裡挑對」，**不是「翻譯對不對」**，數值天生偏高，
    誤當品質引用會嚴重高估。故另算一組對完整參考的指標並列呈現。
    """
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            e = json.loads(line)
            out.setdefault(e["id"], e["gloss_text"])
    return out


def obj_nr_fallback(r: dict) -> bool:
    """抓不到機率時的退路：讀模型自己輸出的 needs_review。"""
    obj = parse_output(r.get("raw", ""))
    return bool(obj.get("needs_review", False)) if obj else False


def _prf(tp: int, fp: int, fn: int) -> dict:
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    return {"precision": round(prec, 4) if prec is not None else None,
            "recall": round(rec, 4) if rec is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "tp": tp, "fp": fp, "fn": fn}


def evaluate(rows: list[dict], nr_threshold: float = NEEDS_REVIEW_THRESHOLD,
             full_refs: dict[str, str] | None = None) -> dict:
    id2gloss = load_inventory()

    valid = 0
    viol_rows = 0          # 至少違反一次的題數
    viol_tokens = 0        # 違反的 ID 個數
    total_pred_ids = 0
    unplayable = 0         # 格式像 ID 但總表查無此 ID
    ref_gloss, hyp_gloss = [], []
    groups = []

    # needs_review 混淆矩陣。同時算兩組：
    #   greedy     — 直接讀模型輸出的 true/false（貪婪解碼的硬決策）
    #   calibrated — 讀 p_needs_review 再套門檻（需推論時有存該欄）
    tp = fp = tn = fn = 0
    c_tp = c_fp = c_tn = c_fn = 0
    have_p = 0

    for r in rows:
        obj = parse_output(r.get("raw", ""))
        cands = set(r.get("candidate_ids") or [])
        ref_ids = list(r.get("ref_sign_ids") or [])
        ref_nr = bool(r.get("ref_needs_review", False))

        # 校準決策：有機率就用門檻，抓不到就**退回貪婪**——這才是可上線的策略，
        # 而且分母與 greedy 那組一致。若把抓不到的排除在外，recall 會被灌水。
        p_nr = r.get("p_needs_review")
        if p_nr is not None or obj_nr_fallback(r) is not None:
            if p_nr is not None:
                have_p += 1
                cal = p_nr >= nr_threshold
            else:
                cal = obj_nr_fallback(r)
            if ref_nr and cal:
                c_tp += 1
            elif not ref_nr and cal:
                c_fp += 1
            elif not ref_nr and not cal:
                c_tn += 1
            else:
                c_fn += 1

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

    # 舊的「詞彙表內率」在新格式下已由候選清單取代（候選只從有影片的手語裡出），
    # 留著會顯示 0.0% 誤導成模型全錯。改由 Playable% 承擔這個角色。
    out = {k: v for k, v in base.items() if not k.startswith("InVocab")}
    # ---- 對完整參考再算一組（含被候選漏掉的詞，一律計為未命中）----
    full_block = None
    if full_refs:
        f_ref = [full_refs.get(g, "") for g in groups]
        hit = sum(1 for x in f_ref if x)
        if hit:
            try:
                fb = metrics.evaluate(f_ref, hyp_gloss, vocab, groups=groups)
                full_block = {k: v for k, v in fb.items()
                              if k.split("%")[0] in ("BLEU-4", "ROUGE-L", "ExactMatch")
                              or k in ("BLEU-4", "ROUGE-L", "ExactMatch%")}
                ref_tok = sum(len([t for t in x.split("/") if t.strip()]) for x in f_ref)
                trunc_tok = sum(len(r.get("ref_sign_ids") or []) for r in rows)
                full_block["ref_tokens_full"] = ref_tok
                full_block["ref_tokens_in_candidates"] = trunc_tok
                full_block["ref_tokens_dropped%"] = (
                    round(100 * (ref_tok - trunc_tok) / ref_tok, 2) if ref_tok else None)
                full_block["rows_matched"] = hit
            except Exception as e:  # noqa: BLE001
                full_block = {"metrics_error": str(e)}

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
            **_prf(tp, fp, fn), "tn": tn,
            "ref_positive_rate": round((tp + fn) / n, 4) if n else None,
            "decision": "greedy（模型輸出的 true/false）",
        },
    })
    if have_p:
        out["NeedsReview_calibrated"] = {
            **_prf(c_tp, c_fp, c_fn), "tn": c_tn,
            "threshold": nr_threshold,
            "rows_with_prob": have_p,
            "rows_fallback_to_greedy": n - have_p,
            "decision": f"p_needs_review >= {nr_threshold}（門檻在 dev 上選定）",
        }
    else:
        out["NeedsReview_calibrated"] = None   # 推論時沒存 p_needs_review

    # 主 BLEU/ROUGE/EM 是對「候選內參考」算的，改名讓語意不會被誤讀
    out["_ref_scope"] = ("上方 BLEU-4／ROUGE-L／ExactMatch%% 是對**候選內參考**算的"
                         "（撈不到的詞不在參考裡），回答『有沒有從候選裡挑對』；"
                         "Full_reference 那組才是含檢索缺口的系統整體水準。")
    out["Full_reference"] = full_block
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True,
                    help="逐句預測 jsonl（需含 candidate_ids / ref_sign_ids / raw）")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--full-ref", type=Path, default=None,
                    help="原始切分檔（data/splits/<split>.jsonl），用來對**完整參考**"
                         "另算一組指標。不給則依 --pred 檔名推斷。")
    ap.add_argument("--nr-threshold", type=float, default=NEEDS_REVIEW_THRESHOLD,
                    help="needs_review 的機率門檻。預設 %(default)s，"
                         "是 2026-08-21 在 dev 上依『recall>=0.7 下最大化 precision』選的。"
                         "**不要看 test 的數字調這個。**")
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            args.pred.read_text(encoding="utf-8").splitlines() if l.strip()]
    split_file = args.full_ref
    if split_file is None:
        stem = args.pred.stem
        # 長名在前：test_textbook 若排在 test 之後會先被 "test" 誤配
        for name in ("test_textbook", "test_corpus", "test_papers", "test", "dev", "train"):
            if stem.endswith(name):
                split_file = BASE / "data" / "splits" / f"{name}.jsonl"
                break
    full_refs = load_full_refs(split_file) if split_file else {}
    if not full_refs:
        print(f"⚠ 找不到完整參考（{split_file}），Full_reference 那組略過。"
              f"data/splits/*.jsonl 有 gitignore，需先跑 split_data.py 再生。")
    res = evaluate(rows, nr_threshold=args.nr_threshold, full_refs=full_refs)
    print(json.dumps(res, ensure_ascii=False, indent=2))

    out = args.out or args.pred.with_name(args.pred.stem + "_scriptmetrics.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n寫出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
