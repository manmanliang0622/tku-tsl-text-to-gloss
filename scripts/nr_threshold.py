#!/usr/bin/env python3
"""候選覆蓋風險旗標（v1 名為 needs_review）的門檻選定與重算，不需重新推論。

背景：v14–v17 的門檻都是用「recall >= 0.7 下最大化 precision」選的。
2026-08-27 的診斷發現這條規則偏保守——dev 上改成直接最大化 F1 會選到
0.0421（F1 0.741 vs 現行 0.095349 的 0.702），而且該值在 corpus 與
textbook 上同向更好（+0.083 / +0.056），所以不是過擬合 dev，是選法問題。

用法：
    # 在 dev 上選門檻（兩種規則都印出來對照）
    python3 nr_threshold.py select results/v17cd_dev.jsonl

    # 用指定門檻重算某個結果檔的 CandidateCoverageRisk 區塊
    python3 nr_threshold.py score results/v17cd_dev.jsonl --threshold 0.0421

    # 驗證：用舊門檻重算，應與既有 *_scriptmetrics.json 的區塊逐欄相同
    python3 nr_threshold.py score results/v17cd_dev.jsonl --threshold 0.095349

只重算 CandidateCoverageRisk / *_calibrated——BLEU、ROUGE、違反率、
Full_reference 都與門檻無關，不受影響。

2026-08-31 起讀取端 v1／v2 兩種欄位名都收（見 script_schema.read_flag），
所以 v14–v17 的舊結果檔與日後 v2 的新結果檔都能直接餵進來。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import script_schema  # noqa: E402


def load(path):
    """回傳 [(參考旗標, 機率, greedy 旗標)]，v1／v2 鍵名皆收。"""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                greedy = script_schema.read_flag(json.loads(obj["raw"]), default=None)
            except Exception:
                greedy = None          # 破 JSON → 落回 greedy 時視為未預測
            ref_flag = next((bool(obj[k]) for k in
                             ("ref_candidate_coverage_risk", "ref_needs_review")
                             if k in obj), False)
            rows.append((ref_flag,
                         obj.get("p_needs_review"),
                         greedy))
    return rows


def confusion(rows, threshold):
    """門檻判定；沒有機率的列落回模型自己輸出的 true/false（與既有管線一致）。"""
    tp = fp = fn = tn = fallback = with_prob = 0
    for ref, prob, greedy in rows:
        if prob is None:
            pred = bool(greedy)
            fallback += 1
        else:
            pred = prob >= threshold
            with_prob += 1
        if pred and ref:
            tp += 1
        elif pred and not ref:
            fp += 1
        elif not pred and ref:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn, with_prob, fallback


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else None
    return precision, recall, f1


def block(rows, threshold):
    tp, fp, fn, tn, with_prob, fallback = confusion(rows, threshold)
    precision, recall, f1 = prf(tp, fp, fn)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4) if f1 is not None else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "threshold": threshold,
        "rows_with_prob": with_prob,
        "rows_fallback_to_greedy": fallback,
        "decision": f"p_needs_review >= {threshold}（門檻在 dev 上選定）",
    }


def greedy_block(rows):
    tp = fp = fn = tn = 0
    for ref, _prob, greedy in rows:
        pred = bool(greedy)
        if pred and ref:
            tp += 1
        elif pred and not ref:
            fp += 1
        elif not pred and ref:
            fn += 1
        else:
            tn += 1
    precision, recall, f1 = prf(tp, fp, fn)
    n = len(rows)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4) if f1 is not None else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "ref_positive_rate": round(sum(1 for r, _, _ in rows if r) / n, 4) if n else 0.0,
        "decision": "greedy（模型輸出的 true/false）",
    }


def candidates(rows):
    """門檻的候選切點：取相鄰機率的中點，避免門檻正好壓在某一列上。"""
    probs = sorted({p for _ref, p, _g in rows if p is not None})
    if not probs:
        return []
    cuts = [probs[0] / 2]
    cuts += [(a + b) / 2 for a, b in zip(probs, probs[1:])]
    cuts.append(probs[-1] + 1e-6)
    return cuts


def select(rows, min_recall=0.7):
    """回傳 {規則: (門檻, precision, recall, f1)}。"""
    best_f1 = (None, -1.0)
    best_legacy = (None, -1.0)
    for t in candidates(rows):
        tp, fp, fn, _tn, _wp, _fb = confusion(rows, t)
        precision, recall, f1 = prf(tp, fp, fn)
        if f1 is not None and f1 > best_f1[1]:
            best_f1 = (t, f1)
        if recall >= min_recall and precision > best_legacy[1]:
            best_legacy = (t, precision)

    out = {}
    for name, t in (("max_f1", best_f1[0]),
                    (f"legacy_recall>={min_recall}_max_precision", best_legacy[0])):
        if t is None:
            continue
        tp, fp, fn, _tn, _wp, _fb = confusion(rows, t)
        precision, recall, f1 = prf(tp, fp, fn)
        out[name] = (t, precision, recall, f1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["select", "score"])
    ap.add_argument("results", nargs="+")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--min-recall", type=float, default=0.7)
    args = ap.parse_args()

    if args.mode == "select":
        for path in args.results:
            rows = load(path)
            print(f"=== {path}  n={len(rows)}")
            for rule, (t, precision, recall, f1) in select(rows, args.min_recall).items():
                print(f"  {rule:38} t={t:.6f}  P={precision:.4f} R={recall:.4f} F1={f1:.4f}")
        return 0

    if args.threshold is None:
        print("score 模式需要 --threshold", file=sys.stderr)
        return 2
    for path in args.results:
        rows = load(path)
        print(f"=== {path}")
        print(json.dumps({"CandidateCoverageRisk": greedy_block(rows),
                          "CandidateCoverageRisk_calibrated": block(rows, args.threshold)},
                         ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
