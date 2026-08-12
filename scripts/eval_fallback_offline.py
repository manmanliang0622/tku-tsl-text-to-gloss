#!/usr/bin/env python3
"""離線量測 fallback 的效果：對既有的三層評估結果套用修復，再重算指標。

不需 GPU、不需重跑推論——直接讀 evaluation_results/*.csv 的既有預測，
套 gloss_fallback.repair_gloss 後重算，故可與原始結果逐句對照。

**要看的主要指標是「可播放率」而非 EM。** fallback 的目的不是提高字面正確率
（指拼標記本來就不會與參考答案字面相同），而是讓下游動作庫**收到的每個詞都
有辦法處理**：不是查得到的手語詞，就是明確的指拼指示。修復前，表外詞對下游
就是查不到、直接失敗。

用法：
  python3 scripts/eval_fallback_offline.py
輸出：evaluation_results/fallback_comparison.json
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import metrics
import eval_metrics_ext as ext
import gloss_fallback as fb

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "evaluation_results"


def renderable_rate(preds, rend):
    """可播放率：輸出詞中下游查得到動作的比例。指拼標記另計，不算可播放。"""
    toks = [t for p in preds for t in ext.tokenize(p)]
    if not toks:
        return 0.0, 0.0
    ok = sum(1 for t in toks
             if t in rend or fb.normalize(t) in rend)
    fs = sum(1 for t in toks if t.startswith(fb.FINGERSPELL_PREFIX))
    return round(ok / len(toks) * 100, 2), round(fs / len(toks) * 100, 2)


def score(refs, preds, vocab_for_oov, rend):
    rows = [ext.score_pair(r, p, vocab_for_oov) for r, p in zip(refs, preds)]
    agg = ext.aggregate(rows)
    r_tok = [metrics.tokenize(r.replace(" ", "/")) for r in refs]
    p_tok = [metrics.tokenize(p.replace(" ", "/")) for p in preds]
    agg["BLEU-4"] = round(metrics.corpus_bleu(r_tok, p_tok), 2)
    ok, fs = renderable_rate(preds, rend)
    agg["可播放率%"] = ok
    agg["指拼標記率%"] = fs
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+", default=["seen", "paraphrase", "unseen"])
    args = ap.parse_args()

    vocab, rend = fb.load_vocab()
    out = {"note": "主要看『可播放率』；EM 不會因 fallback 提高，指拼標記本就不與參考字面相同",
           "vocab_size": len(vocab), "renderable_size": len(rend), "tiers": {}}

    for tier in args.tiers:
        path = RESULTS / f"{tier}_results.csv"
        if not path.exists():
            print(f"  略過 {tier}（找不到 {path.name}）")
            continue
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
        refs = [r["expected_gloss"] for r in rows]
        before = [r["predicted_gloss"] for r in rows]

        after, rule_stats = [], Counter()
        for p in before:
            fixed, st = fb.repair_gloss(p, vocab, rend)
            after.append(fixed)
            rule_stats.update(st)

        b = score(refs, before, vocab, rend)
        a = score(refs, after, vocab, rend)
        changed = sum(1 for x, y in zip(before, after) if x != y)
        out["tiers"][tier] = {"修復前": b, "修復後": a,
                              "有改動的句子": changed, "規則使用": dict(rule_stats)}

        print(f"\n=== {tier}（{len(rows)} 句，{changed} 句被改動）===")
        print(f"{'指標':<14}{'修復前':>10}{'修復後':>10}{'變化':>10}")
        for k in ("可播放率%", "指拼標記率%", "ExactMatch%", "TokenF1", "GER", "BLEU-4"):
            bv, av = b.get(k, 0), a.get(k, 0)
            print(f"{k:<14}{bv:>10.2f}{av:>10.2f}{av - bv:>+10.2f}")
        print(f"規則使用：{dict(rule_stats)}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "fallback_comparison.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n輸出 → evaluation_results/fallback_comparison.json")


if __name__ == "__main__":
    main()
