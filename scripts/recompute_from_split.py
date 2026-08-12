#!/usr/bin/env python3
"""拿既有的模型輸出檔，換上目前切分裡的參考答案，離線重算指標與分維度breakdown。

用途一：參考答案被修正時（如 2026-08-13 論文替代詞解析修正），模型輸出並未
改變，不需要重跑推論或重訓——只要把 `results/*.jsonl` 裡的 `ref` 換成切分
檔的最新 `gloss_text` 再算一次即可。

用途二：**不碰 GPU 就能產出教授要求的分維度評估**。既有的 `results/*.jsonl`
已存有逐句預測，故語序／選詞／漏詞／亂加詞／未知詞的分類可以完全離線算出，
不必重跑推論。（語意維度自動指標答不了，需母語者人工評分。）

OOV 判定基準為 **詞彙總表 ∪ 訓練詞彙**（`gloss_fallback.load_vocab`），
不是訓練詞彙——後者會把「合法但訓練沒出現的手語詞」誤判成造詞，實測高估 67%。

對齊方式：逐列索引。`eval_json_model.py` 會先去掉句尾標點才餵給模型，
故 `chinese` 欄可能與切分檔差一個標點；本腳本以索引對齊並驗證去標點後
的中文完全相同，不同即中止（代表切分已被重新產生，結果檔不可再對齊）。

用法：
  python3 scripts/recompute_from_split.py \
      --results results/v11_test_papers_test.jsonl --split test_papers
"""
import argparse
import csv
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
import eval_metrics_ext as ext  # noqa: E402
import metrics  # noqa: E402
from gloss_fallback import load_vocab  # noqa: E402

# 半形逗號／句點也要納入：語料庫轉寫混用全形與半形（「…多又好吃,」），
# 漏掉會讓對齊閘門誤報不符
PUNCT = " 。，、！!？?；;：:.,"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="模型輸出 jsonl（含 chinese/ref/pred）")
    ap.add_argument("--split", required=True, help="切分名稱，如 test_papers")
    ap.add_argument("--csv", default=None, help="另存逐句結果 CSV 的路徑")
    args = ap.parse_args()

    recs = [json.loads(l) for l in Path(args.results).open()]
    rows = [json.loads(l) for l in (BASE / "data" / "splits" / f"{args.split}.jsonl").open()]
    if len(recs) != len(rows):
        sys.exit(f"列數不同：結果檔 {len(recs)}、切分 {len(rows)}，無法對齊")
    for i, (a, b) in enumerate(zip(recs, rows)):
        if a["chinese"].strip(PUNCT) != b["chinese"].strip(PUNCT):
            sys.exit(f"第 {i} 列中文不符：{a['chinese']!r} vs {b['chinese']!r}\n"
                     "切分可能已重新產生，請重跑推論而非離線重算。")

    vocab = set(json.load((BASE / "data" / "tsl_gloss_vocab.json").open())["glosses"])
    preds = [r["pred"] for r in recs]
    refs = [r["gloss_text"] for r in rows]
    old = metrics.evaluate([r["ref"] for r in recs], preds, vocab)
    new = metrics.evaluate(refs, preds, vocab)
    changed = sum(1 for a, b in zip(recs, rows) if a["ref"] != b["gloss_text"])

    print(f"{args.results}  n={len(recs)}  參考答案有變動 {changed} 句\n")
    print(f"{'指標':<14}{'舊參考':>10}{'新參考':>10}{'差':>9}")
    for k in ("ExactMatch%", "ROUGE-L", "BLEU-4", "InVocab%", "InVocabRef%"):
        d = new[k] - old[k]
        print(f"{k:<14}{old[k]:>10.2f}{new[k]:>10.2f}{d:>+9.2f}")

    # --- 分維度 breakdown（教授要求：不要只看 EM／BLEU）---
    legal, _ = load_vocab()               # 合法詞＝詞彙總表 ∪ 訓練詞彙
    per = [ext.score_pair(ref, p, legal) for ref, p in zip(refs, preds)]
    agg = ext.aggregate(per)
    print(f"\n分維度（OOV 基準：詞彙總表 ∪ 訓練詞彙，{len(legal)} 詞）")
    print(f"  TokenF1 {agg['TokenF1']:.4f}   GER {agg['GER']:.4f}"
          f"   平均編輯距離 {agg['AvgEditDistance']}")
    print(f"\n  {'錯誤型態':<16}{'句數':>6}{'佔比':>9}")
    for k, v in agg["ErrorTypes"].items():
        print(f"  {k:<16}{v:>6}{agg['ErrorTypes%'][k]:>8.2f}%")
    print("\n  ※ 語意維度自動指標答不了，需母語者 5 分制人工評分（計畫 6.2）")

    if args.csv:
        out = Path(args.csv)
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "chinese", "ref", "pred"]
                               + list(per[0].keys()))
            w.writeheader()
            for rec, row, s in zip(recs, rows, per):
                w.writerow({"id": row["id"], "chinese": row["chinese"],
                            "ref": row["gloss_text"], "pred": rec["pred"], **s})
        print(f"\n逐句結果 → {out}")


if __name__ == "__main__":
    main()
