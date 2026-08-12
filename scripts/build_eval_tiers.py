#!/usr/bin/env python3
"""建立三層診斷評估的資料集（不動任何訓練資料與模型）。

診斷目的：分辨模型是「沒學會」還是「學會了但泛化不足」。

  A. Seen         訓練集原句原輸入 → 連背過的都做不到，就是訓練設定有問題
  B. Paraphrase   同一批句子、中文換句話說 → 掉很多代表在記憶中文字面而非學轉換
  C. Unseen       完全沒看過的句子，且用詞多半學過 → 掉很多代表不會重組語序

**為什麼 A 與 B 用同一批 100 句**：若 A、B 各自獨立抽樣，兩者的差距可能只是
抽樣差異（句子難度不同）。用同一批句子、只改中文說法，差距才能歸因於
「換句話說」這個唯一變因。

**為什麼 C 要排除 OOV 過多的句子**：若測試句大量使用訓練沒出現過的 Gloss 詞，
模型必然錯，但那是「詞彙沒教過」而非「不會組合語序」。混在一起就分不出病因，
故預設只收 OOV 比例 ≤20% 的句子。

輸出：data/eval_tiers/{seen,paraphrase,unseen}.jsonl

用法：
  python3 scripts/build_eval_tiers.py
  python3 scripts/build_eval_tiers.py --n-unseen 100 --max-oov 0.2
"""
import argparse
import json
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SPLITS = BASE / "data" / "splits_json"
OUT = BASE / "data" / "eval_tiers"

SEEN_SEED = 20260812      # 固定種子：每次抽到同一批（計畫要求可重現）


def load(path):
    p = Path(path)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def gloss_of(row):
    return json.loads(row["output"])["gloss"]


def char_supported(token, chinese):
    """Gloss 詞是否「在中文裡找得到依據」——至少一個字出現在中文句中。

    這是刻意寬鬆的啟發式，只當**安全網**用：抓出改寫後某個 Gloss 詞
    完全失去中文依據的情況（如「沮喪」被改成「難過」）。
    它抓不到語意細微改變（如「借」→「租」都有字但語意不同），
    故輸出仍保留 transform 欄位供人工複核。
    """
    return any(ch in chinese for ch in str(token))


def build_seen(train_rows, n, min_len):
    """A 層：從訓練集抽 n 句（固定種子）。太短的句子改寫空間有限，故設下限。"""
    cand = [r for r in train_rows if len(r["input"]) >= min_len]
    picked = random.Random(SEEN_SEED).sample(cand, min(n, len(cand)))
    return [{"idx": i, "test_type": "seen", "chinese": r["input"],
             "expected_gloss": gloss_of(r), "context": r.get("context", ""),
             "source": "train"}
            for i, r in enumerate(picked, 1)]


def build_paraphrase(seen_rows, para_path):
    """B 層：把改寫句對回 A 層同 idx 的句子。

    自動安全網會找出「原句在中文裡找得到依據、改寫後找不到」的 Gloss 詞，
    這代表該句需要**語意映射**而非照抄字面（如「不過」→ Gloss「但是」）。

    ⚠️ 這個旗標不等於「標準答案要改」。2026-08-12 逐句複核 9 個被標記的句子，
    語意全部等價、Gloss 皆不需改變，已在 paraphrases.jsonl 記為
    `reviewed_gloss_same` 並附裁定理由。未經複核者一律標為待複核，
    評分時單獨列出，不會默默當成答案相同。
    """
    paras = {r["idx"]: r for r in load(para_path)}
    out, need_mapping, unreviewed = [], 0, 0
    for s in seen_rows:
        p = paras.get(s["idx"])
        if not p:
            continue
        orig, new = s["chinese"], p["chinese"]
        if new.strip() == orig.strip():
            continue                       # 改寫等同原句者不計入（無測試價值）
        lost = [t for t in s["expected_gloss"].split()
                if char_supported(t, orig) and not char_supported(t, new)]
        reviewed = bool(p.get("reviewed_gloss_same"))
        need_mapping += bool(lost)
        unreviewed += bool(lost) and not reviewed
        out.append({"idx": s["idx"], "test_type": "paraphrase",
                    "chinese": new, "chinese_original": orig,
                    "expected_gloss": s["expected_gloss"],
                    "context": s["context"], "source": "train_paraphrased",
                    "transform": p.get("transform", ""),
                    # 需語意映射：中文字面已無依據，最能區辨「理解」與「背誦」
                    "needs_semantic_mapping": bool(lost),
                    "reviewed_gloss_same": reviewed,
                    "review_note": p.get("review_note", ""),
                    "lost_support_tokens": " ".join(lost)})
    return out, need_mapping, unreviewed


def build_unseen(train_rows, n, max_oov, seed):
    """C 層：正式 test 集中模型沒看過、且用詞多半學過的句子。"""
    train_vocab = set()
    for r in train_rows:
        train_vocab.update(gloss_of(r).split())
    train_inputs = {r["input"] for r in train_rows}

    pool = []
    for name in ("test", "test_corpus", "test_papers"):
        p = SPLITS / f"{name}.jsonl"
        if not p.exists():
            continue
        for r in load(p):
            # 訓練出現過的中文句一律排除：那不是「未見」，混進來會高估泛化
            if r["input"] in train_inputs:
                continue
            toks = gloss_of(r).split()
            if not toks:
                continue
            oov = [t for t in toks if t not in train_vocab]
            rate = len(oov) / len(toks)
            if rate > max_oov:
                continue
            pool.append({"test_type": "unseen", "chinese": r["input"],
                         "expected_gloss": gloss_of(r), "context": r.get("context", ""),
                         "source": name, "oov_rate": round(rate, 3),
                         "oov_tokens": " ".join(oov)})
    picked = random.Random(seed).sample(pool, min(n, len(pool)))
    for i, r in enumerate(picked, 1):
        r["idx"] = i
    return picked, len(pool), len(train_vocab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seen", type=int, default=100)
    ap.add_argument("--n-unseen", type=int, default=100)
    ap.add_argument("--min-len", type=int, default=6,
                    help="A/B 層的中文最短長度；太短的句子沒有改寫空間")
    ap.add_argument("--max-oov", type=float, default=0.2,
                    help="C 層允許的 OOV Gloss 比例上限")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    train = load(SPLITS / "train.jsonl")

    seen = build_seen(train, args.n_seen, args.min_len)
    (OUT / "seen.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in seen) + "\n", encoding="utf-8")
    print(f"A. Seen       : {len(seen)} 句（訓練集原句，seed={SEEN_SEED}）")

    para_path = OUT / "paraphrases.jsonl"
    if para_path.exists():
        para, need_mapping, unreviewed = build_paraphrase(seen, para_path)
        (OUT / "paraphrase.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in para) + "\n", encoding="utf-8")
        print(f"B. Paraphrase : {len(para)} 句（其中 {need_mapping} 句中文字面已無依據、"
              f"需語意映射，單獨統計）")
        if unreviewed:
            print(f"                ⚠️ {unreviewed} 句尚未人工複核標準答案是否該改，"
                  f"請見 paraphrase.jsonl 的 lost_support_tokens")
    else:
        print(f"B. Paraphrase : 略過（找不到 {para_path.relative_to(BASE)}）")

    unseen, pool_n, vocab_n = build_unseen(train, args.n_unseen, args.max_oov, args.seed)
    (OUT / "unseen.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in unseen) + "\n", encoding="utf-8")
    by_src = {}
    for r in unseen:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    print(f"C. Unseen     : {len(unseen)} 句（符合條件的池子 {pool_n} 句，"
          f"OOV≤{args.max_oov:.0%}；訓練詞彙 {vocab_n} 詞）")
    print(f"                來源分布：{by_src}")
    print(f"\n輸出 → {OUT.relative_to(BASE)}/")


if __name__ == "__main__":
    main()
