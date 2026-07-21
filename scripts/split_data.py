#!/usr/bin/env python3
"""第一階段 3.4：train/dev/test 切分。

實驗設計（本計畫決策，對應計畫 6.4 對照組）：
  - **test = 33 句真實已審核句**（tsl_sentences.jsonl 排除模板句 S24/S26），
    與 Stage A 提示法基線用的評測集完全相同 → 微調結果可與基線直接比較，
    且這批真實句永不進訓練集（無洩漏）。
  - **train/dev = 合成句（synth）＋ 中正辭典例句（twtsl）**。
    dev 由訓練池分層抽樣（依 source/confidence），供 early stopping。

洩漏防護：train/dev 中若有句子的中文或 gloss_text 與 test 完全相同即剔除。
（句型會與 test 重疊——這正是要測的泛化；只剔除「完全相同的句」。）

資料品質標記：synth 與 twtsl 目前 review_status=pending（未經本團隊人工審核）。
本切分產出的 manifest.json 會記錄各來源筆數與審核狀態，供報告據實說明。

用法：
  python3 scripts/split_data.py                       # 預設：synth 全部 + twtsl 例句
  python3 scripts/split_data.py --exclude-rule-derived # 只用 attested/corpus + twtsl
  python3 scripts/split_data.py --include-words 500    # 額外加 N 筆辭典詞→gloss 對
"""
import argparse
import json
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT = DATA / "splits"


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def norm_record(e, split_source):
    """統一訓練用欄位。"""
    return {
        "id": e["id"],
        "chinese": e["chinese"],
        "gloss_text": e["gloss_text"],
        "source": split_source,
        "confidence": e.get("confidence"),
        "review_status": e.get("review_status", "n/a"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-ratio", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-rule-derived", action="store_true",
                    help="排除 confidence=rule-derived 的合成句")
    ap.add_argument("--include-words", type=int, default=0,
                    help="額外加入 N 筆辭典詞→gloss 對（0=不加）")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    OUT.mkdir(exist_ok=True)

    # --- test：真實句（排除模板佔位符句） ---
    real = load_jsonl(DATA / "tsl_sentences.jsonl")
    test = [norm_record(e, "real") for e in real if not e.get("is_template")]
    test_chinese = {e["chinese"] for e in test}
    test_gloss = {e["gloss_text"] for e in test}

    # --- train 池：synth + twtsl 例句 ---
    pool = []
    synth = load_jsonl(DATA / "synth" / "tsl_synth.jsonl")
    for e in synth:
        if args.exclude_rule_derived and e.get("confidence") == "rule-derived":
            continue
        pool.append(norm_record(e, "synth"))
    twtsl_sents = load_jsonl(DATA / "twtsl" / "twtsl_sentences.jsonl")
    for e in twtsl_sents:
        pool.append(norm_record(e, "twtsl-sentence"))

    if args.include_words > 0:
        words = load_jsonl(DATA / "twtsl" / "twtsl_words.jsonl")
        rng.shuffle(words)
        added = 0
        for e in words:
            if added >= args.include_words:
                break
            pool.append(norm_record(e, "twtsl-word"))
            added += 1

    # --- 去重 + 洩漏防護 ---
    seen, dedup, leaked = set(), [], 0
    for e in pool:
        if e["chinese"] in test_chinese or e["gloss_text"] in test_gloss:
            leaked += 1
            continue
        key = (e["chinese"], e["gloss_text"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(e)

    # --- 分層抽 dev（依 source 分層，確保各來源都有 dev 樣本） ---
    from collections import defaultdict
    by_src = defaultdict(list)
    for e in dedup:
        by_src[e["source"]].append(e)
    train, dev = [], []
    for src, items in by_src.items():
        rng.shuffle(items)
        n_dev = max(1, round(len(items) * args.dev_ratio)) if len(items) >= 10 else 0
        dev.extend(items[:n_dev])
        train.extend(items[n_dev:])
    rng.shuffle(train)
    rng.shuffle(dev)

    for name, rows in [("train", train), ("dev", dev), ("test", test)]:
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for e in rows:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    def compo(rows):
        from collections import Counter
        return dict(Counter(e["source"] for e in rows))

    manifest = {
        "seed": args.seed,
        "dev_ratio": args.dev_ratio,
        "exclude_rule_derived": args.exclude_rule_derived,
        "include_words": args.include_words,
        "counts": {"train": len(train), "dev": len(dev), "test": len(test)},
        "train_composition": compo(train),
        "dev_composition": compo(dev),
        "test_composition": compo(test),
        "leaked_removed": leaked,
        "note": ("test=Stage A 相同的 33 句真實已審核句，永不進訓練；"
                 "train/dev 來源 synth 與 twtsl 目前 review_status=pending，"
                 "本輪為管線驗證，最終報告需依人工審核結果更新。"),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
