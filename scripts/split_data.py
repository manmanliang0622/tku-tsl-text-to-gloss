#!/usr/bin/env python3
"""第一階段 3.4：train/dev/test 切分。

實驗設計（本計畫決策，對應計畫 6.4 對照組）：
  - **test = 33 句真實已審核句**（tsl_sentences.jsonl 排除模板句 S24/S26），
    與 Stage A 提示法基線用的評測集完全相同 → 微調結果可與基線直接比較，
    且這批真實句永不進訓練集（無洩漏）。
  - **train/dev = 合成句（synth）＋ 中正辭典例句（twtsl）**。
    dev 由訓練池分層抽樣（依 source/confidence），供 early stopping。

洩漏防護（兩層）：
  1. train/dev 中若有句子的中文或 gloss_text 與 test 完全相同即剔除。
     （句型會與 test 重疊——這正是要測的泛化；只剔除「完全相同的句」。）
  2. train↔dev 去洩漏（2026-07-23）：dev 依 group 整組留存，同一段對話／詞條／
     模板不會同時落在 train 與 dev，避免同源近似句造成 dev 分數虛高、影響 early
     stopping。manifest 的 dev_group_leakage 應為 0。

資料品質標記：synth 與 twtsl 目前 review_status=pending（未經本團隊人工審核）。
本切分產出的 manifest.json 會記錄各來源筆數與審核狀態，供報告據實說明。

用法：
  python3 scripts/split_data.py                       # 預設：排除 rule-derived，只用
                                                      #   attested/corpus 合成 + twtsl + 語料庫
  python3 scripts/split_data.py --include-rule-derived # 納回 rule-derived（僅供實驗，非正式訓練）
  python3 scripts/split_data.py --no-corpus            # 不加文化部語料庫（回到舊組成）
  python3 scripts/split_data.py --include-words 500    # 額外加 N 筆辭典詞→gloss 對

2026-07-23 更新：依全資料審核，rule-derived 合成句未經母語者逐句裁定，**預設排除**。
需納入時明確加 --include-rule-derived，且結果只能作管線驗證、不得作正式訓練報告依據。

2026-07-21 更新：train/dev 池加入文化部《臺灣手語語料庫》全爬平行語料
（data/tslcorpus/parallel.jsonl，5,272 句真實 Text↔Gloss）。這是最大宗真實資料，
預設納入；以 --min-gloss-len 過濾過短碎片（是／爺爺 等單詞句）。
"""
import argparse
import json
import random
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT = DATA / "splits"


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def group_key(e, source):
    """去洩漏切分的分組鍵：同組資料只會整組落在 train 或 dev，不跨兩邊。

    語料庫按對話分組（seg_uuid 去掉段落號 P\\d+，讓同一段對話的各段落／句子
    同進退）；辭典例句按詞條；合成句按模板；其餘每筆自成一組。
    """
    if source == "tslcorpus":
        u = e.get("seg_uuid") or (e.get("corpus_id", "").split("/")[0]) or e["id"]
        return "corpus:" + re.sub(r"P\d+$", "", u)
    if source == "twtsl-sentence":
        return "twtsl:" + str(e.get("headword") or e["id"])
    if source == "synth":
        return "synth:" + str(e.get("template_id") or e["id"])
    return f"{source}:{e['id']}"


def norm_record(e, split_source):
    """統一訓練用欄位。"""
    return {
        "id": e["id"],
        "chinese": e["chinese"],
        "gloss_text": e["gloss_text"],
        "source": split_source,
        "group": group_key(e, split_source),
        "confidence": e.get("confidence"),
        "review_status": e.get("review_status", "n/a"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-ratio", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-rule-derived", action="store_true",
                    help="納入 rule-derived 合成句（預設排除；依 2026-07-23 審核，"
                         "rule-derived 未經母語者逐句裁定，不得進正式訓練）")
    ap.add_argument("--include-words", type=int, default=0,
                    help="額外加入 N 筆辭典詞→gloss 對（0=不加）")
    ap.add_argument("--no-corpus", action="store_true",
                    help="不加入文化部語料庫平行語料")
    ap.add_argument("--min-gloss-len", type=int, default=2,
                    help="語料庫句最小 Gloss token 數（濾掉過短碎片，預設 2）")
    args = ap.parse_args()
    exclude_rule_derived = not args.include_rule_derived  # 預設 True（審核安全預設）
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
        if exclude_rule_derived and e.get("confidence") == "rule-derived":
            continue
        pool.append(norm_record(e, "synth"))
    twtsl_sents = load_jsonl(DATA / "twtsl" / "twtsl_sentences.jsonl")
    for e in twtsl_sents:
        pool.append(norm_record(e, "twtsl-sentence"))

    corpus_dropped_short = 0
    corpus_path = DATA / "tslcorpus" / "parallel.jsonl"
    if not args.no_corpus and corpus_path.exists():
        for e in load_jsonl(corpus_path):
            if len(e.get("gloss", [])) < args.min_gloss_len:
                corpus_dropped_short += 1
                continue
            pool.append(norm_record(e, "tslcorpus"))

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

    # --- 依 source 分層、依 group 去洩漏抽 dev（整組進 train 或 dev） ---
    from collections import defaultdict
    by_src = defaultdict(list)
    for e in dedup:
        by_src[e["source"]].append(e)
    train, dev = [], []
    for src, items in by_src.items():
        groups = defaultdict(list)
        for e in items:
            groups[e["group"]].append(e)
        gkeys = list(groups.keys())
        rng.shuffle(gkeys)
        target = round(len(items) * args.dev_ratio) if len(items) >= 10 else 0
        dev_groups, dev_count = set(), 0
        for gk in gkeys:
            if dev_count >= target:
                break
            dev_groups.add(gk)
            dev_count += len(groups[gk])
        for gk in gkeys:
            (dev if gk in dev_groups else train).extend(groups[gk])
    rng.shuffle(train)
    rng.shuffle(dev)

    # 去洩漏驗證：同一 group 不得同時出現在 train 與 dev
    train_groups = {e["group"] for e in train}
    dev_groups_all = {e["group"] for e in dev}
    group_leak = len(train_groups & dev_groups_all)
    assert group_leak == 0, f"分組洩漏 {group_leak} 組同時在 train/dev"

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
        "exclude_rule_derived": exclude_rule_derived,
        "include_words": args.include_words,
        "counts": {"train": len(train), "dev": len(dev), "test": len(test)},
        "train_composition": compo(train),
        "dev_composition": compo(dev),
        "test_composition": compo(test),
        "leaked_removed": leaked,
        "split_method": "group-holdout（語料庫按對話 seg_uuid、twtsl 按詞條、synth 按模板整組留存）",
        "dev_group_leakage": group_leak,
        "n_groups": {"train": len(train_groups), "dev": len(dev_groups_all)},
        "corpus_dropped_short": corpus_dropped_short,
        "min_gloss_len": args.min_gloss_len,
        "no_corpus": args.no_corpus,
        "note": ("test=Stage A 相同的 33 句真實已審核句，永不進訓練；"
                 "train/dev 來源 synth／twtsl／tslcorpus 目前 review_status=pending，"
                 "本輪為管線驗證，最終報告需依人工審核結果更新。"
                 "tslcorpus＝文化部語料庫全爬真實平行語料（最大宗真實資料）。"),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
