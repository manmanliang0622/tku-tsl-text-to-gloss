#!/usr/bin/env python3
"""建立統一 Gloss 主詞彙表（計畫 6.1 節「詞彙表內率」的分母來源）。

合併三個來源的 Gloss，標註每個詞的出處、語料庫頻率與是否有辭典詞條：
  1. 自有詞彙表   data/tsl_gloss_vocab.json（有自有動作影片）
  2. 中正手語辭典 data/twtsl/twtsl_words.jsonl（正式名＋別名；有示範影片）
  3. 文化部語料庫 data/tslcorpus/parallel.jsonl（真實句子用到的 Gloss token）

正規化：辭典多用「台」、語料庫多用「臺」等異體字；本表建立正規化鍵
（臺→台、著→着 等）來判斷「換算後是否有辭典詞條」，避免僅因字體差異
而誤判為詞彙表外。原字形保留於 surface，不改寫資料本身。

輸出：
  data/vocab/gloss_master.jsonl  一詞一行（surface, sources, corpus_freq,
                                  has_dict_entry, norm_has_dict_entry, twtsl_ids）
  data/vocab/coverage.json       覆蓋率摘要（供評估腳本引用）
"""
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT = DATA / "vocab"

# 異體字正規化（僅供覆蓋率比對，不改寫原資料）
NORM_MAP = str.maketrans({"臺": "台", "着": "著", "菸": "煙", "祐": "佑"})


def norm(s):
    return s.translate(NORM_MAP)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    own = set(json.load((DATA / "tsl_gloss_vocab.json").open(encoding="utf-8"))["glosses"])

    twtsl_canon, twtsl_alias, twtsl_ids = set(), set(), {}
    for line in (DATA / "twtsl" / "twtsl_words.jsonl").read_text(encoding="utf-8").splitlines():
        w = json.loads(line)
        twtsl_canon.add(w["gloss_text"])
        twtsl_ids.setdefault(w["gloss_text"], []).append(w["twtsl_id"])
        for a in w["aliases"]:
            twtsl_alias.add(a)

    corpus_freq = Counter()
    corpus_path = DATA / "tslcorpus" / "parallel.jsonl"
    if corpus_path.exists():
        for line in corpus_path.read_text(encoding="utf-8").splitlines():
            for g in json.loads(line)["gloss"]:
                corpus_freq[g] += 1

    dict_surface = twtsl_canon | twtsl_alias        # 辭典可直接對到的字形
    dict_norm = {norm(x) for x in dict_surface | own}

    all_surface = own | twtsl_canon | twtsl_alias | set(corpus_freq)
    rows = []
    for g in sorted(all_surface):
        sources = []
        if g in own:
            sources.append("own")
        if g in twtsl_canon:
            sources.append("twtsl")
        elif g in twtsl_alias:
            sources.append("twtsl-alias")
        if g in corpus_freq:
            sources.append("corpus")
        rows.append({
            "surface": g,
            "sources": sources,
            "corpus_freq": corpus_freq.get(g, 0),
            "has_dict_entry": g in dict_surface or g in own,
            "norm_has_dict_entry": norm(g) in dict_norm,
            "twtsl_ids": twtsl_ids.get(g, []),
        })

    with (OUT / "gloss_master.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 覆蓋率摘要（以語料庫真實使用的 Gloss 為母體）
    corpus_vocab = set(corpus_freq)
    in_dict = {g for g in corpus_vocab if g in dict_surface or g in own}
    in_dict_norm = {g for g in corpus_vocab if norm(g) in dict_norm}
    tok_total = sum(corpus_freq.values())
    tok_in_norm = sum(f for g, f in corpus_freq.items() if norm(g) in dict_norm)
    coverage = {
        "own_gloss": len(own),
        "twtsl_canonical": len(twtsl_canon),
        "twtsl_alias": len(twtsl_alias),
        "corpus_unique_gloss": len(corpus_vocab),
        "corpus_token_total": tok_total,
        "master_total_surface": len(rows),
        "corpus_type_in_dict_raw": len(in_dict),
        "corpus_type_in_dict_raw_pct": round(len(in_dict) / len(corpus_vocab) * 100, 1),
        "corpus_type_in_dict_norm": len(in_dict_norm),
        "corpus_type_in_dict_norm_pct": round(len(in_dict_norm) / len(corpus_vocab) * 100, 1),
        "corpus_token_in_dict_norm_pct": round(tok_in_norm / tok_total * 100, 1),
    }
    (OUT / "coverage.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"OK: 主詞彙表 {len(rows)} 詞 → data/vocab/gloss_master.jsonl")
    for k, v in coverage.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
