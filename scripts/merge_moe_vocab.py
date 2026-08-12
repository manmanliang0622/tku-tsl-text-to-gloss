#!/usr/bin/env python3
"""清理教育部辭典爬取結果，併成可供 fallback 使用的詞彙表。

**這批資料的用途是擴充「合法詞彙判定」，不是當訓練資料。**
理由與中正辭典相同：辭典給的是「中文詞 → 手語示範影片」，中文詞本身就是
Gloss 標籤，等於同形對應。把同形對應餵進訓練只會強化「照抄中文」——
那正是三層診斷找到的病因（`results/three_tier_report.md`）。

真正的用途：`scripts/gloss_fallback.py` 要判斷「模型產出的詞是合法手語詞，
還是自創／照抄中文」。判定基準的涵蓋率直接決定誤判率——實測用訓練詞彙當
基準會把 40% 的合法詞誤判為造詞。教育部辭典帶來 6,549 個新詞，
可讓這個判定準得多。

清理項目（實測 8,442 詞中 59 個有問題，佔 0.7%）：
  - 去前後空白（46 個）與控制字元（2 個）
  - 括號註（10 個，如「敘薪（銓薪）」）拆成主詞＋別名，兩者都算合法
  - 無漢字或空字串者剔除

⚠️ 授權未查證，見 scripts/scrape_moe_signdict.py 開頭說明。輸出留在
data/moe/（未納入版本控制）。

用法：
  python3 scripts/merge_moe_vocab.py
輸出：data/moe/moe_vocab_clean.jsonl
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "data" / "moe" / "moe_words.jsonl"
OUT = BASE / "data" / "moe" / "moe_vocab_clean.jsonl"
PAREN = re.compile(r"^(.+?)[（(]([^）)]+)[）)]$")


def clean(text):
    """去控制字元與前後空白；回傳 None 表示應剔除。"""
    if not text:
        return None
    t = "".join(c for c in str(text) if unicodedata.category(c) != "Cc").strip()
    t = t.strip("　 ")
    if not t or not re.search(r"[一-鿿]", t):
        return None
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"找不到 {src}，請先跑 scripts/scrape_moe_signdict.py")
        return

    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    merged, dropped, split_alias = {}, 0, 0
    for r in rows:
        t = clean(r.get("chinese"))
        if not t:
            dropped += 1
            continue
        # 「敘薪（銓薪）」→ 主詞「敘薪」＋別名「銓薪」，兩者都是合法說法
        aliases = []
        m = PAREN.match(t)
        if m:
            main_t, alt = clean(m.group(1)), clean(m.group(2))
            if main_t and alt:
                t, aliases = main_t, [alt]
                split_alias += 1
        entry = merged.setdefault(t, {
            "surface": t, "aliases": [], "english": r.get("english"),
            "tags": set(), "has_video": False,
            "source": "moe-signlanguage",
        })
        entry["aliases"] = sorted(set(entry["aliases"]) | set(aliases))
        if r.get("tags"):
            entry["tags"].add(r["tags"])
        if r.get("youtube_key"):
            entry["has_video"] = True

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for e in sorted(merged.values(), key=lambda x: x["surface"]):
            e["tags"] = sorted(e["tags"])
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    with_video = sum(1 for e in merged.values() if e["has_video"])
    print(f"清理後：{len(merged)} 個詞條（剔除 {dropped}、括號拆出別名 {split_alias}）")
    print(f"  有示範影片者：{with_video} ({with_video / len(merged) * 100:.1f}%)")
    print(f"輸出 → {out.relative_to(BASE)}")
    print("⚠️ 授權未查證；此表僅供 fallback 合法詞判定，勿當訓練資料（同形對應）")


if __name__ == "__main__":
    main()
