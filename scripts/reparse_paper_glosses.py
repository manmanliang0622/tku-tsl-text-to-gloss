#!/usr/bin/env python3
"""用修正後的解析器重算論文例句的 Gloss（不需要原始 PDF）。

背景：`extract_paper_examples.py` 的 `clean_gloss_line` 在 2026-08-13 修正了
「替代詞被串成同一個 Gloss 句子」的 bug（詳見該函式 docstring）。原始 .txt
（pdftotext 產出）已不在版控內，但每一筆都保留了 `raw_gloss_line`，重算
與重跑抽取等價——本腳本 dry-run 會先驗證「舊解析器 + 既有 raw_gloss_line」
能否重現既有 gloss_text，確認等價後才允許套用。

影響範圍僅止於論文來源，而論文例句**全部落在 test_papers**（train/dev 各
0 筆），故本次修正不動訓練資料、不需重訓、也不需重跑推論。

用法：
  python3 scripts/reparse_paper_glosses.py           # dry-run，只報告差異
  python3 scripts/reparse_paper_glosses.py --apply   # 寫回 papers/ 與 splits/
套用後需重跑：python3 scripts/build_json_targets.py --splits test_papers
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
from extract_paper_examples import clean_gloss_line, has_cjk  # noqa: E402

PAPERS = BASE / "data" / "papers"
SPLITS = BASE / "data" / "splits"


def clean_gloss_line_legacy(line):
    """修正前的版本，僅供 dry-run 驗證等價性用。"""
    s = unicodedata.normalize("NFKC", line).strip()
    if not s or not has_cjk(s):
        return None
    s = re.sub(r"[（(][^（）()]*[）)]", "", s)
    s = s.strip(" 。，、？?!！；;：:")
    if re.search(r"[A-Za-z]{3,}", s):
        return None
    s = s.replace("，", " ").replace(",", " ")
    toks = []
    for raw in s.split():
        t = raw.strip("，。、；：？?!！")
        if not t:
            continue
        t = t.split("/")[0]
        t = re.sub(r"\d+$", "", t)
        t = t.strip("＋+")
        if t and has_cjk(t):
            if not toks or toks[-1] != t:
                toks.append(t)
    return toks if len(toks) >= 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="寫回檔案（預設只報告）")
    args = ap.parse_args()

    # --- 1. 等價性驗證：舊解析器能否從 raw_gloss_line 重現既有 gloss_text ---
    rows = [json.loads(l) for l in (PAPERS / "paper_examples_all.jsonl").open()]
    mismatch = [r for r in rows
                if "/".join(clean_gloss_line_legacy(r["raw_gloss_line"]) or [])
                != r["gloss_text"]]
    print(f"等價性驗證：{len(rows) - len(mismatch)}/{len(rows)} 筆可由 raw_gloss_line 重現")
    if mismatch:
        print(f"  ⚠ {len(mismatch)} 筆無法重現，不可套用（raw_gloss_line 與產出不同源）")
        for r in mismatch[:5]:
            print(f"    {r['id']} {r['chinese']}")
        if args.apply:
            sys.exit(1)

    # --- 2. 用修正後的解析器重算 ---
    changes = {}          # id -> (old, new)
    dropped = []          # 修正後 token 數 < 2、無法成句者
    for r in rows:
        toks = clean_gloss_line(r["raw_gloss_line"])
        if not toks:
            dropped.append(r["id"])
            continue
        new = "/".join(toks)
        if new != r["gloss_text"]:
            changes[r["id"]] = (r["gloss_text"], new)

    tp_ids = {json.loads(l)["id"] for l in (SPLITS / "test_papers.jsonl").open()}
    in_test = [i for i in changes if i in tp_ids]
    print(f"重算後有變動：{len(changes)}/{len(rows)} 筆，其中 test_papers {len(in_test)}/{len(tp_ids)}")
    if dropped:
        print(f"  ⚠ 修正後不足兩個 Gloss、保留原值：{len(dropped)} 筆 {dropped[:5]}")

    for i in sorted(in_test)[:10]:
        old, new = changes[i]
        print(f"  {i}\n    舊 {old}\n    新 {new}")
    if len(in_test) > 10:
        print(f"  …另 {len(in_test) - 10} 筆")

    if not args.apply:
        print("\n（dry-run；加 --apply 寫回）")
        return

    # --- 3. 寫回 papers/ 兩檔與 splits/test_papers.jsonl ---
    for name in ("paper_examples_all.jsonl", "paper_examples.jsonl"):
        path = PAPERS / name
        out = []
        for line in path.open():
            r = json.loads(line)
            if r["id"] in changes:
                toks = clean_gloss_line(r["raw_gloss_line"])
                r["gloss"], r["gloss_text"] = toks, "/".join(toks)
            out.append(json.dumps(r, ensure_ascii=False))
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"OK → {path.relative_to(BASE)}")

    path = SPLITS / "test_papers.jsonl"
    out, n = [], 0
    for line in path.open():
        r = json.loads(line)
        if r["id"] in changes:
            r["gloss_text"] = changes[r["id"]][1]
            n += 1
        out.append(json.dumps(r, ensure_ascii=False))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"OK → {path.relative_to(BASE)}（更新 {n} 句參考答案）")
    print("\n接著跑：python3 scripts/build_json_targets.py --splits test_papers")


if __name__ == "__main__":
    main()
