#!/usr/bin/env python3
"""把「影片待辦清單」的打勾列與下載回來的補片資料夾對起來，產出入庫清單。

補片這件事橫跨兩台機器：清單與比對在本機（xlsx 在這裡、lexicon 鏡像也在
這裡），抽 landmark 與寫入動作庫在學校主機。這支負責前半段，輸出的
`plan.json` 送上主機給 `ingest_new_videos.py` 吃。

一支影片一列，欄位對應關係是這樣決定的：

  檔名就是詞      `工作.mp4` → 詞條鍵「工作」。多數情況如此。
  找「X」         備註寫「找『敵對』」時，那支片的檔名是 `敵對.mp4`，
                  「敵人」這個鍵要一起改指過去（aliases 欄）。
                  **但如果「敵人」自己也有一支片，以自己那支為準**——
                  這件事在 ingest 端判，因為要等品質分數出來才知道。
  異體字          `猫/悠閑/游戲` 是手寫檔名時打成的異體或簡體，動作庫的鍵是
                  `貓/悠閒/遊戲`。這張表刻意用手列的：離線沒有可靠的簡繁轉換，
                  自動轉會把「余/餘」「面/麵」這種一對多轉錯，寧可漏掉讓它
                  出現在「對不到清單」裡由人補。
  同一詞多支      同一個詞在不同資料夾各有一支（`資料` 就是）時，第二支起
                  編成 `_B`、`_C`。哪一支進主鍵由品質分數決定，不在這裡決定。

備註欄裡的同義關係（找「X」、等於「A」「B」）另有出口：那是語言知識不是
入庫指令，走 `data/signs/synonym_manual.jsonl`，見 build_synonym_groups.py。

用法：

    python3 scripts/make_video_ingest_plan.py \\
        --xlsx ~/Desktop/影片待辦清單＿1.xlsx \\
        --videos ~/Downloads/缺片-該錄 ~/Downloads/最佳poor \\
        --batch incoming_20260901

    scp plan.json tku-gpu:'~/0813/incoming_20260901/'

`--batch` 是主機上暫存資料夾的名字，只用來組 `src` 的相對路徑
（`<batch>/<資料夾名>/<檔名>`），影片本身用 rsync 另外傳。

需要 openpyxl。macOS 的 /usr/bin/python3 自帶。
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
LEXICON = BASE / "data" / "video" / "lexicon.json"
ENTRIES = BASE / "data" / "video" / "entries_final.csv"

# 手寫檔名的異體/簡體 → 動作庫實際的鍵。逐筆查證過，不要改成自動轉換。
VARIANTS = {"猫": "貓", "悠閑": "悠閒", "游戲": "遊戲"}

QUOTED = re.compile(r"[「『]([^」』]+)[」』]")
MARK_COLS = ("【重找了嗎】", "【處理了嗎】")


def parse_note(note) -> tuple[str | None, list[str], str]:
    """備註 → (找的目標, 等於的詞們, 原文)。

    體例是逗號分段、每段一個關鍵詞：「找『紀錄』，等於『記』」兩種都有。
    """
    if not note:
        return None, [], ""
    text = str(note).strip()
    found, equals = None, []
    for part in re.split(r"[，,]\s*", text):
        part = part.strip()
        if part.startswith("找"):
            hits = QUOTED.findall(part)
            if hits:
                found = hits[0]
        elif part.startswith("等於"):
            equals += QUOTED.findall(part)
    return found, equals, text


def read_sheet(xlsx: Path) -> list[dict]:
    import openpyxl

    wb = openpyxl.load_workbook(xlsx)
    rows = []
    for ws in wb.worksheets:
        table = list(ws.iter_rows(values_only=True))
        header_i = next((i for i, r in enumerate(table) if r and r[0] == "詞"), None)
        if header_i is None:      # 說明分頁沒有表頭
            continue
        header = [c for c in table[header_i] if c]
        for raw in table[header_i + 1:]:
            if not raw or not raw[0]:
                continue
            d = dict(zip(header, raw))
            mark = next((d.get(c) for c in MARK_COLS if d.get(c)), "")
            found, equals, note = parse_note(d.get("【備註】"))
            rows.append({
                "sheet": ws.title,
                "word": str(d["詞"]).strip(),
                "mark": str(mark).strip(),
                "note": note,
                "find": found,
                "equals": equals,
                "usage": d.get("我方次數"),
                "eval_sets": d.get("影響評測集") or "",
                "sheet_tier": d.get("目前最佳 tier"),
            })
    return rows


def collect_videos(dirs: list[Path]) -> dict[str, list[tuple[str, str]]]:
    """詞 → [(資料夾名, 檔名)]，檔名的異體字已正規化成動作庫的鍵。"""
    by_word: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for d in dirs:
        for f in sorted(p.name for p in d.iterdir() if p.suffix.lower() == ".mp4"):
            stem = Path(f).stem
            by_word[VARIANTS.get(stem, stem)].append((d.name, f))
    return by_word


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", required=True, type=Path, help="影片待辦清單 xlsx")
    ap.add_argument("--videos", required=True, nargs="+", type=Path, help="補片資料夾")
    ap.add_argument("--batch", required=True, help="主機上暫存資料夾名，例：incoming_20260901")
    ap.add_argument("--prefix", default="twtsl2", help="入庫檔名前綴")
    ap.add_argument("--lexicon", type=Path, default=LEXICON)
    ap.add_argument("--entries", type=Path, default=ENTRIES,
                    help="quality_scan/entries_final.csv 的鏡像；沒有就不填 old_tier")
    ap.add_argument("--out", type=Path, default=Path("plan.json"))
    args = ap.parse_args()

    rows = read_sheet(args.xlsx)
    by_word = collect_videos([d.expanduser() for d in args.videos])
    lex = json.loads(args.lexicon.read_text(encoding="utf-8"))
    tiers = {}
    if args.entries.is_file():
        tiers = {r["label"]: r.get("tier") for r in csv.DictReader(
            args.entries.open(encoding="utf-8"))}

    by_row = {r["word"]: r for r in rows}
    # 備註寫「找『X』」而 X 真的有片時，那個詞跟著改指
    aliases: dict[str, list[str]] = collections.defaultdict(list)
    for r in rows:
        if r["mark"] and r["find"] and r["find"] != r["word"] and r["find"] in by_word:
            aliases[r["find"]].append(r["word"])

    plan = []
    for word in sorted(by_word):
        for i, (dirname, filename) in enumerate(by_word[word]):
            suffix = "" if i == 0 else f"_{chr(ord('B') + i - 1)}"
            row = by_row.get(word, {})
            stem = Path(filename).stem
            plan.append({
                "word": word,
                "rec": f"{args.prefix}_{word}{suffix}",
                "dup_index": i,
                "src": f"{args.batch}/{dirname}/{filename}",
                "filename_orig": stem if stem != word else None,
                "in_lexicon": word in lex,
                "old_recording": lex.get(word, {}).get("recording"),
                "old_source": lex.get(word, {}).get("source"),
                "old_tier": tiers.get(word),
                "sheet": row.get("sheet"),
                "sheet_tier": row.get("sheet_tier"),
                "usage": row.get("usage"),
                "evalsets": row.get("eval_sets", ""),
                "note": row.get("note"),
                "aliases": aliases.get(word, []),
            })

    args.out.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")

    marked = [r for r in rows if r["mark"]]
    served = {p["word"] for p in plan}
    missing = [r for r in marked
               if r["word"] not in served and (r["find"] or "") not in served]
    unclaimed = sorted(set(by_word) - {r["word"] for r in rows}
                       - {r["find"] for r in rows if r["find"]})
    print(f"{args.out}：{len(plan)} 支影片、{len(by_word)} 個詞")
    print(f"  新增鍵 {sum(1 for p in plan if not p['in_lexicon'])}、"
          f"別名改指 {sum(len(p['aliases']) for p in plan)}、"
          f"同詞多支 {sum(1 for p in plan if p['dup_index'])}")
    print(f"清單打勾 {len(marked)} 列，其中 {len(missing)} 列這批沒有對應影片")
    if unclaimed:
        print(f"⚠ 對不到任何清單列的檔名（可能是異體字或多找的）：{unclaimed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
