#!/usr/bin/env python3
"""建 sign_id 總表：把「動作庫實際演得出來的詞」編成穩定 ID。

為什麼需要這張表（2026-08-20，教授指定的訓練資料格式）：
    教授要求訓練資料改成「從候選清單挑 sign_id」的形式：

        {"sign_id":"TSL_TODAY","gloss":"今天"}

    但本專案至今是拿 **gloss 中文字串**當事實上的 ID，沒有穩定代號，
    也沒有一張「這個詞到底演不演得出來」的權威表。少了它，
    `asset_class: natural_playable` 只是一句空話。

錨定在哪裡：
    `data/video/lexicon.json` 是 0813 虛擬人動作庫的實際索引
    （gloss → 錄影檔＋起訖秒數），**它有的才播得出來**。
    所以 sign_id 一律以 lexicon 的鍵為準，不是以辭典詞條數為準
    —— 中正辭典有 3,500 詞不代表虛擬人演得出 3,500 詞。

ID 穩定性（重要）：
    ID 一旦發出去就會寫進訓練資料、模型權重與下游腳本，**永不可變**。
    故採 synthesize.py 的 SYN 編號同一原則：重跑時先讀回既有 ID
    原封不動，只有新出現的詞才接續編號。排序只影響首次建表。

用法：
    python scripts/build_sign_inventory.py            # 建表／增量更新
    python scripts/build_sign_inventory.py --check    # 只檢查不寫檔

輸出：
    data/signs/sign_inventory.jsonl   逐筆：sign_id / gloss / 來源 / 影片位置
    data/signs/gloss_to_sign.json     查詢索引：gloss → sign_id（含別名解析）
    data/signs/inventory_stats.json   組成統計，供報告引用
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_video_coverage import Library, norm  # noqa: E402  複用既有分級解析器

BASE = Path(__file__).resolve().parent.parent
LEXICON = BASE / "data" / "video" / "lexicon.json"
OUT_DIR = BASE / "data" / "signs"
INVENTORY = OUT_DIR / "sign_inventory.jsonl"
INDEX = OUT_DIR / "gloss_to_sign.json"
STATS = OUT_DIR / "inventory_stats.json"

ID_PREFIX = "TSL_"
ID_WIDTH = 5

# recording 檔名前綴 → 素材來源。來源會影響授權標註，必須逐筆帶著走。
SOURCE_PREFIX = {
    "moe": "moe_signdict",      # 教育部常用手語辭典（可訓練，資料不得公開）
    "moc": "moc_corpus",        # 文化部臺灣手語語料庫（訓練＋散布皆可，須標出處）
    "pn": "tsl_placenames",     # 台灣手語地名網
    "fn": "tsl_familynames",    # 台灣手語姓氏網
    "tasli": "tasli",
}


def source_of(recording: str) -> str:
    prefix = recording.split("_", 1)[0].split(".", 1)[0]
    return SOURCE_PREFIX.get(prefix, "other")


def load_existing() -> dict[str, str]:
    """讀回既有 gloss → sign_id 對應，確保 ID 永不變動。"""
    if not INVENTORY.exists():
        return {}
    mapping = {}
    with INVENTORY.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                mapping[row["gloss"]] = row["sign_id"]
    return mapping


def next_id_counter(existing: dict[str, str]) -> int:
    """從既有 ID 推出下一個編號，避免撞號。"""
    top = 0
    for sid in existing.values():
        m = re.fullmatch(rf"{ID_PREFIX}(\d+)", sid)
        if m:
            top = max(top, int(m.group(1)))
    return top + 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", type=Path, default=LEXICON)
    ap.add_argument("--check", action="store_true", help="只報告，不寫檔")
    args = ap.parse_args()

    if not args.lexicon.exists():
        print(f"找不到 {args.lexicon}\n"
              f"先抓下來：scp -P 2288 b310ai@163.13.202.125:"
              f"'~/0813/recordings/lexicon.json' data/video/", file=sys.stderr)
        return 1

    lexicon: dict[str, dict] = json.loads(args.lexicon.read_text(encoding="utf-8"))
    existing = load_existing()
    counter = next_id_counter(existing)
    print(f"動作庫 {len(lexicon)} 筆；既有 ID {len(existing)} 筆，下一號 {counter}")

    rows, new_count = [], 0
    for gloss in sorted(lexicon):                       # 排序只影響首次建表
        entry = lexicon[gloss]
        sign_id = existing.get(gloss)
        if sign_id is None:
            sign_id = f"{ID_PREFIX}{counter:0{ID_WIDTH}d}"
            counter += 1
            new_count += 1
        recording = entry.get("recording", "")
        start, end = entry.get("start"), entry.get("end")
        rows.append({
            "sign_id": sign_id,
            "gloss": gloss,
            "source": source_of(recording),
            "recording": recording,
            "start": start,
            "end": end,
            "duration": (round(end - start, 3)
                         if isinstance(start, (int, float)) and isinstance(end, (int, float))
                         else None),
            # 動作庫有這筆＝虛擬人演得出來。這是 asset_class 的唯一事實依據。
            "asset_class": "natural_playable",
        })

    # ---- 查詢索引：除了本名，也把 T1–T6 能解到同一支影片的寫法一併收錄 ----
    # 訓練與推論都要用這份索引把 gloss 轉成 sign_id，寫法不一致才不會漏。
    lib = Library(lexicon)
    by_gloss = {r["gloss"]: r["sign_id"] for r in rows}
    index: dict[str, str] = dict(by_gloss)
    alias_added = 0
    for gloss in list(by_gloss):
        n = norm(gloss)
        if n and n not in index:
            index[n] = by_gloss[gloss]
            alias_added += 1

    stats = {
        "lexicon_entries": len(lexicon),
        "signs": len(rows),
        "new_ids_this_run": new_count,
        "index_keys": len(index),
        "alias_keys_added": alias_added,
        "by_source": dict(collections.Counter(r["source"] for r in rows).most_common()),
        "by_gloss_len": dict(collections.Counter(
            min(len(r["gloss"]), 6) for r in rows).most_common()),
        "missing_timing": sum(1 for r in rows if r["duration"] is None),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.check:
        print("--check：未寫檔")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with INVENTORY.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"寫出 {INVENTORY}（{len(rows)}）／{INDEX}（{len(index)}）／{STATS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
