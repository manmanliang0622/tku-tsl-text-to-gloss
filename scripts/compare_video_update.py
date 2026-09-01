#!/usr/bin/env python3
"""影片庫更新前後的比較：這批換片到底改善了什麼。

動機：0813 的動作庫會持續補片與換源（2026-09-01 一次換掉 621 支、新增 28 支）。
每次更新都要回答同樣三個問題，手動查很煩也容易漏：

  1. 換掉的那些，品質是變好還是變壞？
  2. 缺口清單（待重錄／庫裡沒有）被打中多少？
  3. 硬編的判定表（DUPLICATE_OF／SUPERSEDED_BY／BOTH_UNUSABLE）有沒有過期？

⚠️ **品質比較需要重跑過的掃描表。** 換片之後如果 entries_final.csv 還是舊的，
新錄影一律查無資料，這支會據實說「無品質資料」而不是假裝沒變。

用法：
    # 先把新的掃描表拉過來
    scp tku-gpu:'~/0813/quality_scan/entries_final.csv' data/video/
    python3 scripts/compare_video_update.py

    # 與指定的舊總表比較（預設拿 git HEAD 的版本）
    python3 scripts/compare_video_update.py --before /path/to/old_inventory.jsonl
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))

LEXICON = BASE / "data" / "video" / "lexicon.json"
QUALITY = BASE / "data" / "video" / "entries_final.csv"
INVENTORY = BASE / "data" / "signs" / "sign_inventory.jsonl"


def load_before(path: Path | None) -> dict:
    """舊總表：預設取 git HEAD 那版（＝更新前的狀態）。"""
    if path:
        text = path.read_text(encoding="utf-8")
    else:
        text = subprocess.run(
            ["git", "-C", str(BASE), "show", "HEAD:data/signs/sign_inventory.jsonl"],
            capture_output=True, text=True).stdout
    return {r["gloss"]: r for r in (json.loads(l) for l in text.splitlines() if l.strip())}


def load_quality() -> dict:
    if not QUALITY.exists():
        return {}
    with QUALITY.open(encoding="utf-8") as fh:
        return {(r["label"], r["recording"]): r for r in csv.DictReader(fh)}


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def motion_sec(r) -> float | None:
    a, d = _f(r.get("act_max")), _f(r.get("dur"))
    return None if a is None or d is None else round(a * d, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=Path, default=None,
                    help="更新前的 sign_inventory.jsonl（預設用 git HEAD 那版）")
    args = ap.parse_args()

    lex = json.loads(LEXICON.read_text(encoding="utf-8"))
    before = load_before(args.before)
    q = load_quality()

    changed = {g for g in lex if g in before
               and lex[g].get("recording") != before[g].get("recording")}
    added = {g for g in lex if g not in before}
    removed = {g for g in before if g not in lex}

    print("=" * 66)
    print(f"動作庫 {len(before)} → {len(lex)} 筆")
    print(f"  換源 {len(changed)}／新增 {len(added)}／消失 {len(removed)}")
    if added:
        srcs = collections.Counter((lex[g].get("source") or "?").split(":")[0] for g in added)
        print(f"  新增來源：{dict(srcs.most_common(5))}")

    # ── 1. 品質變化 ────────────────────────────────────────────────
    print("\n[品質變化]")
    if not q:
        print(f"  ⚠ 找不到 {QUALITY.name}，無法比較")
    else:
        stale = [g for g in (changed | added) if (g, lex[g]["recording"]) not in q]
        if stale:
            print(f"  ⚠ {len(stale)}/{len(changed | added)} 筆新錄影**沒有品質資料**"
                  f"——掃描表比動作庫舊，請重跑掃描後再看這一節。")
        pairs = []
        for g in changed:
            old = q.get((g, before[g]["recording"]))
            new = q.get((g, lex[g]["recording"]))
            if old and new:
                pairs.append((g, old, new))
        if not pairs:
            print("  （沒有前後都有品質資料的項目，無法比較）")
        else:
            trans = collections.Counter((o["tier"], n["tier"]) for _, o, n in pairs)
            better = sum(v for (a, b), v in trans.items()
                         if _rank(b) > _rank(a))
            worse = sum(v for (a, b), v in trans.items() if _rank(b) < _rank(a))
            same = len(pairs) - better - worse
            print(f"  可比對 {len(pairs)} 筆：變好 {better}／持平 {same}／變差 {worse}")
            for (a, b), v in trans.most_common(8):
                if a != b:
                    print(f"    {a} → {b}: {v}")
            ms = [(motion_sec(o), motion_sec(n)) for _, o, n in pairs]
            ms = [(a, b) for a, b in ms if a is not None and b is not None]
            if ms:
                import statistics as st
                print(f"  動作秒數中位：{st.median(a for a, _ in ms):.2f}s → "
                      f"{st.median(b for _, b in ms):.2f}s")

    # ── 2. 缺口清單命中 ────────────────────────────────────────────
    print("\n[缺口清單命中]")
    touched = changed | added
    for name in ("待重錄_動作過短.csv", "true_gaps_no_alias.csv"):
        path = BASE / "data" / "video" / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        key = "詞" if rows and "詞" in rows[0] else "base"
        hit = [r for r in rows if r.get(key) in touched]
        extra = ""
        if hit and "總次數" in hit[0]:
            extra = f"，共 {sum(int(r['總次數']) for r in hit)} 次"
        print(f"  {name}: {len(rows)} 詞 → 命中 {len(hit)}{extra}")

    # ── 3. 硬編判定表是否過期 ──────────────────────────────────────
    print("\n[硬編判定表]")
    import build_sign_inventory as bsi

    def rec(g):
        return (lex.get(g) or {}).get("recording")

    stale_n = 0
    for label, table in (("DUPLICATE_OF", bsi.DUPLICATE_OF),
                         ("SUPERSEDED_BY", bsi.SUPERSEDED_BY)):
        for k, (other, r1, r2) in table.items():
            if {rec(k), rec(other)} != {r1, r2}:
                stale_n += 1
                print(f"  ✗ {label}[{k!r}]：當初 {r1}/{r2}，現在 {rec(k)}/{rec(other)}")
                up = (lex.get(other) or {}).get("replaced_from")
                if up:
                    print(f"      上游記錄 replaced_from={up}")
    for g in bsi.BOTH_UNUSABLE:
        hits = [k for k in lex if bsi.clean_gloss(k) == g]
        recs = [rec(h) for h in hits]
        tiers = {r: (q.get((h, r)) or {}).get("tier") for h, r in zip(hits, recs)}
        if any(t not in (None, "severe", "no_hands_raised") for t in tiers.values()):
            stale_n += 1
            print(f"  ✗ BOTH_UNUSABLE[{g!r}]：掃描結果 {tiers}")
    if not stale_n:
        print("  ✓ 全部仍然成立")
    else:
        print(f"  → {stale_n} 組需人工複核後更新（build_sign_inventory.py）")
        print("     重建總表會被守衛擋下，這是刻意的：判定的前提變了就不能沿用。")
    return 0


_TIER_ORDER = {"no_hands_raised": 0, "severe": 1, "poor": 2, "ok": 3}


def _rank(t):
    return _TIER_ORDER.get(t, -1)


if __name__ == "__main__":
    sys.exit(main())
