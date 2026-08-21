#!/usr/bin/env python3
"""合成同義詞組表：同一個手語有多個中文詞面時，把它們串在一起。

**為什麼需要**：`data/signs/sign_inventory.jsonl` 是以「中文詞面」為單位編 ID 的，
但 16,624 個詞面背後只有 15,355 個不同動作——同一支影片被多個詞面共用。
候選檢索目前只認字面，句子寫「孩子」就撈不到動作庫裡叫「小孩」的那支，
即使兩者是同一個手語。

**實測效益（先講清楚，免得被高估）**：接上候選檢索後**是淨負的**——
train 詞涵蓋率 92.6%→90.7%（−1.9pp），dev 只有 +0.1pp，故該通道預設關閉
（`sign_candidates.candidates(n_syn=0)`）。原因是固定 k 之下候選是零和的，
塞進同義詞就得擠掉別的候選；種子放寬到例句／對齊通道再掉 0.7pp。
真正的量體在**評分端**：dev 有 34 個參考 token 自己不在候選裡、但同義詞在，
佔 1.20pp。不過其中只有 3 個來自 same_clip（按定義成立），31 個來自
dict_alias——那些詞在動作庫裡是**不同的影片**，判等於放寬正確性定義，
不應該預設採計。故本表只負責提供關係，採信到哪一層由下游決定。

**兩個來源，可信度不同，故分開標不合併**：

  same_clip   兩個詞面指向**同一支錄影的同一個時間區段**。按定義成立——
              播出來就是同一個動作，沒有詮釋空間。1,012 組。
  dict_alias  中正辭典 `twtsl_words.jsonl` 的 `aliases` 欄（813 個詞條）。
              較鬆：`成 → 完成／成功` 在中文裡不是全等，辭典的意思是
              「這幾個中文詞可以用同一個手語打」。教授指出的
              `今天 → 現在` 就在這一組。
  moe_alias   教育部 `moe_vocab_clean.jsonl` 的 aliases，只有 9 筆，聊備一格。

**不做遞移閉包**：A 同 B（same_clip）、B 同 C（dict_alias）不推出 A 同 C。
兩種關係的強度不同，串起來會把弱關係傳染給強關係。一個詞可以同時屬於多組，
下游自己決定要採信到哪一層。

用法：
    python3 scripts/build_synonym_groups.py            # 建表
    python3 scripts/build_synonym_groups.py --check    # 只報告不寫檔

輸出：data/signs/synonym_groups.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SIGNS = BASE / "data" / "signs"
INVENTORY = SIGNS / "sign_inventory.jsonl"
TWTSL_WORDS = BASE / "data" / "twtsl" / "twtsl_words.jsonl"
MOE_VOCAB = BASE / "data" / "moe" / "moe_vocab_clean.jsonl"
OUT = SIGNS / "synonym_groups.json"


def load_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def build_groups():
    rows = [r for r in load_jsonl(INVENTORY)
            if "duplicate_of" not in r and "superseded_by" not in r]
    in_library = {r["gloss_clean"] for r in rows}
    groups = []

    # ---- same_clip：同一支錄影同一區段 = 同一個動作 ----
    by_clip = collections.defaultdict(list)
    for r in rows:
        by_clip[(r["recording"], r["start"], r["end"])].append(r)
    for (rec, start, end), members in sorted(by_clip.items()):
        if len(members) < 2:
            continue
        members.sort(key=lambda r: r["sign_id"])
        groups.append({
            "id": f"SYN_C{len(groups) + 1:05d}",
            "source": "same_clip",
            "members": [m["gloss_clean"] for m in members],
            "sign_ids": [m["sign_id"] for m in members],
            "evidence": {"recording": rec, "start": start, "end": end},
        })
    n_clip = len(groups)

    # ---- dict_alias：中正辭典的別名欄（同一詞條可能重複出現，需去重）----
    seen_alias = set()
    for e in load_jsonl(TWTSL_WORDS):
        head = str(e.get("chinese") or "").strip()
        aliases = [str(a).strip() for a in (e.get("aliases") or []) if str(a).strip()]
        if not head or not aliases:
            continue
        key = (head, tuple(sorted(aliases)))
        if key in seen_alias:
            continue
        seen_alias.add(key)
        members = [head] + [a for a in aliases if a != head]
        groups.append({
            "id": f"SYN_D{len(groups) - n_clip + 1:05d}",
            "source": "dict_alias",
            "members": members,
            "in_library": [m for m in members if m in in_library],
            "evidence": {"dataset": "twtsl_words.jsonl", "headword": head},
        })
    n_dict = len(groups) - n_clip

    # ---- moe_alias ----
    for e in load_jsonl(MOE_VOCAB):
        head = str(e.get("surface") or "").strip()
        aliases = [str(a).strip() for a in (e.get("aliases") or []) if str(a).strip()]
        if not head or not aliases:
            continue
        members = [head] + [a for a in aliases if a != head]
        groups.append({
            "id": f"SYN_M{len(groups) - n_clip - n_dict + 1:05d}",
            "source": "moe_alias",
            "members": members,
            "in_library": [m for m in members if m in in_library],
            "evidence": {"dataset": "moe_vocab_clean.jsonl", "headword": head},
        })

    # ---- 查詢索引：詞面 → 所屬組 ----
    index = collections.defaultdict(list)
    for g in groups:
        for m in g["members"]:
            if g["id"] not in index[m]:
                index[m].append(g["id"])

    covered = {m for g in groups for m in g["members"] if m in in_library}
    stats = {
        "library_signs": len(rows),
        "distinct_clips": len(by_clip),
        "groups_total": len(groups),
        "groups_by_source": dict(collections.Counter(g["source"] for g in groups)),
        "library_signs_with_synonym": len(covered),
        "library_signs_with_synonym_pct": round(len(covered) / len(rows) * 100, 1),
        "index_keys": len(index),
    }
    return groups, dict(index), stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只報告，不寫檔")
    args = ap.parse_args()

    groups, index, stats = build_groups()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    for src in ("same_clip", "dict_alias", "moe_alias"):
        sample = [g for g in groups if g["source"] == src][:3]
        for g in sample:
            print(f"  [{src}] {g['id']}: {' / '.join(g['members'])}")

    if args.check:
        print("--check：未寫檔")
        return 0
    OUT.write_text(json.dumps(
        {"stats": stats, "groups": groups, "index": index},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"寫出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
