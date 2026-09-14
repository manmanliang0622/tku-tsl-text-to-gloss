#!/usr/bin/env python3
"""把 quality_compare.py 判定「換新片」的鍵寫進 lexicon（寫法比照 ingest_new_videos.point()）。

    python3 apply_quality_compare.py incoming_20260914c --source "unknown-d:綠幕-2026-09-14"          # dry run
    python3 apply_quality_compare.py incoming_20260914c --source "unknown-d:綠幕-2026-09-14" --apply

- 只複製至少贏了一個鍵的片（_store/<rec>.mp4 與 _extracted/<rec>.json）；沒贏的留在批次目錄。
- 寫入的區段是 lexicon 區段（trim_span 切點），不是播放區段——composer 播放時自己會再切。
- 非主鍵（groups.json 裡 keys[1:]）加 alias_of＝主鍵。
- 不新增鍵，所以不必重啟模型服務。
"""
import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path.home() / "0813"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--source", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    batch = ROOT / args.batch
    rows = json.load(open(batch / "quality_compare.json", encoding="utf-8"))
    lex_path = ROOT / "recordings" / "lexicon.json"
    lex = json.load(open(lex_path, encoding="utf-8"))

    plan = []
    for r in rows:
        primary = r["keys"][0]["key"]
        for k in r["keys"]:
            if not k["take_new"]:
                continue
            if k["key"] not in lex:
                raise SystemExit(f"{k['key']} 不在 lexicon——這支只改既有鍵")
            plan.append((k["key"], primary, r, k))
    recs = sorted({r["rec"] for _, _, r, _ in plan})
    print(f"改指 {len(plan)} 個鍵，入庫 {len(recs)} 支片")
    for key, primary, r, k in plan:
        inc = k["incumbent"] or {}
        print(f"  {key:5s} {inc.get('recording', '?'):24s} -> {r['rec']}.json "
              f"{r['new']['lex_start']}-{r['new']['lex_end']}  {k['why']}")
    if not args.apply:
        print("\ndry-run only — 加 --apply 才寫入。")
        return

    ts = time.strftime("%Y%m%d-%H%M%S")
    bak = f"{lex_path}.bak-newvideos-{ts}"
    shutil.copy2(lex_path, bak)
    for rec in recs:
        for src, dst in ((batch / "_store" / f"{rec}.mp4", ROOT / "recordings" / f"{rec}.mp4"),
                         (batch / "_extracted" / f"{rec}.json", ROOT / "recordings" / f"{rec}.json")):
            if dst.exists() and dst.stat().st_size != src.stat().st_size:
                raise SystemExit(f"{dst} 已存在且內容不同，不覆蓋")
            shutil.copy2(src, dst)

    manifest = []
    for key, primary, r, k in plan:
        e = dict(lex[key])
        old_rec = e.get("recording")
        e.update({"recording": f"{r['rec']}.json", "start": r["new"]["lex_start"],
                  "end": r["new"]["lex_end"], "gloss": e.get("gloss", key), "source": args.source})
        if old_rec and old_rec != f"{r['rec']}.json":
            e.setdefault("replaced_from", old_rec)
            e.pop("system", None)
            e.pop("text", None)
        if key != primary:
            e["alias_of"] = primary
        lex[key] = e
        manifest.append({"key": key, "rec": r["rec"], "file": r["file"], "replaced": old_rec,
                         "why": k["why"], "new": r["new"], "incumbent": k["incumbent"]})
    lex_path.write_text(json.dumps(lex, ensure_ascii=False), encoding="utf-8")
    (batch / "ingest_manifest.json").write_text(json.dumps(
        {"applied_at": ts, "source": args.source, "backup": bak, "entries": manifest},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n寫入完成：{len(plan)} 鍵、{len(recs)} 支片，lexicon 共 {len(lex)} 鍵。備份 {bak}")


if __name__ == "__main__":
    main()
