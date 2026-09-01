#!/usr/bin/env python3
"""把手工找回來的補片入庫：抽 landmark → 量品質 → 換掉 lexicon 指向。

**這支在學校主機上跑**（動作庫與 MediaPipe 都在那裡），放在 repo 只是為了
版本控制與交接；用法是複製到 `~/0813/` 底下與 `clip_metrics.py` 同一層。
跟 `apply_gap_fill.py` 的差別：那支是把詞改指到庫裡**既有**的語料庫片段，
這支是**新素材進庫**，recordings/ 會多出 mp4 與 sidecar。

清單 `<root>/<batch>/plan.json` 由本機的 `make_video_ingest_plan.py` 產生，
一筆一支影片：

    word            這支片打的是哪個詞（動作庫的鍵）
    rec             入庫後的檔名前綴
    src             <batch>/ 底下的相對路徑
    aliases         備註寫「找『X』」的那些詞，一起改指到這支
    old_recording   目前這個鍵指向誰

## 兩段式品質關

新片抽完 landmark 才知道好不好，分兩段收：

  預設        只收 ok 級（act_eff >= 0.60 且 jit <= 0.05）。
  --allow-upgrade
              再收「不到 ok、但比現用那支好 0.05 以上、且 act >= 0.30」的。
              「重找」清單上的詞現用片多半 act 0.0–0.3（整段抓不到手），
              一律卡在 0.60 等於讓那個詞繼續維持不能看；0.30 以下不收，
              換了也還是抓不到手。2026-09-01 首批 439 支：365 + 25 = 390。

**新舊比較必須同口徑**：`entries_final.csv` 的 tier 是 scan_quality.py 算的，
跟這裡不同，直接比會比錯。所以舊片也用 `clip_metrics.metrics()` 重算一次
（結果快取在 `<batch>/old_metrics.json`）。口徑本身的兩個坑見 clip_metrics
的 docstring——踩過一次，29 支好片被誤判成 severe。

    python3 ingest_new_videos.py --batch incoming_20260901 --extract   # 抽 landmark
    python3 ingest_new_videos.py --batch incoming_20260901             # dry run
    python3 ingest_new_videos.py --batch incoming_20260901 --apply     # 寫入

`--extract` 要在有 mediapipe 的環境跑（`cd ~/0813/signavatar-core &&
uv run python ~/0813/ingest_new_videos.py …`），其餘步驟系統 python3 即可。
可續跑：已經抽過的會跳過。

備份：`recordings/lexicon.json.bak-newvideos-<ts>`；入庫紀錄寫進
`<batch>/ingest_manifest.json`。

**寫完別忘了同步線上那份**——`~/0821_bundle/recordings/` 是實體複本不是
symlink，不同步的話線上播的還是舊片：

    rsync -a --include="twtsl2_*" --include="lexicon.json" --exclude="*" \\
        ~/0813/recordings/ ~/0821_bundle/recordings/

新增鍵還要重啟模型服務（可播放詞表是啟動時載入的）。
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip_metrics import metrics  # noqa: E402

SOURCE_TAG = "twtsl:rescan-2026-09"


def _extract_one(src: str, out: str, label: str):
    from signavatar.capture.extractor import extract
    rec = extract(src, out, label=label)
    return out, (len(rec.frames) if rec else 0)


def do_extract(plan, paths, workers: int) -> None:
    paths["ext"].mkdir(parents=True, exist_ok=True)
    jobs = [(str(paths["root"] / p["src"]), str(paths["ext"] / f"{p['rec']}.json"), p["word"])
            for p in plan if not (paths["ext"] / f"{p['rec']}.json").is_file()]
    print(f"{len(plan)} clips in plan, {len(plan) - len(jobs)} already extracted, "
          f"{len(jobs)} to go ({workers} workers)", flush=True)
    if not jobs:
        return
    t0 = time.time()
    done = fail = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_extract_one, *j): j[1] for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            name = Path(futs[fut]).name
            try:
                _, n = fut.result()
                done += 1
                msg = f"{n} frames"
            except Exception as ex:
                fail += 1
                msg = f"FAILED: {type(ex).__name__}: {ex}"
            if i % 10 == 0 or fail:
                el = time.time() - t0
                print(f"[{i}/{len(jobs)}] {name} {msg} "
                      f"({el/i:.1f}s/clip, ~{(len(jobs)-i)*el/i/60:.0f} min left)", flush=True)
    print(f"extracted {done}, failed {fail}, {(time.time()-t0)/60:.1f} min", flush=True)


def load_metrics(plan, paths, refresh: bool) -> dict:
    f = paths["metrics"]
    if f.is_file() and not refresh:
        return json.loads(f.read_text(encoding="utf-8"))
    out = {}
    for p in plan:
        sidecar = paths["ext"] / f"{p['rec']}.json"
        if not sidecar.is_file():
            continue
        try:
            out[p["rec"]] = metrics(sidecar)
        except Exception as ex:
            out[p["rec"]] = {"error": f"{type(ex).__name__}: {ex}"}
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def load_old_metrics(plan, lex, paths, refresh: bool) -> dict:
    """這些詞現在指向的片段，用同一支指標重算一次。"""
    f = paths["old_metrics"]
    if f.is_file() and not refresh:
        return json.loads(f.read_text(encoding="utf-8"))
    out, cache = {}, {}
    words = sorted({p["word"] for p in plan if p["word"] in lex})
    for i, w in enumerate(words, 1):
        e = lex[w]
        rec, s0, s1 = e.get("recording"), e.get("start"), e.get("end")
        if not rec or not (paths["rec"] / rec).is_file():
            continue
        key = (rec, s0, s1)
        if key not in cache:
            try:
                cache[key] = metrics(paths["rec"] / rec, s0, s1)
            except Exception as ex:
                cache[key] = {"error": f"{type(ex).__name__}: {ex}"}
        out[w] = dict(cache[key], recording=rec)
        if i % 50 == 0:
            print(f"  舊片重算 {i}/{len(words)}", flush=True)
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path(os.environ.get("TSL_0813_ROOT", "~/0813")).expanduser(),
                    help="動作庫根目錄（含 recordings/），預設 ~/0813")
    ap.add_argument("--batch", default="incoming_20260901",
                    help="root 底下的暫存資料夾名（含 plan.json 與影片）")
    ap.add_argument("--extract", action="store_true", help="只抽 landmark")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--apply", action="store_true", help="寫入 recordings/ 與 lexicon")
    ap.add_argument("--allow-upgrade", action="store_true",
                    help="不到 ok 級、但比現用片好的也換（見 --min-upgrade-act）")
    ap.add_argument("--min-upgrade-act", type=float, default=0.30,
                    help="升級模式的地板：低於此換了也還是抓不到手，不換")
    ap.add_argument("--refresh-metrics", action="store_true")
    args = ap.parse_args()

    root = args.root.expanduser()
    inc = root / args.batch
    paths = {
        "root": root, "inc": inc, "ext": inc / "_extracted",
        "rec": root / "recordings", "lex": root / "recordings" / "lexicon.json",
        "metrics": inc / "metrics.json", "old_metrics": inc / "old_metrics.json",
        "manifest": inc / "ingest_manifest.json",
    }
    plan = json.loads((inc / "plan.json").read_text(encoding="utf-8"))

    if args.extract:
        do_extract(plan, paths, args.workers)
        return 0

    m = load_metrics(plan, paths, args.refresh_metrics)
    missing = [p["rec"] for p in plan if p["rec"] not in m]
    if missing:
        print(f"⚠ {len(missing)} 支還沒抽 landmark，先跑 --extract：{missing[:5]}")

    lex = json.loads(paths["lex"].read_text(encoding="utf-8"))
    old = load_old_metrics(plan, lex, paths, args.refresh_metrics)

    keep, drop, nofile = [], [], []
    for p in plan:
        q = m.get(p["rec"])
        if not q or "act_eff" not in q:
            nofile.append(p); continue
        p["_m"] = q
        om = old.get(p["word"])
        p["_old_m"] = om if om and "act_eff" in om else None
        oa = p["_old_m"]["act_eff"] if p["_old_m"] else None
        p["_upgrade"] = (args.allow_upgrade and q["tier"] != "ok"
                         and q["act_eff"] >= args.min_upgrade_act
                         and (oa is None or q["act_eff"] > oa + 0.05))
        (keep if (q["tier"] == "ok" or p["_upgrade"]) else drop).append(p)

    # 同一個詞有多支候選時只取最好的那支進主鍵，其餘入庫成變體鍵，不丟掉。
    best = {}
    for p in keep:
        cur = best.get(p["word"])
        if cur is None or p["_m"]["act_eff"] > cur["_m"]["act_eff"]:
            best[p["word"]] = p

    print(f"\n品質分佈（{len(keep)+len(drop)} 支抽好的）：",
          dict(collections.Counter(p["_m"]["tier"] for p in keep + drop)))
    print("  在打的手：", dict(collections.Counter(p["_m"]["hands"] for p in keep + drop)))
    print(f"過關 {len(keep)}、未過 {len(drop)}、沒抽到 {len(nofile)}")
    print(f"  其中「升級」（不到 ok 但勝過現用片）: {sum(1 for p in keep if p['_upgrade'])}")

    writes, alias_writes, variants = [], [], []
    for p in keep:
        if best.get(p["word"], p)["rec"] != p["rec"]:
            variants.append(p)
            continue
        writes.append(p)
        for a in p["aliases"]:
            alias_writes.append((a, p))
    # 別名的那個詞自己也有一支過關的補片時，用自己那支：檔名寫著那個詞是最直接
    # 的證據，也免得那支片入了庫卻沒有任何鍵指向它。（竹／破產／轉變 兩邊都有片。）
    own_ok = {p["word"] for p in writes}
    alias_writes = [(a, p) for a, p in alias_writes if a not in own_ok]

    print(f"主鍵更新 {len(writes)}（其中新增鍵 "
          f"{sum(1 for p in writes if not p['in_lexicon'])}）、"
          f"別名改指 {len(alias_writes)}、同詞變體另存 {len(variants)}")

    print("\n前 25 筆：")
    for p in writes[:25]:
        q, o = p["_m"], p["_old_m"]
        tag = "升級" if p["_upgrade"] else ("新增" if not p["in_lexicon"] else "更新")
        if o:
            print(f"  {tag} {p['word']:10s} {str(p['old_recording'])[:24]:24s} -> "
                  f"{p['rec']}.json {q['start']}-{q['end']} "
                  f"(act {o['act_eff']:.2f}->{q['act_eff']:.2f})")
        else:
            print(f"  {tag} {p['word']:10s} {'(新鍵)':24s} -> "
                  f"{p['rec']}.json {q['start']}-{q['end']} (act={q['act_eff']:.2f})")

    worse = [p for p in writes if p["_old_m"]
             and p["_old_m"]["act_eff"] - p["_m"]["act_eff"] > 0.25]
    if worse:
        print(f"\n⚠ 過關但比舊片明顯差（act_eff 掉超過 0.25），值得人工過目 {len(worse)} 支：")
        for p in sorted(worse, key=lambda p: p["_m"]["act_eff"])[:20]:
            print(f"  {p['word']:10s} 舊 {p['_old_m']['act_eff']:.2f} "
                  f"({p['old_recording']}) -> 新 {p['_m']['act_eff']:.2f}  usage={p['usage']}")

    if drop:
        print("\n未過品質關（不寫入，保留原狀）：")
        for p in sorted(drop, key=lambda p: p["_m"]["act_eff"])[:40]:
            q, o = p["_m"], p["_old_m"]
            tail = f"| 舊 act={o['act_eff']:.2f} {o['tier']}" if o else "| 舊 無"
            print(f"  {p['word']:10s} 新 act={q['act_eff']:.2f} {q['tier']:6s} "
                  f"手={q['hands']:4s} {tail} usage={p['usage']}")
        if len(drop) > 40:
            print(f"  … 其餘 {len(drop)-40} 支見 metrics.json")

    if not args.apply:
        print("\ndry-run only — 加 --apply 才會寫入。")
        return 0

    ts = time.strftime("%Y%m%d-%H%M%S")
    bak = str(paths["lex"]) + f".bak-newvideos-{ts}"
    shutil.copy2(paths["lex"], bak)
    manifest = []
    for p in writes + variants:
        for src, dst in ((root / p["src"], paths["rec"] / f"{p['rec']}.mp4"),
                         (paths["ext"] / f"{p['rec']}.json", paths["rec"] / f"{p['rec']}.json")):
            if dst.is_file() and dst.stat().st_size == src.stat().st_size:
                continue      # 重跑時不必再複製一次 2 GB
            shutil.copy2(src, dst)
        manifest.append({"rec": p["rec"], "word": p["word"], "src": p["src"],
                         "metrics": p["_m"], "role": "variant" if p in variants else "primary",
                         "upgrade": p["_upgrade"], "replaced": p["old_recording"],
                         "old_tier": p["old_tier"]})

    def point(key, p, extra=None):
        q = p["_m"]
        e = dict(lex.get(key) or {})
        old_rec = e.get("recording")
        e.update({"recording": f"{p['rec']}.json", "start": q["start"], "end": q["end"],
                  "gloss": e.get("gloss", key), "source": SOURCE_TAG})
        if old_rec and old_rec != f"{p['rec']}.json":
            # 保留最初的出處：重跑不可以把它蓋成上一輪的補片
            e.setdefault("replaced_from", old_rec)
        if extra:
            e.update(extra)
        lex[key] = e

    for p in writes:
        point(p["word"], p)
    for a, p in alias_writes:
        point(a, p, {"gloss": a, "alias_of": p["word"]})
    for p in variants:
        point(f"{p['word']}_{p['rec'].rsplit('_', 1)[-1]}", p)

    paths["lex"].write_text(json.dumps(lex, ensure_ascii=False), encoding="utf-8")
    paths["manifest"].write_text(json.dumps(
        {"applied_at": ts, "source": SOURCE_TAG, "backup": bak,
         "allow_upgrade": args.allow_upgrade, "min_upgrade_act": args.min_upgrade_act,
         "entries": manifest}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n寫入完成：{len(writes)} 主鍵 + {len(alias_writes)} 別名 + {len(variants)} 變體，"
          f"lexicon 共 {len(lex)} 鍵。備份 {bak}")
    print("記得同步 ~/0821_bundle/recordings/（見 docstring），新增鍵還要重啟模型服務。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
