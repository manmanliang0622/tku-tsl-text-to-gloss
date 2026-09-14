#!/usr/bin/env python3
"""補片頭尾靜止段（手垂著／擱在腿上，但手仍被偵測到）auto span 切不掉，
用手腕高度找真正在打的區段再重算 metrics。口徑同 2026-09-08 批：
手腕 y（相對肩中點、肩寬為單位，向下為正）比整段第 90 百分位高 0.30 以上、
且絕對值 < 1.25 的最長區段（容忍 0.2 秒斷點），前後各留 0.15 秒。"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home() / "0813"))
from clip_metrics import _arrays, metrics

def trimmed_span(path):
    d = json.load(open(path, encoding="utf-8"))
    a, ts, fps, n = _arrays(d)
    ys = []
    for i in range(n):
        cand = [a["wrist_" + s][i][1] for s in ("L", "R")
                if a["vis_" + s][i] > 0.5 and not np.isnan(a["wrist_" + s][i][1])]
        ys.append(min(cand) if cand else np.nan)
    ys = np.array(ys)
    ok = ~np.isnan(ys)
    if ok.sum() < 5:
        return None
    p90 = np.nanpercentile(ys, 90)
    act = ok & (ys < p90 - 0.30) & (ys < 1.25)
    idx = np.flatnonzero(act)
    if not len(idx):
        return None
    gap = max(1, int(round(0.2 * (fps or 30))))
    runs, s0 = [], idx[0]
    for p, q in zip(idx, idx[1:]):
        if q - p > gap:
            runs.append((s0, p)); s0 = q
    runs.append((s0, idx[-1]))
    i0, i1 = max(runs, key=lambda r: r[1] - r[0])
    return (round(max(0.0, float(ts[i0]) - 0.15), 3),
            round(min(float(ts[-1]), float(ts[i1]) + 0.15), 3),
            round(float(p90), 3))

batch = Path.home() / "0813" / sys.argv[1]
plan = json.load(open(batch / "plan.json"))
new = json.load(open(batch / "metrics.json"))
old = json.load(open(batch / "old_metrics.json"))
rows = []
for p in plan:
    rec = p["rec"]; sc = batch / "_extracted" / f"{rec}.json"
    auto = new[rec]; o = old.get(p["word"])
    tr = trimmed_span(sc)
    t = metrics(sc, tr[0], tr[1]) if tr else None
    rows.append({"word": p["word"], "rec": rec, "old": o, "auto": auto, "trim": t,
                 "trim_p90": tr[2] if tr else None})
json.dump(rows, open(batch / "compare.json", "w"), ensure_ascii=False, indent=1)
print(f"{'word':6s} | {'舊 act jit  tier   dur':24s} | {'新auto act tier  span':22s} | 新trim act jit tier span hands")
for r in rows:
    o, a, t = r["old"], r["auto"], r["trim"]
    os_ = f"{o['act_eff']:.2f} {o['jit'] or 0:.3f} {o['tier']:6s} {o['dur']:.2f}s" if o else "(無)"
    as_ = f"{a['act_eff']:.2f} {a['tier']:6s} {a['start']}-{a['end']}"
    ts_ = (f"{t['act_eff']:.2f} {t['jit'] or 0:.3f} {t['tier']:6s} {t['start']}-{t['end']} {t['hands']}"
           if t else "(找不到)")
    print(f"{r['word']:6s} | {os_:24s} | {as_:22s} | {ts_}")
