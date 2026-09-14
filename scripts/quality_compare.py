#!/usr/bin/env python3
"""新片 vs 庫裡現用片：在「虛擬人實際會播的區段」上同口徑逐鍵比，決定換不換。

composer.js 播每個詞之前依序做 repairHandedness（用姿態手腕重新指派左右手）、
trimRestPosture、trimNeutral，這裡照同樣順序移植，指標量的是虛擬人實際拿到的東西。

為什麼要算「播放區段」：composer.js 播每個詞之前會先切頭尾——
trimRestPosture（手垂在身側、正在往上抬的段）再 trimNeutral（頭尾靜止段）。
辭典／自錄片的 lexicon 區段裡常含抬手段，那段手還沒完全入鏡、偵測率低，
但虛擬人根本不會播；語料庫句中切段則是 no-op。不先切掉，act 會系統性地
偏袒 0.3–0.5 秒的句中切段。這裡把兩個函式照 composer.js 移植（門檻相同）。

比的指標（全部在播放區段上）：
  act_eff / tier   clip_metrics.metrics()，口徑不變
  ajit             手指關節角抖動（度）：hand world_landmarks（3D、公尺，不受
                   手掌朝向造成的投影縮短影響）算 15 個指節彎曲角，降到約 30fps，
                   5 幀高通 RMS。虛擬人的手指就是照錄影裡的關節角去彎的。
  hand_px          手掌長（腕→中指 MCP）中位數，原始像素，landmark 解析度參考。

判準：
  1. act_eff 差 >= 0.05：高的贏。
  2. 平手（差 < 0.05）：挑戰者非 ok 而現任 ok → 不換；否則 ajit 低的贏
     （差不到 10% 時比 hand_px）。
每個鍵只跟自己的現用片比，不做鍵與鍵之間的互相改指（異體鍵的現用片
打法是否相同沒有驗證過）。

用法：python3 quality_compare.py incoming_20260914c   （讀 groups.json、_extracted/）
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path.home() / "0813"
sys.path.insert(0, str(ROOT))
from clip_metrics import _arrays, metrics  # noqa: E402

# ── composer.js 移植（0821_bundle/composer.js，數值照抄） ──────────────
REST_ENTER, REST_SETTLE = -0.65, -0.35
HAND_MATCH_MAX, HAND_MATCH_MARGIN = 0.12, 0.02


def repair_handedness(frames):
    """composer.repairHandedness：雙手靠近時 MediaPipe 常把兩隻都標成同一側，
    composer 播之前會用姿態手腕重新指派。不先修，指標會把「兩隻都有抓到」
    算成「一隻整段不見」（看醫生 左手 0.20，實際兩手都在）。"""
    for fr in frames:
        hs = fr.get("hands") or []
        pose = fr.get("pose") or {}
        if not hs or not pose.get("landmarks"):
            continue
        vis = pose.get("visibility") or []
        pw = {"Left": pose["landmarks"][15], "Right": pose["landmarks"][16]}
        ok = {"Left": (vis[15] if len(vis) > 15 and vis[15] is not None else 1) > 0.5,
              "Right": (vis[16] if len(vis) > 16 and vis[16] is not None else 1) > 0.5}

        def dist(h, side):
            w, p = h["landmarks"][0], pw[side]
            return float(np.hypot(w[0] - p[0], w[1] - p[1]))
        if len(hs) == 1 and (ok["Left"] or ok["Right"]):
            h = hs[0]
            dl = dist(h, "Left") if ok["Left"] else float("inf")
            dr = dist(h, "Right") if ok["Right"] else float("inf")
            if min(dl, dr) < HAND_MATCH_MAX and abs(dl - dr) > HAND_MATCH_MARGIN:
                h["handedness"] = "Left" if dl < dr else "Right"
        elif len(hs) >= 2 and ok["Left"] and ok["Right"]:
            h0, h1 = hs[0], hs[1]
            straight = dist(h0, "Left") + dist(h1, "Right")
            crossed = dist(h0, "Right") + dist(h1, "Left")
            if min(straight, crossed) < 2 * HAND_MATCH_MAX:
                if straight <= crossed:
                    h0["handedness"], h1["handedness"] = "Left", "Right"
                else:
                    h0["handedness"], h1["handedness"] = "Right", "Left"
    return frames


def _wrist_height(f):
    p = (f.get("pose") or {}).get("landmarks")
    if not p or len(p) < 25:
        return None
    sho_y = (p[11][1] + p[12][1]) / 2
    torso = (p[23][1] + p[24][1]) / 2 - sho_y
    if not torso > 1e-4:
        return None
    return max((sho_y - p[15][1]) / torso, (sho_y - p[16][1]) / torso)


def _trim_rest_posture(seg):
    if len(seg) < 8:
        return seg
    h = [_wrist_height(f) for f in seg]
    if all(v is None for v in h):
        return seg
    s = 0
    while s < len(seg) and (h[s] is None or h[s] < REST_ENTER):
        s += 1
    e = len(seg) - 1
    while e > s and (h[e] is None or h[e] < REST_ENTER):
        e -= 1
    if s >= e:
        return seg
    room = int((e - s) * 0.35)
    s2, e2 = s, e
    while (s2 < s + room and h[s2] is not None and h[s2] < REST_SETTLE
           and h[s2 + 1] is not None and h[s2 + 1] > h[s2]):
        s2 += 1
    while (e2 > e - room and h[e2] is not None and h[e2] < REST_SETTLE
           and h[e2 - 1] is not None and h[e2 - 1] > h[e2]):
        e2 -= 1
    if e2 - s2 < 4:
        s2, e2 = s, e
    out = seg[s2:e2 + 1]
    return out if len(out) >= 5 else seg


def _trim_neutral(seg):
    if len(seg) < 8:
        return seg
    act = []
    for i in range(1, len(seg)):
        a = 0.0
        dt = max(1e-3, seg[i]["timestamp"] - seg[i - 1]["timestamp"])
        for side in ("Left", "Right"):
            h1 = next((h for h in seg[i - 1].get("hands") or [] if h.get("handedness") == side), None)
            h2 = next((h for h in seg[i].get("hands") or [] if h.get("handedness") == side), None)
            if h1 and h2:
                (x1, y1), (x2, y2) = h1["landmarks"][0][:2], h2["landmarks"][0][:2]
                a = max(a, float(np.hypot(x2 - x1, y2 - y1)) / dt)
            elif h1 or h2:
                a = max(a, 1.0)
        act.append(a)
    dur = seg[-1]["timestamp"] - seg[0]["timestamp"]
    max_trim = min(0.8, dur * 0.25)
    s = 0
    while s < len(act) and act[s] < 0.12 and seg[s + 1]["timestamp"] - seg[0]["timestamp"] <= max_trim:
        s += 1
    e = len(seg) - 1
    while e > s + 1 and act[e - 1] < 0.12 and seg[-1]["timestamp"] - seg[e - 1]["timestamp"] <= max_trim:
        e -= 1
    out = seg[s:e + 1]
    return out if len(out) >= 5 else seg


def played_span(d, start, end):
    """lexicon 區段 [start, end] 裡，composer 真正會播的那段（秒）。"""
    fr = [f for f in d["frames"] if start - 1e-6 <= f["timestamp"] <= end + 1e-6]
    if len(fr) < 2:
        return start, end
    seg = _trim_neutral(_trim_rest_posture(fr))
    # 不取整：取到小數 3 位再拿去切，端點那一幀會被捨掉
    return seg[0]["timestamp"], seg[-1]["timestamp"]


# ── 新片的 lexicon 區段：trim_span.py 的手腕高度切法（照抄） ───────────
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
            round(min(float(ts[-1]), float(ts[i1]) + 0.15), 3))


# ── 手部細節 ───────────────────────────────────────────────────────────
CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]


def _joint_angles(w):
    out = []
    for ch in CHAINS:
        for j in range(1, 4):
            u, v = w[ch[j]] - w[ch[j - 1]], w[ch[j + 1]] - w[ch[j]]
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            if nu < 1e-9 or nv < 1e-9:
                return None
            out.append(np.degrees(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1, 1))))
    return np.array(out)


def hand_detail(d, start, end, in_play):
    frames = d["frames"]
    fps = d.get("fps") or 30
    W, H = d.get("source_width") or 1, d.get("source_height") or 1
    step = max(1, int(round(fps / 30)))
    idx = [i for i, f in enumerate(frames) if start - 1e-6 <= f["timestamp"] <= end + 1e-6][::step]
    seq = {"L": [], "R": []}
    for i in idx:
        f = frames[i]
        got = {}
        for hd in f.get("hands") or []:   # 標籤已經 repair_handedness 修過
            side = "L" if hd.get("handedness") == "Left" else "R"
            if side not in got:
                got[side] = hd
        for s in ("L", "R"):
            seq[s].append(got.get(s))
    res, px = [], []
    for s in in_play:
        run = []
        for hd in seq[s] + [None]:
            if hd is not None:
                lm = np.array(hd["landmarks"])[:, :2] * [W, H]
                px.append(float(np.linalg.norm(lm[9] - lm[0])))
                ang = _joint_angles(np.array(hd["world_landmarks"])) if hd.get("world_landmarks") else None
                if ang is not None:
                    run.append(ang)
                    continue
            if len(run) >= 7:
                x = np.array(run)
                k = np.ones(5) / 5
                sm = np.vstack([np.convolve(x[:, c], k, mode="same") for c in range(x.shape[1])]).T
                res.append((x - sm)[2:-2])
            run = []
    out = {"hand_px": round(float(np.median(px)), 1) if px else 0.0}
    if res:
        r = np.vstack(res)
        out["ajit"], out["ajit_n"] = round(float(np.sqrt(np.mean(r ** 2))), 2), int(len(r))
    else:
        out["ajit"], out["ajit_n"] = None, 0
    return out


def eval_clip(path, start, end):
    """lexicon 區段 → 播放區段 → 指標。"""
    d = json.load(open(path, encoding="utf-8"))
    repair_handedness(d["frames"])
    p0, p1 = played_span(d, start, end)
    m = metrics(d, p0, p1)
    m.update(lex_start=start, lex_end=end)
    if "act_eff" in m:
        hands = [c for c in m.get("hands", "") if c in "LR"] or ["L", "R"]
        m.update(hand_detail(d, p0, p1, hands))
    return m


def beats(a, b):
    """a（挑戰者）是否勝過 b（現任）。回傳 (bool, 理由)。"""
    if b is None or "act_eff" not in b:
        return True, "現任無指標"
    da = a["act_eff"] - b["act_eff"]
    if da >= 0.05:
        return True, f"act +{da:.2f}"
    if da <= -0.05:
        return False, f"act {da:.2f}"
    if a["tier"] != "ok" and b["tier"] == "ok":
        return False, "平手但挑戰者非 ok"
    fa, fb = a.get("ajit"), b.get("ajit")
    if fa is not None and fb is not None and abs(fa - fb) > 0.10 * max(fa, fb):
        return (fa < fb), f"平手，指節抖動 {fb:.1f}°→{fa:.1f}°"
    if fa is not None and fb is None:
        return True, "平手，現任量不到指節抖動"
    pa, pb = a.get("hand_px", 0), b.get("hand_px", 0)
    return (pa > pb), f"平手，抖動相近，手掌 {pb:.0f}→{pa:.0f}px"


def main():
    batch = ROOT / sys.argv[1]
    groups = json.load(open(batch / "groups.json", encoding="utf-8"))
    lex = json.load(open(ROOT / "recordings" / "lexicon.json", encoding="utf-8"))
    cache, rows = {}, []

    def old_m(key):
        e = lex.get(key)
        if not e or not e.get("recording"):
            return None
        k = (e["recording"], e.get("start"), e.get("end"))
        if k not in cache:
            p = ROOT / "recordings" / e["recording"]
            cache[k] = dict(eval_clip(p, e.get("start"), e.get("end")), recording=e["recording"]) \
                if p.is_file() else None
        return cache[k]

    for g in groups:
        sc = batch / "_extracted" / f"{g['rec']}.json"
        if not sc.is_file():
            print(f"⚠ 沒抽到 {g['rec']}")
            continue
        tr = trimmed_span(sc)
        if tr is None:
            a = metrics(sc)
            tr = (a["start"], a["end"])
        new = dict(eval_clip(sc, *tr), recording=f"{g['rec']}.json")
        keys = []
        for k in g["keys"]:
            inc = old_m(k)
            win, why = beats(new, inc)
            keys.append({"key": k, "incumbent": inc, "take_new": win, "why": why})
        rows.append({"file": g["file"], "rec": g["rec"], "word": g["word"], "new": new, "keys": keys})
    json.dump(rows, open(batch / "quality_compare.json", "w"), ensure_ascii=False, indent=1)

    def fmt(m):
        if not m or "act_eff" not in m:
            return "(無)"
        aj = f"{m['ajit']:4.1f}" if m.get("ajit") is not None else "  - "
        return (f"{m['act_eff']:.2f} {m['tier']:6s} {aj}° {m['hand_px']:4.0f}px "
                f"{m['end'] - m['start']:.2f}/{m['lex_end'] - m['lex_start']:.2f}s")
    print(f"{'鍵':5s} | {'現任 act tier 抖動 手掌 播/存':34s} | {'新片':34s} | 結果")
    for r in rows:
        for k in r["keys"]:
            print(f"{k['key']:5s} | {fmt(k['incumbent']):34s} | {fmt(r['new']):34s} | "
                  f"{'換新片' if k['take_new'] else '維持'}（{k['why']}） "
                  f"[{(k['incumbent'] or {}).get('recording', '')}]")


if __name__ == "__main__":
    main()
