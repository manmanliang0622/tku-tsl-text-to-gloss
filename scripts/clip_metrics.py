#!/usr/bin/env python3
"""單支 sidecar（或既有詞條的某個區段）的品質指標與有效區段。

跟 scan_quality.py 的關係：那支掃全庫、按 lexicon 既有切點算，指標 act_rate＝
「手腕高於髖線期間手部被偵測到的比例」。那個口徑在**單詞citation片**上會失真，
踩過兩次：

  1. 整支片算：這批補片前後各有 1–3 秒手放下的靜止段（「難過」1.5 秒時手還垂著），
     算進去偵測率直接砍半。故改成只算**有效區段**——手被偵測到的最長連續段。
  2. 髖線判 raised：補片是上半身特寫，髖部在畫面外，MediaPipe 仍外插出一條線，
     結果垂在腰邊的閒置手也被判成「舉起來在打」。雙手取較差側之後，單手詞
     （台灣、紅、生病…）就被算成 act=0.00，29 支好片被誤判成 severe。
     改用**姿態判哪隻手在打**：手腕相對肩寬的最大位移 mov，只有 mov 夠大的那側
     才進較差側取值。實測 台灣 L=0.21／R=0.72、幫 L=0.74／R=0.73 分得很開。

所以 act_eff ＝「真正在打的那些手，在有效區段裡被偵測到的比例」。要跟
entries_final.csv 的 tier 比較時**兩邊都要用這支算**，口徑才對得起來
（ingest_new_videos.py 就是這樣比新舊片的）。
"""
import json

import numpy as np

MOV_ABS = 0.25   # 手腕位移下限（肩寬為單位）：低於此視為閒置手
MOV_REL = 0.40   # 且至少要有主動手的四成，否則同樣視為閒置


def _arrays(d):
    frames = d.get("frames", [])
    n = len(frames)
    fps = d.get("fps") or 0
    ts = np.array([f.get("timestamp", i / fps if fps else i) for i, f in enumerate(frames)])
    a = {}
    for s in ("L", "R"):
        a["pres_" + s] = np.zeros(n, bool)
        a["wrist_" + s] = np.full((n, 2), np.nan)
        a["vis_" + s] = np.zeros(n)
    for i, f in enumerate(frames):
        pose = f.get("pose") or {}
        lm, vis = pose.get("landmarks"), pose.get("visibility")
        if lm and len(lm) >= 33:
            l11, l12 = np.array(lm[11][:2]), np.array(lm[12][:2])
            mid = (l11 + l12) / 2
            sw = np.linalg.norm(l11 - l12) or 1.0
            for s, wi in (("L", 15), ("R", 16)):
                a["wrist_" + s][i] = (np.array(lm[wi][:2]) - mid) / sw
                if vis and len(vis) > wi:
                    a["vis_" + s][i] = vis[wi]
        for hd in (f.get("hands") or []):
            s = "L" if hd.get("handedness") == "Left" else "R"
            a["pres_" + s][i] = True
    return a, ts, fps, n


def _highpass_rms(x, win=5):
    n = len(x)
    if n < win + 2:
        return float("nan")
    k = np.ones(win) / win
    sm = np.vstack([np.convolve(x[:, i], k, mode="same") for i in range(x.shape[1])]).T
    t = win // 2
    hp = (x - sm)[t:n - t]
    return float(np.sqrt(np.nanmean(hp ** 2)))


def _active_span(pres, ts, fps):
    """手被偵測到的最長連續段（容忍 <=0.2 秒的斷點）的 index 範圍。"""
    idx = np.flatnonzero(pres)
    if not len(idx):
        return None
    gap = max(1, int(round(0.2 * (fps or 30))))
    runs, s0 = [], idx[0]
    for p, q in zip(idx, idx[1:]):
        if q - p > gap:
            runs.append((s0, p)); s0 = q
    runs.append((s0, idx[-1]))
    return max(runs, key=lambda r: r[1] - r[0])


def metrics(path, start=None, end=None):
    """start/end 給了就量那個區段（比既有詞條時用），否則自己找有效區段。"""
    d = json.loads(open(path, encoding="utf-8").read())
    a, ts, fps, n = _arrays(d)
    out = {"n_frames": n, "fps": round(fps, 2),
           "dur": round(float(ts[-1] - ts[0]), 3) if n > 1 else 0.0,
           "width": d.get("source_width"), "height": d.get("source_height")}
    if n == 0:
        return out

    if start is not None and end is not None:
        i0 = int(np.searchsorted(ts, start, "left"))
        i1 = min(n - 1, int(np.searchsorted(ts, end, "right")))
        out["start"], out["end"] = round(float(start), 3), round(float(end), 3)
        out["span_source"] = "given"
    else:
        span = _active_span(a["pres_L"] | a["pres_R"], ts, fps)
        if span is None:
            out.update(start=0.0, end=round(float(ts[-1]), 3), span_source="none",
                       act_eff=0.0, tier="severe", hands="none")
            return out
        i0, i1 = span
        pad = 0.12
        out["start"] = round(max(0.0, float(ts[i0]) - pad), 3)
        out["end"] = round(min(float(ts[-1]), float(ts[i1]) + pad), 3)
        out["span_source"] = "auto"
    if i1 <= i0:
        i1 = min(n - 1, i0 + 1)
    sl = slice(i0, i1 + 1)
    out["span_frames"] = int(i1 - i0 + 1)
    out["span_dur"] = round(float(ts[i1] - ts[i0]), 3)

    mov, hr = {}, {}
    for s in ("L", "R"):
        hr[s] = float(np.mean(a["pres_" + s][sl]))
        w, vis = a["wrist_" + s][sl], a["vis_" + s][sl] > 0.5
        ww = w[vis]
        mov[s] = (float(np.nanmax(np.linalg.norm(ww - np.nanmedian(ww, axis=0), axis=1)))
                  if len(ww) > 3 and not np.isnan(ww).all() else 0.0)
        if np.isnan(mov[s]):
            mov[s] = 0.0
        out[f"hand_rate_{s}"] = round(hr[s], 4)
        out[f"mov_{s}"] = round(mov[s], 3)
        wok = np.where(vis[:, None], w, np.nan)
        for c in range(wok.shape[1]):
            col = wok[:, c]
            miss = np.isnan(col)
            if miss.all():
                continue
            ix = np.arange(len(col))
            col[miss] = np.interp(ix[miss], ix[~miss], col[~miss])
            wok[:, c] = col
        out[f"jit_{s}"] = None if np.isnan(wok).any() else round(_highpass_rms(wok), 5)

    top = max(mov.values())
    in_play = [s for s in ("L", "R")
               if mov[s] >= max(MOV_ABS, MOV_REL * top) or mov[s] == top]
    out["hands"] = "".join(in_play) or "none"
    out["act_eff"] = round(min(hr[s] for s in in_play), 4) if in_play else 0.0
    jits = [out[f"jit_{s}"] for s in in_play if out.get(f"jit_{s}") is not None]
    out["jit"] = round(max(jits), 5) if jits else None

    act, jit = out["act_eff"], (out["jit"] or 0.0)
    out["tier"] = "severe" if act < 0.30 else ("poor" if act < 0.60 or jit > 0.05 else "ok")
    return out


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        print(p.split("/")[-1], json.dumps(metrics(p), ensure_ascii=False))
