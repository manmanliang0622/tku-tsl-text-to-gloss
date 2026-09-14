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
  3. 光看 mov 還是會漏：單手詞的另一隻手臂抬起又放下，位移跟主手一樣大。
     2026-09-02 那批辭典片有 3 支（叔叔／姑姑／幾歲）主手偵測 0.89–0.94，
     卻因為閒置手被算進來而 act 掉到 0.01–0.12，又是一輪假 severe。再加一道
     門檻：整段幾乎抓不到手（hr < HR_FLOOR）**且**手腕從沒抬過腰
     （high > WAIST）的那側直接剔除，兩件事要同時成立才算閒置。
  4. mov／high 取的是極值，一個瞬間跳點就能讓兩道門檻同時失效。2026-09-15 站姿
     綠幕片：閒置手垂在畫面底，另一手從旁經過時姿態模型把左腕拉到胸口 4 幀
     （67ms），high 從 2.2 變 0.75，喝（右手偵測 0.96）被算成 act=0.006。
     故 mov／high 改用 SPIKE_WIN 秒的滑動中位數去跳點；jit 仍用原始訊號。
  5. 同一批還有反過來的情況：動得少的詞（一點／什麼／多少錢），畫面外那隻閒置手
     外插出來的手腕位移反而最大，mov 只挑中它，第 3 條又規定不能把在打的手剔光，
     act 就拿閒置手的 0 來算。mov 挑中的全是閒置手時，改用偵測率 >= 0.60 的那側。

所以 act_eff ＝「真正在打的那些手，在有效區段裡被偵測到的比例」。要跟
entries_final.csv 的 tier 比較時**兩邊都要用這支算**，口徑才對得起來
（ingest_new_videos.py 就是這樣比新舊片的）。
"""
import json

import numpy as np

MOV_ABS = 0.25   # 手腕位移下限（肩寬為單位）：低於此視為閒置手
MOV_REL = 0.40   # 且至少要有主動手的四成，否則同樣視為閒置
HR_FLOOR = 0.15  # 閒置手第二道門檻：整段幾乎抓不到手
WAIST = 0.90     # 且手腕（相對肩中點、肩寬為單位）從沒抬過腰
SPIKE_WIN = 0.15  # 秒：mov／high 去跳點的中位數視窗（壓得掉 <0.075 秒的跳點）


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


def _despike(ww, fps):
    """姿態手腕序列的滑動中位數（視窗 SPIKE_WIN 秒、奇數幀），只給 mov／high 用。"""
    k = max(3, int(round(SPIKE_WIN * (fps or 30))) | 1)
    if len(ww) < k:
        return ww
    h = k // 2
    pad = np.pad(ww, ((h, h), (0, 0)), mode="edge")
    return np.nanmedian(np.lib.stride_tricks.sliding_window_view(pad, k, axis=0), axis=-1)


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
    """start/end 給了就量那個區段（比既有詞條時用），否則自己找有效區段。
    path 也可以直接給已載入的 sidecar dict（例如先修過 handedness 的）。"""
    d = path if isinstance(path, dict) else json.loads(open(path, encoding="utf-8").read())
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

    mov, hr, high = {}, {}, {}
    for s in ("L", "R"):
        hr[s] = float(np.mean(a["pres_" + s][sl]))
        w, vis = a["wrist_" + s][sl], a["vis_" + s][sl] > 0.5
        ww = _despike(w[vis], fps)
        mov[s] = (float(np.nanmax(np.linalg.norm(ww - np.nanmedian(ww, axis=0), axis=1)))
                  if len(ww) > 3 and not np.isnan(ww).all() else 0.0)
        if np.isnan(mov[s]):
            mov[s] = 0.0
        high[s] = float(np.nanmin(ww[:, 1])) if len(ww) else 9.9
        out[f"hand_rate_{s}"] = round(hr[s], 4)
        out[f"mov_{s}"] = round(mov[s], 3)
        out[f"high_{s}"] = round(high[s], 3)
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
    # mov 一招擋不住「單手詞的另一隻手臂抬起又放下」：位移一樣大，卻沒在打。
    # 2026-09-02 這批辭典片踩到 3 支（叔叔／姑姑／幾歲，主手 0.89–0.94，
    # 卻因閒置手被算進來而 act=0.01–0.12，一律誤判 severe）。閒置手的特徵是
    # 「整段幾乎抓不到 **且** 手腕從沒抬過腰」——真的在打的手不會兩件事同時
    # 成立（最低的 邀請你 底手也有 hr 0.26、抬到 0.67）。代價是：真的在腰下
    # 打、又整段抓不到手的雙手詞會被當成單手詞放行，這種片本來就該人工過目。
    lead = max(hr.values())
    if lead >= 0.60:
        idle = [s for s in in_play if hr[s] < HR_FLOOR and high[s] > WAIST]
        if idle and len(idle) < len(in_play):
            in_play = [s for s in in_play if s not in idle]
        elif idle:
            # mov 挑中的全是閒置手：垂在畫面外的那隻，姿態模型外插的手腕會飄，
            # 位移比「真正在打、但動得少」的手還大（一點／什麼／多少錢，偵測 0.99
            # 卻 act=0.00）。改由偵測率過 0.60 的那側當在打的手。
            in_play = [s for s in ("L", "R") if hr[s] >= 0.60]
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
