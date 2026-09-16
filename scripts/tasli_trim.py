#!/usr/bin/env python3
"""tasli（新詞學習網）詞條的切段修短：只留第一個打法的第一次示範。

影片結構（抽 20 支＋逐格看過）：同一個詞的 A/B/C… 打法接在一起，有的每個打法示範兩次
（新住民＝A 外國/嫁來 ×2、B ×2、C 新/住/民 ×2），有的各一次（帳戶凍結＝A 銀行+存摺+封鎖、
B 銀行+存摺+套住、C 銀行+號碼+封鎖），有的換打法時連打者和背景一起換（蜂炮）。
詞條原本 start=0、end=片長（中位 9.5 秒、最長 34 秒），等速播放後會播 15–30 秒。

切法三步：
1. 休息／活動：每幀 h＝(肩線 y − 較高那隻手腕 y)／軀幹長（同 composer.wristHeight）。
   手放身側（h < REST_ENTER）是休息；雙手交握在腹前的打者（帳戶凍結、居服員）h 只有
   -0.55，所以「低於 LOW 且手腕幾乎不動」也算休息。
2. 換打法的時間點：只靠休息長短分不出來——帳戶凍結 詞內停頓 0.3–0.5 秒、換打法 0.5–0.8 秒。
   用字幕：背景是靜止純色，把打者（landmark 外框＋PAD）遮掉後，畫面其餘部分只有字幕換字
   或換場景會突變。字幕樣式位置每支都不同，所以不讀字，只看「區塊差異的單點尖峰」：
   > CAP_MIN、是前後鄰點中位數的 CAP_RATIO 倍，且發生在休息時（換打法時打者一定停著；
   手部動作漏出來的變化是連續鼓包、而且在活動中）。換場景（差異 > CUT_MIN）一律算。
   換字前後那塊畫面要「持續」不同（尖峰前 0.4–1.0 秒 vs 後 0.4–1.0 秒）：手放下時手臂
   漏出遮罩的變化只有一瞬間（新住民 4.4／9.8 秒就是這種假尖峰）。
   字幕壓在軀幹上、只換一個字母的（薪資凍漲 A→B→C→D、器官移植、轉乘優惠）被整個人的
   遮罩蓋掉了，所以再跑一次只遮「會動的部位」（頭、手臂、手）的版本；軀幹與腿不太動，
   字幕照樣看得到。這版假尖峰多（手在字幕前面打），只收落在兩段示範之間空檔的。
3. 第一個打法的時間窗裡，活動段一段段往後併，直到遇到下一次示範為止。跟這次示範的
   第一段比軌跡（雙手腕相對肩膀、肩寬為單位的 DTW）並看停頓：
   - DTW < 0.25：重複示範（實測 0.06–0.11：讀卡機、Twitter）
   - 停頓 ≥ 1.2 秒且 DTW < 0.45、或停頓 ≥ 1.5 秒且 DTW < 0.65：沒抓到字幕的下一個打法
     （薪資凍漲 0.33／1.8 秒、器官移植 0.33／1.3 秒、轉乘優惠 0.35／1.3 秒、日環食 0.59／1.5 秒）
   - 其他：同一個詞的下一部分（家樂福超市 0.39／0.9 秒、帳戶凍結 0.50／0.4 秒、
     行人庇護島 0.72／1.3 秒、監護宣告 1.0／1.1 秒）
   單看停頓或單看 DTW 都分不開這幾組。短於 0.8 秒的碎段、或停頓不到 0.5 秒的，不做重複判定
   （招潮蟹 開頭被切成 0.3 秒一段，DTW 0.21 會被誤判成重複）。取併好的第一次示範，前後各留 PAD_SEC。
   composer 播放時會再切掉頭尾的休息。

    python3 tasli_trim.py            # 試算，輸出 ~/0813/tasli_trim_plan.json
    python3 tasli_trim.py --apply    # 寫入 ~/0813/recordings/lexicon.json（有備份）
"""
import json, math, statistics as st, subprocess, sys, time, shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REST_ENTER = -0.65
LOW, STILL = -0.40, 0.8          # 腹前交握：低於 LOW 且手腕速度 < STILL 肩寬/秒
MIN_RUN, REP_DTW, PAD_SEC = 0.3, 0.25, 0.25
W, H, CFPS, BLOCK, MASK_PAD = 160, 90, 5, 10, 0.12
CAP_MIN, CAP_RATIO, CUT_MIN = 12.0, 3.5, 100.0
ROOT = Path.home() / "0813"
REC = ROOT / "recordings"

# 自動切完仍 > 8 秒的 12 條逐條讀字幕核對過，其中 5 條（4 支影片）是「打法之間只停 0.6–0.8 秒、
# 軌跡又相近」，自動規則分不出來，這裡直接指定第一個打法的區段（值取自 runs，前後各留 PAD_SEC）。
# 其餘 7 條確認是單一打法的長複合詞，維持自動結果：
# 大全聯 A=大/全(北部,南部)/聯、Instagram A=I/G、AI A=AI/人/工/智慧、iOS、蘋果作業系統、
# 日環食 A=太陽與月球重疊/黑影遮蓋+黃色環形、心律不整=心跳/線條上上下下。
MANUAL = {
    "tasli_00169_已讀不回.json": [1.43, 5.64],          # A 見(手機)/顯示/回/無（B/C/D 在後面）
    "tasli_00039_飛蚊症.json": [1.47, 6.47],            # A（B 從 7.7 秒開始）
    "tasli_00185_維基百科.json": [1.0, 3.87],           # A (右)W放在(左)1/解釋（B 從 5.5 秒開始）
    "tasli_00170_不讀不回（未讀）.json": [1.07, 5.64],   # A 顯示畫面/點進去/不（B 從 6.4 秒開始）
}


def rest_mask(d):
    fr = d["frames"]
    n = len(fr)
    ts = [f["timestamp"] for f in fr]
    fps = d.get("fps") or 30
    h, wr, sw = [None] * n, [None] * n, []
    for i, f in enumerate(fr):
        p = (f.get("pose") or {}).get("landmarks")
        if not p or len(p) < 25:
            continue
        sho = (p[11][1] + p[12][1]) / 2
        torso = (p[23][1] + p[24][1]) / 2 - sho
        if torso > 1e-4:
            h[i] = max((sho - p[15][1]) / torso, (sho - p[16][1]) / torso)
        wr[i] = (p[15][:2], p[16][:2])
        sw.append(math.hypot(p[11][0] - p[12][0], p[11][1] - p[12][1]))
    swm = st.median(sw) if sw else 0.2
    k = max(1, int(round(fps / 15)))   # 約 ±67ms 算速度
    rest = []
    for i in range(n):
        a, b = wr[max(0, i - k)], wr[min(n - 1, i + k)]
        dt = ts[min(n - 1, i + k)] - ts[max(0, i - k)]
        sp = (max(math.hypot(b[s][0] - a[s][0], b[s][1] - a[s][1]) for s in (0, 1)) / swm / dt) if a and b and dt > 0 else 0.0
        rest.append(h[i] is None or h[i] < REST_ENTER or (h[i] < LOW and sp < STILL))
    # 5 幀多數決去雜訊
    sm = [sum(rest[max(0, i - 2):i + 3]) * 2 > len(rest[max(0, i - 2):i + 3]) for i in range(n)]
    return ts, sm


def active_runs(ts, rest):
    runs, i, n = [], 0, len(rest)
    while i < n:
        if rest[i]:
            i += 1; continue
        j = i
        while j + 1 < n and not rest[j + 1]:
            j += 1
        if ts[j] - ts[i] >= MIN_RUN:
            runs.append([ts[i], ts[j]])
        i = j + 1
    return runs


MOVING = list(range(0, 11)) + list(range(13, 23))   # 頭、手肘、手腕、手部的姿態點
ARM_SEGS = [(11, 13), (13, 15), (12, 14), (14, 16)]
TIGHT_PAD = 0.07


def caption_changes(mp4, d):
    raw = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(mp4), "-vf",
                          f"fps={CFPS},scale={W}:{H},format=gray", "-f", "rawvideo", "-"],
                         capture_output=True, check=True).stdout
    g = np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.float32)
    fr = d["frames"]
    fts = np.array([f["timestamp"] for f in fr])
    full, tight = [], []
    for s in range(len(g)):
        i = int(np.clip(np.searchsorted(fts, s / CFPS), 0, len(fr) - 1))
        xs, ys = [], []
        mt = np.zeros((H, W), bool)

        def box(x, y):
            x = 1 - x   # extractor 做了 selfie 翻轉
            mt[int(max(0, (y - TIGHT_PAD) * H)):int(min(H, (y + TIGHT_PAD) * H)) + 1,
               int(max(0, (x - TIGHT_PAD) * W)):int(min(W, (x + TIGHT_PAD) * W)) + 1] = True
        for j in range(max(0, i - 3), min(len(fr), i + 4)):
            p = (fr[j].get("pose") or {}).get("landmarks")
            if p:
                for q in p[:25]:
                    xs.append(q[0]); ys.append(q[1])
                for q in MOVING:
                    box(p[q][0], p[q][1])
                for a, b in ARM_SEGS:
                    for u in (0.25, 0.5, 0.75):
                        box(p[a][0] + (p[b][0] - p[a][0]) * u, p[a][1] + (p[b][1] - p[a][1]) * u)
            for hd in fr[j].get("hands") or []:
                for q in hd["landmarks"]:
                    xs.append(q[0]); ys.append(q[1])
                    box(q[0], q[1])
        m = np.zeros((H, W), bool)
        if xs:
            x0, x1 = 1 - max(xs), 1 - min(xs)
            m[int(max(0, (min(ys) - MASK_PAD) * H)):, int(max(0, (x0 - MASK_PAD) * W)):int(min(W, (x1 + MASK_PAD) * W))] = True
        full.append(m)
        tight.append(mt)
    return spikes(g, full), spikes(g, tight)


def spikes(g, masks):
    curve, where = [0.0], [None]
    for s in range(1, len(g)):
        keep = ~(masks[s] | masks[s - 1])
        ad = np.abs(g[s] - g[s - 1])
        best, at = 0.0, (0, 0)
        for r in range(0, H - BLOCK + 1, BLOCK // 2):
            for c in range(0, W - BLOCK + 1, BLOCK // 2):
                if keep[r:r + BLOCK, c:c + BLOCK].mean() >= 0.9:
                    v = float(ad[r:r + BLOCK, c:c + BLOCK].mean())
                    if v > best:
                        best, at = v, (r, c)
        curve.append(best)
        where.append(at)
    out = []
    for s in range(1, len(curve)):
        nb = [curve[j] for j in range(max(1, s - 3), min(len(curve), s + 4)) if j != s]
        base = st.median(nb) if nb else 0.0
        if curve[s] > CUT_MIN:
            out.append((s / CFPS, curve[s])); continue
        if not (curve[s] > CAP_MIN and curve[s] > CAP_RATIO * max(base, 1.0)):
            continue
        r, c = where[s]
        before = [g[j][r:r + BLOCK, c:c + BLOCK] for j in range(max(0, s - 5), max(0, s - 1)) if not masks[j][r:r + BLOCK, c:c + BLOCK].any()]
        after = [g[j][r:r + BLOCK, c:c + BLOCK] for j in range(min(len(g), s + 2), min(len(g), s + 6)) if not masks[j][r:r + BLOCK, c:c + BLOCK].any()]
        if before and after and float(np.abs(np.mean(before, 0) - np.mean(after, 0)).mean()) > 0.5 * curve[s]:
            out.append((s / CFPS, curve[s]))
    return out


def wrist_traj(d, a, b, n=40):
    pts = []
    for f in d["frames"]:
        if not (a <= f["timestamp"] <= b):
            continue
        p = (f.get("pose") or {}).get("landmarks")
        if not p:
            continue
        mx, my = (p[11][0] + p[12][0]) / 2, (p[11][1] + p[12][1]) / 2
        sw = math.hypot(p[11][0] - p[12][0], p[11][1] - p[12][1]) or 0.2
        pts.append([(p[15][0] - mx) / sw, (p[15][1] - my) / sw, (p[16][0] - mx) / sw, (p[16][1] - my) / sw])
    if len(pts) < 2:
        return None
    pts = np.array(pts)
    return pts[np.round(np.linspace(0, len(pts) - 1, n)).astype(int)]


def dtw(A, B, band=10):
    n, m = len(A), len(B)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            D[i, j] = np.linalg.norm(A[i - 1] - B[j - 1]) + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


def plan_one(rec):
    d = json.load(open(REC / rec, encoding="utf-8"))
    ts, rest = rest_mask(d)
    dur = ts[-1] if ts else 0.0
    runs = active_runs(ts, rest)
    if not runs:
        return rec, {"dur": dur, "runs": [], "changes": [], "span": None}
    changes, tight_changes = caption_changes(REC / (rec[:-5] + ".mp4"), d)

    def resting_near(t):
        i0 = int(np.searchsorted(ts, t - 0.3)); i1 = int(np.searchsorted(ts, t + 0.3))
        return any(rest[max(0, i0):min(len(rest), i1 + 1)])

    def between_runs(t):
        return not any(a + 0.3 < t < b - 0.3 for a, b in runs)
    bounds = sorted({t for t, v in changes if (v > CUT_MIN or resting_near(t)) and t > runs[0][0] + MIN_RUN}
                    | {t for t, v in tight_changes if between_runs(t) and t > runs[0][0] + MIN_RUN})
    end_win = bounds[0] if bounds else dur + 1
    win = [[r[0], min(r[1], end_win)] for r in runs if r[0] < end_win]
    first = wrist_traj(d, *win[0])
    g0, dtws = list(win[0]), []
    for r in win[1:]:
        t = wrist_traj(d, *r)
        dist = dtw(first, t) if first is not None and t is not None else 9.9
        gap = r[0] - g0[1]
        dtws.append([round(dist, 2), round(gap, 2)])
        # 很短的碎段（招潮蟹 0.3 秒、停頓 0.2 秒）DTW 不可靠：兩段都夠長、中間真的停過才可能是重複
        solid = min(r[1] - r[0], win[0][1] - win[0][0]) >= 0.8 and gap >= 0.5
        if solid and (dist < REP_DTW or (gap >= 1.2 and dist < 0.45) or (gap >= 1.5 and dist < 0.65)):
            break          # 下一次示範（重複或另一個打法）開始了
        g0[1] = r[1]
    groups = [g0]
    man = MANUAL.get(rec)
    if man:
        g0 = list(man)
    span = [round(max(0.0, g0[0] - PAD_SEC), 3), round(min(dur, g0[1] + PAD_SEC), 3)]
    return rec, {"dur": round(dur, 2), "runs": [[round(a, 2), round(b, 2)] for a, b in runs],
                 "changes": [[round(t, 1), round(v, 1)] for t, v in changes],
                 "tight": [[round(t, 1), round(v, 1)] for t, v in tight_changes], "bounds": [round(b, 1) for b in bounds],
                 "groups": [[round(a, 2), round(b, 2)] for a, b in groups], "dtw": dtws, "span": span}


def main():
    apply = "--apply" in sys.argv
    lex_path = REC / "lexicon.json"
    lex = json.load(open(lex_path, encoding="utf-8"))
    keys = sorted(k for k, e in lex.items() if e.get("recording", "").startswith("tasli_"))
    recs = sorted({lex[k]["recording"] for k in keys})
    with ProcessPoolExecutor(8) as ex:
        info = dict(ex.map(plan_one, recs, chunksize=4))
    plan = []
    for k in keys:
        e = lex[k]; inf = info[e["recording"]]
        plan.append({"key": k, "rec": e["recording"], "old": [e["start"], e["end"]], "new": inf["span"], **{x: inf.get(x) for x in ("dur", "runs", "bounds", "groups", "changes", "dtw")}})
    json.dump(plan, open(ROOT / "tasli_trim_plan.json", "w"), ensure_ascii=False, indent=1)
    ok = [p for p in plan if p["new"]]
    old = [p["old"][1] - p["old"][0] for p in plan]
    new = [p["new"][1] - p["new"][0] for p in ok]
    q = lambda v, x: sorted(v)[min(len(v) - 1, int(x * len(v)))]
    print(f"詞條 {len(plan)}（影片 {len(recs)}）  切得出來 {len(ok)}  找不到活動段 {len(plan) - len(ok)}")
    print(f"舊切段 中位 {st.median(old):.1f}s  p90 {q(old, .9):.1f}s  最長 {max(old):.1f}s")
    print(f"新切段 中位 {st.median(new):.1f}s  p10 {q(new, .1):.1f}s  p90 {q(new, .9):.1f}s  最長 {max(new):.1f}s")
    print("字幕/場景切成幾個打法:", sorted(__import__("collections").Counter(len(info[r]["bounds"] or []) + 1 for r in recs if info[r]["span"]).items()))
    print("新切段 > 6s:", [(p["key"], round(p["new"][1] - p["new"][0], 1)) for p in ok if p["new"][1] - p["new"][0] > 6])
    print("新切段 < 1.2s:", [(p["key"], round(p["new"][1] - p["new"][0], 1)) for p in ok if p["new"][1] - p["new"][0] < 1.2])
    if not apply:
        print("dry-run：加 --apply 才寫入")
        return
    ts = time.strftime("%Y%m%d-%H%M%S")
    bak = f"{lex_path}.bak-tasli-trim-{ts}"
    shutil.copy2(lex_path, bak)
    n = 0
    for p in ok:
        e = lex[p["key"]]
        if [e["start"], e["end"]] != p["old"]:
            continue   # 試算後被別人改過，不動
        e["start"], e["end"] = p["new"]
        e.setdefault("trimmed_from", p["old"])
        n += 1
    lex_path.write_text(json.dumps(lex, ensure_ascii=False), encoding="utf-8")
    print(f"寫入 {n} 條，備份 {bak}")


if __name__ == "__main__":
    main()
