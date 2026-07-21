#!/usr/bin/env python3
"""全爬文化部《臺灣手語語料庫》（https://tslcorpus.moc.gov.tw/，測試版）。

不同於 query_tslcorpus.py（只按主題關鍵詞抓命中句，供查證），本腳本系統性
列舉 18 個主題的全部語料段落，逐段抓真實聾人演繹的句子，取得**完整的
中文↔Gloss 平行語料**——這是語言模型微調價值最高的真實資料。

公開 API（網站前端同款）：
  1. corpusTheme/getCorpusThemeList          → 18 主題（code＋category）
  2. corpus/getCorpusList {category, theme}  → 該主題全部段落 uuid（G1D1P1…）
  3. corpus/findCorpusDetailByUuid {uuid}    → 該段全部句子：
       Text=中文、Hand=Gloss token 陣列、wordList=逐詞時間軸、film_url=影片

著作權：語料庫 © 文化部。本腳本僅抓取文字標記（Gloss／中文）供學術研究，
不下載影片；輸出附 source 與段落 uuid，引用／散布須依網站著作權聲明辦理。

輸出：
  data/refs/tslcorpus_raw.jsonl   一段一行原始回應（斷點續跑用）
  data/tslcorpus/parallel.jsonl   一句一行平行語料（中文↔Gloss）
限速：RATE_DELAY 秒／請求，單執行緒。
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

# 中文標點（含全形空白）——來源 Gloss token 偶帶句讀，需自 Gloss 端去除
PUNCT_RE = re.compile(r"[。，、？！；：「」『』（）　\s]+")


def clean_token(tok):
    """去除 Gloss token 首尾標點；保留 ++（重複貌）等語言學標記。"""
    return PUNCT_RE.sub("", tok).strip()

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "refs" / "tslcorpus_raw.jsonl"
OUT = BASE / "data" / "tslcorpus"
API = "https://tslcorpus.moc.gov.tw/api"
SITE = "https://tslcorpus.moc.gov.tw"
RATE_DELAY = 0.25
SOURCE = "臺灣手語語料庫（測試版），文化部，https://tslcorpus.moc.gov.tw/"


def post(ep, body):
    url = f"{API}/{ep}"
    last = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json",
                         "User-Agent": "TSL-research (academic; chi931209@gmail.com)"})
            with urllib.request.urlopen(req, timeout=40) as r:
                data = json.loads(r.read().decode("utf-8"))
            time.sleep(RATE_DELAY)
            return data
        except Exception as e:      # noqa: BLE001
            last = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"API 連續失敗 {ep} {body}: {last}")


def enumerate_segments():
    """回傳全部段落 uuid（去重）。

    注意：corpus/getCorpusList 的 theme 過濾參數實測無效——不論帶哪個主題，
    都回傳同一批全部段落。故只需呼叫一次即可取得完整清單；每段真正的主題
    改由 findCorpusDetailByUuid 回應中的 theme_data 決定（見 consolidate）。
    """
    data = post("corpus/getCorpusList", {"category": "1", "theme": "LE"})
    uuids = []
    seen = set()
    for seg in (data.get("data") or []):
        if seg["uuid"] not in seen:
            seen.add(seg["uuid"])
            uuids.append((seg["uuid"], seg.get("name")))
    print(f"全語料庫段落數（去重）: {len(uuids)}", flush=True)
    return uuids


def fetch_details(uuids):
    done = set()
    if RAW.exists():
        for line in RAW.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["uuid"])
            except (json.JSONDecodeError, KeyError):
                pass
    todo = [u for u in uuids if u[0] not in done]
    print(f"共 {len(uuids)} 段，已抓 {len(done)}，待抓 {len(todo)}", flush=True)
    RAW.parent.mkdir(parents=True, exist_ok=True)
    with RAW.open("a", encoding="utf-8") as f:
        for n, (uuid, seg_name) in enumerate(todo, 1):
            detail = post("corpus/findCorpusDetailByUuid", {"uuid": uuid})
            f.write(json.dumps({"uuid": uuid, "seg_name": seg_name,
                                "detail": detail.get("data", {})},
                               ensure_ascii=False) + "\n")
            if n % 25 == 0:
                f.flush()
                print(f"進度 {n}/{len(todo)}（{uuid}）", flush=True)


def consolidate():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    for line in RAW.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        d = r.get("detail") or {}
        film = d.get("film_url")
        theme = (d.get("theme_data") or {})
        theme_name = theme.get("name") or r.get("theme_name")
        theme_code = theme.get("code") or r.get("theme_code")
        for s in (d.get("apiData") or []):
            gloss = s.get("Hand") or []
            if isinstance(gloss, str):
                gloss = gloss.replace("／", "/").split()
            gloss = [t for t in (clean_token(g) for g in gloss) if t]
            chinese = (s.get("Text") or "").strip()
            if not gloss or not chinese:
                continue
            key = (s.get("ID"), s.get("TID"))
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "id": f"TC{len(rows)+1:05d}", "type": "sentence",
                "theme": theme_name, "theme_code": theme_code,
                "corpus_id": f'{s.get("ID")}/{s.get("TID")}',
                "seg_uuid": r["uuid"], "seg_name": r["seg_name"],
                "chinese": chinese,
                "gloss": gloss,
                "gloss_text": "/".join(gloss),
                "speaker": s.get("Speaker") or None,
                "film_url": (SITE + film) if film and film.startswith("/") else film,
                "review_status": "pending",
                "batch": "tslcorpus-full", "source": SOURCE,
            })
    with (OUT / "parallel.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    by_theme = Counter(r["theme"] for r in rows)
    print(f"OK: {len(rows)} 句平行語料 → data/tslcorpus/parallel.jsonl", flush=True)
    print("各主題:", dict(by_theme), flush=True)


def main():
    if "--consolidate-only" not in sys.argv:
        segs = enumerate_segments()
        fetch_details(segs)
    consolidate()


if __name__ == "__main__":
    main()
