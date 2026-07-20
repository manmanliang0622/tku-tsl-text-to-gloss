#!/usr/bin/env python3
"""爬取中正大學《台灣手語線上辭典》第五版（https://twtsl.ccu.edu.tw/）。

用途：擴充詞彙（計畫 3.3 節之外部詞彙來源）＋取得帶 Gloss 標記的真實例句。
著作權：辭典內容 © 國立中正大學手語語言學台灣研究中心（蔡素娟、戴浩一、劉世凱、
陳怡君 2026，中文版第五版）。本腳本僅抓取詞條後設資料與例句文字供研究使用，
不下載影片檔；所有輸出均附 source 欄位標明出處，引用時請依網站規定註明。

作法（皆為網站前端實際使用的公開 API）：
  1. /api/pinSearch?field=stroke&value=N  按筆畫列舉全部詞條 id（同前端「筆畫查詢」）
  2. /api/querySearch?id=I               詞條詳情：手形、位置、筆畫、動作描述、影片路徑
  3. /api/sentence?id=I                  常用詞例句：Gloss 標記＋中文翻譯（約 560 詞有）

輸出：
  data/twtsl/twtsl_words.jsonl      詞彙（一詞一行）
  data/twtsl/twtsl_sentences.jsonl  例句（中文 ↔ Gloss 平行資料）
  data/twtsl/raw_details.jsonl      原始 API 回應（斷點續跑用，已抓過的 id 不重抓）

限速：每次請求間隔 RATE_DELAY 秒，單執行緒；全程約 9,300 次請求。
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "data" / "twtsl"
RAW = OUT / "raw_details.jsonl"

API = "https://twtsl.ccu.edu.tw/api"
SITE = "https://twtsl.ccu.edu.tw"
RATE_DELAY = 0.12          # 秒／請求
MAX_STROKE = 35            # 「二十劃以上」逐劃查到此為止（30 劃已為 0）
SOURCE = ("台灣手語線上辭典（中文版第五版），蔡素娟、戴浩一、劉世凱、陳怡君 2026，"
          "國立中正大學手語語言學台灣研究中心，https://twtsl.ccu.edu.tw/")

VARIANT_RE = re.compile(r"^(.+?)_([A-Za-z]{1,2}\d?)$")   # 下午_A → (下午, A)


def get(path, **params):
    url = f"{API}/{path}?" + urllib.parse.urlencode({**params, "lang": "zh"})
    last_err = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "TSL-research-scraper (academic use; contact: chi931209@gmail.com)"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read().decode("utf-8"))
            time.sleep(RATE_DELAY)
            return data
        except Exception as e:          # noqa: BLE001 — 網路暫時性錯誤一律重試
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"API 連續失敗 {url}: {last_err}")


def enumerate_words():
    """按筆畫列舉全部詞條。

    同一手語（同 id）可有多個中文索引名（如 id 2642 同時列於「牛乳」「牛奶」），
    全部保留，寫入 enum_names.json 供 consolidate 當 aliases 用。
    回傳 {id: [名1, 名2, ...]}。
    """
    words = {}
    for stroke in range(1, MAX_STROKE + 1):
        page, page_size = 1, 200
        while True:
            data = get("pinSearch", field="stroke", value=stroke,
                       page=page, pageSize=page_size)
            recs = data.get("Record") or []
            for r in recs:
                words.setdefault(r["id"], [])
                if r["name"] not in words[r["id"]]:
                    words[r["id"]].append(r["name"])
            total = data.get("Total") or 0
            if page * page_size >= total or not recs:
                break
            page += 1
        print(f"筆畫 {stroke}: 累計 {len(words)} 詞", flush=True)
    (OUT / "enum_names.json").write_text(
        json.dumps({str(k): v for k, v in words.items()}, ensure_ascii=False),
        encoding="utf-8")
    return words


def fetch_details(words):
    """逐詞抓 querySearch＋sentence，原始回應寫入 RAW（續跑時跳過已抓 id）。"""
    done = set()
    if RAW.exists():
        for line in RAW.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    todo = [i for i in sorted(words) if i not in done]
    print(f"共 {len(words)} 詞，已抓 {len(done)}，待抓 {len(todo)}", flush=True)
    with RAW.open("a", encoding="utf-8") as f:
        for n, wid in enumerate(todo, 1):
            detail = get("querySearch", id=wid)
            sent = get("sentence", id=wid)
            f.write(json.dumps({"id": wid, "name": words[wid][0],
                                "detail": detail, "sentence": sent},
                               ensure_ascii=False) + "\n")
            if n % 100 == 0:
                f.flush()
                print(f"進度 {n}/{len(todo)}（id={wid} {words[wid]}）", flush=True)


def split_variant(name):
    m = VARIANT_RE.match(name)
    return (m.group(1), m.group(2)) if m else (name, None)


def consolidate():
    """把 RAW 整理成 twtsl_words.jsonl / twtsl_sentences.jsonl。

    詞名以 querySearch 回傳的正式名為準；筆畫索引中的其他中文同義名
    （如 牛乳→牛奶、珍奶→珍珠奶茶）存入 aliases 欄。
    """
    aliases_map = {}
    enum_file = OUT / "enum_names.json"
    if enum_file.exists():
        aliases_map = {int(k): v for k, v in
                       json.loads(enum_file.read_text(encoding="utf-8")).items()}
    words_out, sents_out, seen_sent = [], [], set()
    for line in RAW.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        recs = (row["detail"].get("Record") or [])
        if not recs:
            continue
        d = recs[0]
        canonical = d.get("name") or row["name"]
        gloss, variant = split_variant(canonical)
        aliases = sorted({split_variant(a)[0]
                          for a in aliases_map.get(row["id"], []) + [row["name"]]
                          if split_variant(a)[0] != gloss})
        locations = [d[k] for k in ("location1", "location2", "location3",
                                    "location4", "location5") if d.get(k)]
        handshapes = [d[k] for k in d if k.startswith("lo") and "_hs" in k and d[k]]
        words_out.append({
            "id": f"TW{row['id']:04d}", "twtsl_id": row["id"], "type": "word",
            "chinese": gloss, "name": canonical, "aliases": aliases,
            "gloss": [gloss], "gloss_text": gloss, "variant": variant,
            "stroke": d.get("stroke"), "polysemy": d.get("polysemy"),
            "description": d.get("description"),
            "locations": locations, "handshapes": handshapes,
            "video_url": f"{SITE}/{d['clip']}.mp4" if d.get("clip") else None,
            "has_own_video": False, "batch": "twtsl-ccu-v5", "source": SOURCE,
        })
        for s in (row["sentence"].get("Record") or []):
            key = (s.get("gloss"), s.get("translation"))
            if key in seen_sent or not s.get("gloss"):
                continue
            seen_sent.add(key)
            raw = s["gloss"]
            tokens = [t for t in re.split(r"/+", raw) if t]
            sents_out.append({
                "id": f"TWS{len(sents_out)+1:04d}", "type": "sentence",
                "twtsl_word_id": row["id"], "headword": canonical,
                "chinese": s.get("translation"),
                "gloss_raw": raw,
                "gloss": [split_variant(t)[0] for t in tokens],
                "gloss_text": "/".join(split_variant(t)[0] for t in tokens),
                "clauses": [c for c in raw.split("//") if c],
                "video_url": f"{SITE}/{s['clip']}.mp4" if s.get("clip") else None,
                "review_status": "pending",
                "batch": "twtsl-ccu-v5-例句", "source": SOURCE,
            })
    with (OUT / "twtsl_words.jsonl").open("w", encoding="utf-8") as f:
        for w in sorted(words_out, key=lambda x: x["twtsl_id"]):
            f.write(json.dumps(w, ensure_ascii=False) + "\n")
    with (OUT / "twtsl_sentences.jsonl").open("w", encoding="utf-8") as f:
        for s in sents_out:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"OK: {len(words_out)} 詞 → data/twtsl/twtsl_words.jsonl", flush=True)
    print(f"OK: {len(sents_out)} 例句 → data/twtsl/twtsl_sentences.jsonl", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if "--consolidate-only" not in sys.argv:
        words = enumerate_words()
        fetch_details(words)
    consolidate()


if __name__ == "__main__":
    main()
