#!/usr/bin/env python3
"""查詢文化部《臺灣手語語料庫》（https://tslcorpus.moc.gov.tw/，測試版）。

用途：合成句型前的語序查證（計畫 3.3 節步驟 A 的證據來源）。
語料庫內容為真實聾人演繹的對話／篇章，附臺灣手語 Gloss（Hand 欄）與中文對照
（Text 欄），是核對「中文→Gloss 語序」最直接的官方語料。

著作權：語料庫 © 文化部。本腳本僅按主題關鍵詞抓取少量例句文字作研究查證用，
輸出檔僅供內部核對句型，引用／發表須依網站著作權聲明辦理。

輸出：data/refs/tslcorpus_evidence.jsonl（每句含主題、關鍵詞、Gloss、中文、出處）
"""
import json
import time
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "data" / "refs"
API = "https://tslcorpus.moc.gov.tw/api/corpus/searchGetCorpusList"
SOURCE = "臺灣手語語料庫（測試版），文化部，https://tslcorpus.moc.gov.tw/"

THEMES = {
    "點餐": ["吃", "喝", "餐廳", "菜", "飲料", "好吃", "多少錢", "點餐"],
    "交通": ["公車", "火車", "捷運", "開車", "搭", "騎", "計程車", "高鐵"],
    "問路": ["哪裡", "怎麼走", "迷路", "附近", "地方", "找"],
    "日常對話": ["名字", "謝謝", "天氣", "幾點", "朋友", "工作", "家", "睡覺"],
    "看病": ["醫生", "醫院", "痛", "藥", "感冒", "生病", "掛號", "護士"],
}


def search(word, page=1, page_size=50):
    body = json.dumps({"word": word, "page": page, "pageSize": page_size}).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json",
        "User-Agent": "TSL-research (academic use; contact: chi931209@gmail.com)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seen, rows = set(), []
    for theme, words in THEMES.items():
        for w in words:
            try:
                data = search(w)
            except Exception as e:      # noqa: BLE001
                print(f"跳過 {w}: {e}")
                continue
            lst = (data.get("data") or {}).get("list") or []
            n = 0
            for item in lst:
                key = (item["ID"], item["TID"])
                if key in seen or not item.get("Hand"):
                    continue
                seen.add(key)
                rows.append({
                    "theme": theme, "keyword": w,
                    "chinese": item["Text"].strip(),
                    "gloss_raw": item["Hand"].strip(),
                    "gloss": [t for t in item["Hand"]
                              .replace("。", " ").replace("，", " ")
                              .replace("？", " ").replace("?", " ").split() if t],
                    "corpus_id": f"{item['ID']}/{item['TID']}",
                    "dialogue": item.get("name"),
                    "source": SOURCE,
                })
                n += 1
            print(f"{theme}／{w}: +{n}（累計 {len(rows)}）", flush=True)
            time.sleep(0.5)
    path = OUT / "tslcorpus_evidence.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"OK: {len(rows)} 句語序證據 → {path.relative_to(BASE)}")


if __name__ == "__main__":
    main()
