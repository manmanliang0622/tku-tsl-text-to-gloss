#!/usr/bin/env python3
"""爬教育部常用手語辭典（https://special.moe.gov.tw/signlanguage）的詞條。

**為什麼要這個來源**（2026-08-12 三層診斷 + 涵蓋率實測）：
模型失敗的病因是詞彙覆蓋不足、不會的詞就照抄中文。但實測發現既有最大詞彙
資源「中正大學台灣手語線上辭典」（3,500 詞）**只涵蓋 28 個問題詞中的 1 個**；
教育部辭典則涵蓋 8 個（診所、社區、鬧鐘、寶寶、警報、概念、安定、畫家），
是目前唯一還沒開發、且實測有效的新詞彙來源。

**API**（由瀏覽器網路請求觀察得到，非官方文件）：
    GET /signlanguage/api/Vocabularies?key=<關鍵字>
回傳 JSON 陣列，每筆有 title（中文詞）、description（英譯）、tags（分類）、
key（階層編號如 vocabulary/08/0137）、youtubeKey（示範影片）、
isCommon/isAdvance（常用/進階）、items（打法變體）。

**列舉方式**：該 API 為子字串比對（實測 key=「所」回 33 筆，含診所、廁所、
托兒所…），故以「單字」逐一查詢再取聯集即可枚舉。字集預設取自本專案語料庫
中文句與 Gloss 詞彙的所有漢字——只查我們領域內用得到的字，比盲查全字集有效率。

⚠️ **授權未查證**：文化部語料庫與中正辭典的訓練＋散布授權已於 2026-08-04
查證合法（標明出處即可）；本站頁尾為「Copyright © 2025 教育部常用手語辭典
All Right Reserved.」，站上未見開放資料聲明。**納入訓練或公開散布前需先確認
授權**。本腳本預設輸出到 data/moe/（未納入版本控制），僅供查證後再決定。

用法：
  python3 scripts/scrape_moe_signdict.py                # 完整爬取（約 15–25 分）
  python3 scripts/scrape_moe_signdict.py --chars 診社鬧  # 只查指定字（測試用）
  python3 scripts/scrape_moe_signdict.py --delay 0.5     # 放慢請求
"""
import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "data" / "moe"
API = "https://special.moe.gov.tw/signlanguage/api/Vocabularies?key={}"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 " \
     "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
SOURCE = ("教育部常用手語辭典 https://special.moe.gov.tw/signlanguage"
          "（授權未查證，見本檔說明）")


def domain_chars():
    """取本專案語料庫中文句與 Gloss 詞彙用到的所有漢字。

    只查領域內的字：盲查全部常用字（約 5,000）多半查不到東西，
    用語料庫的字集能把請求數壓到必要範圍。
    """
    chars = set()
    for rel in ("data/splits_json/train.jsonl",
                "data/splits_json/test_corpus.jsonl",
                "data/splits_json/test_papers.jsonl"):
        p = BASE / rel
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            text = r["input"] + json.loads(r["output"])["gloss"]
            chars.update(ch for ch in text if "一" <= ch <= "鿿")
    return sorted(chars)


def fetch(key, delay, retries=3):
    url = API.format(urllib.parse.quote(key))
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
            if attempt == retries - 1:
                print(f"    ⚠ {key}: {e}")
                return []
            time.sleep(delay * (attempt + 2))
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", default=None, help="只查這些字（預設用語料庫字集）")
    ap.add_argument("--delay", type=float, default=0.35, help="每次請求間隔秒數")
    ap.add_argument("--out", default=str(OUT_DIR / "moe_words.jsonl"))
    args = ap.parse_args()

    chars = list(args.chars) if args.chars else domain_chars()
    print(f"[爬取] 查詢字集 {len(chars)} 字，間隔 {args.delay}s，"
          f"預估 {len(chars) * args.delay / 60:.0f} 分鐘")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 續爬：已抓過的字跳過，中斷可直接重跑
    done_path = out_path.with_suffix(".progress.json")
    done = set(json.loads(done_path.read_text(encoding="utf-8"))) if done_path.exists() else set()
    words = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                words[r["id"]] = r

    for i, ch in enumerate(chars, 1):
        if ch in done:
            continue
        for entry in fetch(ch, args.delay) or []:
            if not entry.get("title"):
                continue
            words[entry["id"]] = {
                "id": entry["id"],
                "chinese": entry["title"],
                "english": entry.get("description"),
                "tags": entry.get("tags"),
                "key": entry.get("key"),
                "is_common": entry.get("isCommon"),
                "is_advance": entry.get("isAdvance"),
                "youtube_key": entry.get("youtubeKey"),
                "variants": [it.get("title") for it in (entry.get("items") or [])
                             if it.get("title")],
                "source": SOURCE,
            }
        done.add(ch)
        if i % 25 == 0 or i == len(chars):
            with out_path.open("w", encoding="utf-8") as f:
                for r in words.values():
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            done_path.write_text(json.dumps(sorted(done), ensure_ascii=False),
                                 encoding="utf-8")
            print(f"  [{i}/{len(chars)}] 字「{ch}」…累計詞條 {len(words)}", flush=True)
        time.sleep(args.delay)

    with out_path.open("w", encoding="utf-8") as f:
        for r in words.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    done_path.write_text(json.dumps(sorted(done), ensure_ascii=False), encoding="utf-8")
    print(f"\n完成：{len(words)} 個詞條 → {out_path.relative_to(BASE)}")
    print("⚠️ 授權未查證，納入訓練或公開散布前請先確認（見本檔開頭說明）")


if __name__ == "__main__":
    main()
