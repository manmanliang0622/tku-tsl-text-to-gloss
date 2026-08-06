#!/usr/bin/env python3
"""人工修正回饋：把「模型翻錯的句子＋正確 Gloss」加入訓練資料。

用途：在前端測到語序錯誤時，把正確答案記下來，重訓後模型即可學到。

設計重點（2026-08-06）：
  - **加權重複**：少數幾句修正混在 6,102 句訓練資料裡幾乎沒有影響力，
    故每筆修正預設複製 `weight` 份（預設 5）進訓練集，讓它真的被學到。
  - **模板展開**：若修正的是「某個句型」而非「某個句子」，用 --template 標記，
    再用 scripts/synthesize.py 依相同句型批次生成同類句，才能舉一反三。
  - 修正檔是純文字 JSONL，可直接用編輯器手動增修。

用法：
  # 1) 先問模型現在怎麼翻（會記下錯誤輸出，方便日後檢視改善）
  python3 scripts/add_correction.py --chinese "你可以帶孩子去室內泳池玩水" \\
      --gloss "孩子/你/帶去/去/室內/游泳池/玩/可以" --note "受詞前置"

  # 2) 不連模型、純手動記錄
  python3 scripts/add_correction.py --chinese "..." --gloss "..." --no-query

  # 3) 檢視目前所有修正
  python3 scripts/add_correction.py --list

修正檔：data/corrections/corrections.jsonl
"""
import argparse
import json
import urllib.request
from datetime import date
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "data" / "corrections"
PATH = OUT / "corrections.jsonl"
API = "http://127.0.0.1:8018/translate"


def query_model(text, timeout=120):
    try:
        body = json.dumps({"text": text}).encode()
        req = urllib.request.Request(API, data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read()).get("gloss_text")
    except Exception as e:
        print(f"  （連不上模型 API，略過記錄錯誤輸出：{e}）")
        return None


def load():
    if not PATH.exists():
        return []
    return [json.loads(l) for l in PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chinese")
    ap.add_argument("--gloss", help="正確 Gloss，以 / 分隔")
    ap.add_argument("--note", default="", help="錯誤類型備註，如「受詞前置」")
    ap.add_argument("--weight", type=int, default=5,
                    help="進訓練集時複製幾份（預設 5；少數修正需加權才學得到）")
    ap.add_argument("--template", default="",
                    help="此修正代表的句型代號，供日後用 synthesize.py 批次展開")
    ap.add_argument("--no-query", action="store_true", help="不呼叫模型記錄錯誤輸出")
    ap.add_argument("--list", action="store_true", help="列出現有修正")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rows = load()

    if args.list:
        print(f"目前 {len(rows)} 筆修正：")
        for r in rows:
            print(f"  [{r['id']}] w={r.get('weight',1)} {r['chinese']}")
            print(f"        正確: {r['gloss_text']}")
            if r.get("model_output_at_add"):
                print(f"        當時模型: {r['model_output_at_add']}")
        return

    if not args.chinese or not args.gloss:
        ap.error("需要 --chinese 與 --gloss（或用 --list）")

    gloss = [t for t in args.gloss.replace("／", "/").split("/") if t.strip()]
    if len(gloss) < 2:
        ap.error("Gloss 至少要兩個詞，並以 / 分隔")

    wrong = None if args.no_query else query_model(args.chinese)
    entry = {
        "id": f"FIX{len(rows)+1:04d}",
        "type": "sentence",
        "chinese": args.chinese.strip(),
        "gloss": gloss,
        "gloss_text": "/".join(gloss),
        "nms": None,
        "note": args.note,
        "template_id": args.template,
        "weight": max(1, args.weight),
        "model_output_at_add": wrong,
        "added": date.today().isoformat(),
        "batch": "人工修正回饋",
    }
    with PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"已加入 {entry['id']}（權重 {entry['weight']}）")
    print(f"  中文: {entry['chinese']}")
    print(f"  正確: {entry['gloss_text']}")
    if wrong:
        print(f"  修正前模型輸出: {wrong}")
    print(f"\n目前共 {len(rows)+1} 筆修正。重訓即可生效：")
    print("  python3 scripts/split_data.py --use-all")
    print("  python3 scripts/train_qlora.py --output outputs/qlora_e4b_v7 --epochs 2 ...")


if __name__ == "__main__":
    main()
