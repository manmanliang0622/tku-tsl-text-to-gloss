#!/usr/bin/env python3
"""以文化部語料庫真實語料實證合成規則的語序傾向（可重現稽核）。

動機（審核用）：synthesize.py 的 7 條語法規則原本以文獻＋團隊自蒐例句為據。
本腳本反向操作——用 5,272 句真實聾人語料（data/tslcorpus/parallel.jsonl）
統計標記詞的實際位置，檢驗規則是「文獻＋真實資料雙重支持」還是只是傾向。

方法：對含某類標記詞的句子，取「最後一個標記詞」的相對位置
      rel = idx/(len-1)，分句首段(≤0.34)／中段／句尾段(≥0.67)。
      這是純位置統計，不宣稱任何 NMS、手形或影片層級結論。

輸出：印出各規則的位置分布；不寫檔、不改資料。與審核報告數字對應。
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CORPUS = BASE / "data" / "tslcorpus" / "parallel.jsonl"

# 各規則對應的標記詞（審核可調整；列出即公開可查）
GROUPS = {
    "規則6 否定詞（預期偏句尾）": ["不", "沒有", "沒", "不要", "不能", "不行", "不用", "別"],
    "規則7 WH 疑問詞（預期偏句尾）": ["什麼", "哪裡", "誰", "幾點", "為什麼", "怎麼", "哪", "多少"],
    "規則5 時間詞（預期偏句首）": ["今天", "明天", "昨天", "現在", "以前", "最近", "早上", "晚上", "未來"],
    "規則1 情態詞（檢驗是否偏句尾）": ["要", "想", "可以", "能", "會", "應該"],
}


def pos_stats(rows, markers):
    early = mid = late = total = 0
    for r in rows:
        g = r["gloss"]
        if len(g) < 2:
            continue
        idxs = [i for i, t in enumerate(g) if any(m == t or m in t for m in markers)]
        if not idxs:
            continue
        total += 1
        rel = idxs[-1] / (len(g) - 1)
        if rel <= 0.34:
            early += 1
        elif rel >= 0.67:
            late += 1
        else:
            mid += 1
    return total, early, mid, late


def main():
    rows = [json.loads(l) for l in CORPUS.read_text(encoding="utf-8").splitlines()]
    print(f"語料庫 {len(rows)} 句 — 規則語序傾向實證（純位置統計）\n")
    for name, markers in GROUPS.items():
        total, early, mid, late = pos_stats(rows, markers)
        if not total:
            print(f"{name}: 無樣本")
            continue
        print(f"{name}")
        print(f"  n={total}｜句首段 {early/total*100:.0f}%  中段 {mid/total*100:.0f}%  "
              f"句尾段 {late/total*100:.0f}%")
    print("\n註：位置統計僅支持「常見傾向」，不代表無條件固定公式；"
          "WH／情態詞的句首比例正說明 T13/T14 等競合句須母語者逐句裁定。")


if __name__ == "__main__":
    main()
