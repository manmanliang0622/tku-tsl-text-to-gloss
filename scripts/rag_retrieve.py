#!/usr/bin/env python3
"""RAG 檢索：從訓練資料找出與輸入句最相似的例句，供推理時放進 prompt。

依據：CCL24-Eval 任務10 用「知識庫語料＋專家示例」做 in-context learning；
工研院 ICT Journal「AI手語-虛擬氣象主播」用 RAG 解低頻詞問題。兩篇參考文獻
都採此路線（見計畫第 2 節文獻表）。

為什麼對本專案有用（2026-08-07 診斷）：實測發現模型連**訓練集內**的長句都
復現不出（0/6），例如「前幾天我喝水、吃東西的時候覺得牙齒很痛耶」的正確
Gloss 就在 train 裡，模型仍翻錯。微調把知識壓進權重會遺失細節，檢索則是把
原文重新放到模型眼前。

檢索方法（無外部套件；共用機不裝額外依賴）：
  中文字元 bigram 的 Jaccard 相似度。中文無空白分詞，bigram 是常見且穩健的
  輕量近似；不需訓練、不需 embedding 模型，毫秒級完成。

用法：
  python3 scripts/rag_retrieve.py "前幾天我喝水、吃東西的時候覺得牙齒很痛耶"
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def bigrams(text):
    t = "".join(str(text).split())
    return {t[i:i + 2] for i in range(len(t) - 1)} or {t}


class Retriever:
    def __init__(self, split="train"):
        path = BASE / "data" / "splits" / f"{split}.jsonl"
        self.rows = [json.loads(l) for l in
                     path.read_text(encoding="utf-8").splitlines() if l.strip()]
        self._bg = [bigrams(r["chinese"]) for r in self.rows]

    def search(self, query, k=3, min_score=0.0):
        q = bigrams(query)
        scored = []
        for r, bg in zip(self.rows, self._bg):
            inter = len(q & bg)
            if not inter:
                continue
            score = inter / len(q | bg)          # Jaccard
            if score >= min_score:
                scored.append((score, r))
        scored.sort(key=lambda x: -x[0])
        return [(round(s, 4), r) for s, r in scored[:k]]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    hits = Retriever().search(query, k=k)
    print(f"查詢：{query}\n")
    for score, r in hits:
        print(f"  [{score}] {r['chinese']}")
        print(f"          → {r['gloss_text']}   （{r['source']}）")


if __name__ == "__main__":
    main()
