#!/usr/bin/env python3
"""常用語句整句對照：句子完全命中已校訂的標準答案時，直接回標準腳本，不經模型。

動機（2026-09-14）：v21 線上把「請再說一次」譯成 再/說，漏掉「請」與「一次」，
而且旗標機率只有 0.00038（門檻 0.001814）不會攔下來。這句是 SLR_V2 常用語句
（每句 30 支實錄影片），標準答案 請/再/說/一次 早就在 data/tsl_sentences.jsonl。
v21 在這批 33 句非模板句上錯 10 句，其中 3 句都是句首「請」被吃掉
（請再說一次、請慢一點、請幫我）。

為什麼在服務端查表而不是重訓：這 33 句全部在 test 切分裡，拿去訓練就是
測試集洩漏；而且常用語句的正確性本來就不該取決於模型當天的 greedy 路徑。

守的界線：
  - **整句**命中才用。只正規化全半形、標點與空白（與盲測對帳的口徑一致），
    「請你再說一次」不算命中，照常走模型。
  - 模板句（`我叫＿`）不收——底線是要填的槽，不是手語。
  - 每個 gloss 都必須解得到、而且是**候選合格**的 sign_id（排除重複收錄與
    品質判定演不出動作的影片）。有一個解不到就整句不收，寧可交回模型，
    也不送出一個虛擬人比不出來的腳本。實測 P03 你住在宜蘭嗎 因「宜蘭」
    的影片不合格而不收。

只作用在線上服務；離線評估腳本刻意不套，模型指標保持可比。
**但透過 /translate 評估 test 集時，這些句子會變成必對**，要扣掉或關掉再評。
PHRASEBOOK=0 可整個關掉。
"""
import json
import unicodedata
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PHRASEBOOK_PATH = BASE / "data" / "tsl_sentences.jsonl"


def normalize(text: str) -> str:
    """全半形折疊（NFKC），去掉標點與空白。"""
    t = unicodedata.normalize("NFKC", str(text or ""))
    return "".join(c for c in t
                   if not unicodedata.category(c).startswith(("P", "Z", "C")))


def build(rows, resolve, eligible):
    """rows：tsl_sentences.jsonl 的列；resolve：gloss → sign_id 或 None；
    eligible：可進候選（＝演得出來）的 sign_id 集合。

    回傳 (表, 略過清單)。表的鍵是正規化後的中文，值是
    {"id","chinese","gloss","sign_ids"}；略過清單是 (id, 原因)。
    """
    table, skipped = {}, []
    for row in rows:
        rid = row.get("id")
        if row.get("is_template"):
            skipped.append((rid, "模板句"))
            continue
        key = normalize(row.get("chinese"))
        gloss = [str(g) for g in (row.get("gloss") or [])]
        if not key or not gloss:
            skipped.append((rid, "中文或 gloss 為空"))
            continue
        ids = [resolve(g) for g in gloss]
        bad = [g for g, i in zip(gloss, ids) if i is None or i not in eligible]
        if bad:
            skipped.append((rid, f"演不出來：{'、'.join(bad)}"))
            continue
        if key in table and table[key]["sign_ids"] != ids:
            # 同一句兩個標準答案＝資料本身有歧義，兩個都不採，交回模型
            skipped.append((rid, f"與 {table[key]['id']} 同句但答案不同"))
            table[key]["conflict"] = True
            continue
        table.setdefault(key, {"id": rid, "chinese": row.get("chinese"),
                               "gloss": gloss, "sign_ids": ids})
    for key in [k for k, v in table.items() if v.get("conflict")]:
        del table[key]
    return table, skipped


def load(retriever, path: Path = PHRASEBOOK_PATH):
    """用線上同一支 CandidateRetriever 解 sign_id，確保與候選的合格判定一致。"""
    if not Path(path).exists():
        return {}, [("-", f"找不到 {path}")]
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()
            if l.strip()]
    eligible = {r["sign_id"] for r in retriever.rows}
    return build(rows, retriever.resolve, eligible)


def lookup(table, text):
    return table.get(normalize(text))
