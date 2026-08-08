#!/usr/bin/env python3
"""把切分資料轉成「結構化 JSON 目標」格式，供訓練會輸出語法標註的模型。

Schema 沿用同機 `0804try` 專案 `src/signavatar/tku_json_dataset.py` 的欄位定義，
以維持兩邊模型可直接對照、且前端可共用同一組解析程式：

  {"gloss":"我 台北 住","subject":"none","object":"none","time":"none",
   "topic":"我","verb":"住","verb_type":"unknown","agreement":"none",
   "question_type":"none","negation":false,"nonmanual":"none"}

為什麼要這個格式：純 Gloss 無法承載非手部標記（NMS），而計畫第 1 節指出
下游虛擬人的「表情、頭部、身體同步」需要 NMS。JSON 目標讓模型一次輸出
Gloss ＋ 疑問/否定/NMS，下游不必再猜。

本專案的兩處改良（其餘欄位邏輯與 0804try 相同）：
  1. **nonmanual 用真實標註**：0804try 恆填 "none"；本專案 synth 與 P03 共
     約 530 句（train）帶人工 NMS 文字，直接採用，其餘才回退啟發式。
  2. **verb 不取句末 token**：0804try 直接取最後一個 gloss；但本專案實測
     文化部語料庫「要」有 66% 位於句末、否定詞亦常在句末，直接取末位會把
     情態/否定詞誤標成動詞。故先剔除句末的情態與否定詞再取。

用法：
  python3 scripts/build_json_targets.py                 # 轉 train/dev/test
  python3 scripts/build_json_targets.py --splits train  # 只轉指定
輸出：data/splits_json/{split}.jsonl（instruction/input/output 三欄）
"""
import argparse
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SPLITS = BASE / "data" / "splits"
OUT = BASE / "data" / "splits_json"

INSTRUCTION = "請將中文句子轉成台灣自然手語語序，並標出語法資訊。"

TIME_HINTS = ("以前", "以後", "之前", "之後", "期間", "時候", "昨天", "今天", "明天",
              "早上", "中午", "晚上", "去年", "今年", "明年", "六個月", "月", "年",
              "週", "天", "點", "分", "最近", "現在")
# WH 判定（2026-08-07 修正）：原本把「幾」當無條件疑問詞，導致「前幾天」「好幾次」
# 「二十幾年」等被誤標為 wh 疑問句。實測訓練資料 42 句含「幾」，其中 31 句（74%）
# 並非疑問句 → 標籤錯誤率極高，模型只是忠實照學。
# 改法：①「幾」需排除已知非疑問固定用法；②疑問標點／語氣詞為強訊號。
WH_HINTS = ("誰", "什麼", "哪裡", "哪個", "哪一", "為什麼", "怎麼", "多少")
WH_AMBIGUOUS = ("幾",)          # 需排除固定用法後才算疑問
NON_WH_PATTERNS = re.compile(
    r"前幾|好幾|這幾|那幾|上幾|下幾|幾乎|十幾|廿幾|\d+\s*幾|幾百|幾千|幾萬|零星幾")
QUESTION_MARKS = re.compile(r"[?？]")
YESNO_HINTS = ("嗎", "是不是", "是否", "有沒有", "可不可以")
NEGATION_HINTS = ("不", "沒有", "沒", "無法", "不要", "不能", "沒辦法")
# 句末常見的情態／否定詞：取「主要動詞」時要先剔除（本專案語料實測）
TAIL_NON_VERBS = {"要", "想", "可以", "不", "不要", "沒有", "會", "能", "有", "了"}


def norm_gloss(gloss_text):
    t = str(gloss_text or "").strip()
    if "/" in t:
        return " ".join(x.strip() for x in t.split("/") if x.strip())
    return " ".join(t.split())


def infer_time(tokens):
    m = [t for t in tokens if any(h in t for h in TIME_HINTS)]
    return " ".join(m) if m else "none"


def infer_question_type(chinese, tokens):
    """判斷疑問類型。見 WH_HINTS 上方註解說明「幾」的誤判問題與修正依據。"""
    combined = chinese + "".join(tokens)
    has_mark = bool(QUESTION_MARKS.search(chinese))
    has_yesno = any(h in combined for h in YESNO_HINTS)

    # 明確的 WH 詞
    if any(h in combined for h in WH_HINTS):
        return "wh"
    # 「幾」：先剔除固定用法，再看是否有疑問標點／語氣詞佐證
    if any(h in combined for h in WH_AMBIGUOUS):
        stripped = NON_WH_PATTERNS.sub("", combined)
        if any(h in stripped for h in WH_AMBIGUOUS) and (has_mark or has_yesno):
            return "wh"
    if has_yesno:
        return "yesno"
    return "none"


def infer_negation(chinese, tokens):
    return any(h in (chinese + "".join(tokens)) for h in NEGATION_HINTS)


def infer_verb(tokens):
    """由句末往前找第一個非情態/否定詞者當主要動詞（見檔頭改良 2）。"""
    for t in reversed(tokens):
        if t not in TAIL_NON_VERBS:
            return t
    return tokens[-1] if tokens else "none"


def infer_nonmanual(row, question_type, negation):
    """優先採用人工 NMS 標註；沒有才依疑問/否定回退（見檔頭改良 1）。"""
    nms = row.get("nms")
    if isinstance(nms, list):
        nms = nms[0] if nms else None
    if nms:
        return str(nms)
    if question_type == "yesno":
        return "疑問表情（眉毛上揚）貫穿全句"
    if question_type == "wh":
        return "疑問表情搭配句末疑問詞"
    if negation:
        return "否定可伴隨搖頭或嘴部表情"
    return "none"


def convert(row):
    chinese = str(row.get("chinese") or "").strip().strip("，。！？；：,.!?;: ")
    gloss = norm_gloss(row.get("gloss_text"))
    if not chinese or not gloss:
        return None
    tokens = gloss.split()
    qt = infer_question_type(chinese, tokens)
    neg = infer_negation(chinese, tokens)
    out = {
        "gloss": gloss,
        "subject": "none",
        "object": "none",
        "time": infer_time(tokens),
        "topic": tokens[0] if tokens else "none",
        "verb": infer_verb(tokens),
        "verb_type": "unknown",
        "agreement": "none",
        "question_type": qt,
        "negation": neg,
        "nonmanual": infer_nonmanual(row, qt, neg),
    }
    rec = {"instruction": INSTRUCTION, "input": chinese,
           "output": json.dumps(out, ensure_ascii=False, separators=(",", ":"))}
    if row.get("context"):
        rec["context"] = row["context"]     # 上下文翻譯（SCOPE 路線）用
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in args.splits:
        src = SPLITS / f"{name}.jsonl"
        if not src.exists():
            print(f"  略過 {name}（找不到 {src}）")
            continue
        rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
        recs, real_nms = [], 0
        for r in rows:
            c = convert(r)
            if c:
                recs.append(c)
                if r.get("nms"):
                    real_nms += 1
        path = OUT / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for c in recs:
                f.write(json.dumps(c, ensure_ascii=False, separators=(",", ":")) + "\n")
        print(f"  {name}: {len(recs)} 句（其中 {real_nms} 句用真實 NMS 標註）→ {path.relative_to(BASE)}")


if __name__ == "__main__":
    main()
