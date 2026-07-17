#!/usr/bin/env python3
"""規則模板資料合成（計畫 3.3 節步驟 A＋B）。

原則：
  1. 每個模板都以「自有標記表中實際存在的句型」＋「已查證的臺灣手語語法規則」為依據，
     規則出處見各模板的 rule_basis 欄位（對應計畫文件 3.3 節的 7 條規則）。
  2. 槽位只代入 Gloss 詞彙總表（data/tsl_gloss_vocab.json）內的詞彙，
     保證合成句的每個 Gloss 下游動作庫都有對應動作。
  3. 所有合成句 review_status=pending，須經手語老師／聾人顧問審核（計畫 3.3 節步驟 C）
     才能進入訓練集；同步產出 review_sheet.xlsx 審核表。
  4. confidence 分級：
     - attested-pattern：句型直接複製自有標記表已有句（僅替換地名槽位）
     - rule-derived   ：句型由查證過的語法規則組合推導，尚無自有影片實例，審核優先度最高

輸出：data/synth/tsl_synth.jsonl、data/synth/review_sheet.xlsx
"""
import json
from pathlib import Path

import openpyxl
from openpyxl.styles import Font, PatternFill

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT = DATA / "synth"

# ---------------------------------------------------------------- 詞彙槽位
# 縣市與離島（可自然接「住／上班／去／～人」）
CITIES = ["基隆", "台北", "新北", "桃園", "新竹", "苗栗", "台中", "南投",
          "彰化", "雲林", "嘉義", "台南", "高雄", "屏東", "宜蘭", "花蓮",
          "台東", "澎湖", "金門", "馬祖", "綠島", "蘭嶼"]
# 台北市內地區（住／上班／去 自然；「～人」不自然，故不用於身分句）
DISTRICTS = ["士林", "北投", "西門町", "中山", "淡水", "板橋", "中壢"]
PLACES = CITIES + DISTRICTS            # 29，住／上班／去
IDENTITY_PLACES = ["台灣"] + CITIES    # 23，「{P}人」身分句
TIMES = ["今天", "明天"]               # 時間詞（詞彙表內僅此二詞適用移動句）

NMS_YESNO = "疑問表情（眉毛上揚、身體微前傾）貫穿全句，不比「嗎」"
NMS_WH = "疑問表情（眼睛、眉毛、頭部姿勢）搭配句末疑問詞"
NMS_NEG = "否定可伴隨搖頭、眼睛或嘴部表情"

# ---------------------------------------------------------------- 模板定義
# chinese: {P}=地點 {T}=時間；每個模板列出中文說法變體（同一 Gloss 目標）
TEMPLATES = [
    {"tid": "T1", "name": "定居句（住）",
     "chinese": ["我住在{P}。", "我住{P}。"],
     "gloss": ["我", "{P}", "住"], "nms": None, "slots": PLACES,
     "confidence": "attested-pattern",
     "rule_basis": "規則2 無情態詞之定居類動詞：[主語]/地點/動詞（張榮興2008 空間語法；"
                   "自有實例 P01 我住在台北→我/台北/住、S26 我住在＿→我/＿地點/住）"},
    {"tid": "T2", "name": "定居句（上班）",
     "chinese": ["我在{P}上班。", "我在{P}工作。"],
     "gloss": ["我", "{P}", "上班"], "nms": None, "slots": PLACES,
     "confidence": "attested-pattern",
     "rule_basis": "規則2 同 T1（自有實例 P02 我在新竹上班→我/新竹/上班）。"
                   "「工作」為中文端同義改寫，Gloss 沿用「上班」，需審核確認"},
    {"tid": "T3", "name": "是非問句（住）",
     "chinese": ["你住在{P}嗎？", "你是不是住在{P}？"],
     "gloss": ["你", "{P}", "住"], "nms": NMS_YESNO, "slots": PLACES,
     "confidence": "attested-pattern",
     "rule_basis": "規則2＋規則3 是非問句不比「嗎」、改用非手部標記貫穿全句"
                   "（Tai & Tsay 2015；自有實例 P03 你住在宜蘭嗎→你/宜蘭/住＋NMS）"},
    {"tid": "T4", "name": "時間＋情態移動句",
     "chinese": ["我{T}要去{P}。", "{T}我要去{P}。"],
     "gloss": ["{T}", "我", "去", "{P}", "要"], "nms": None,
     "slots": PLACES, "time_slots": TIMES,
     "confidence": "attested-pattern",
     "rule_basis": "規則1 情態詞「要」句尾＋規則5 時間詞句首（母語者例句「我要去面試→"
                   "我/去/工作/要」；自有實例 P04 我明天要去花蓮→明天/我/去/花蓮/要）"},
    {"tid": "T5", "name": "身分判斷句",
     "chinese": ["我是{P}人。"],
     "gloss": ["我", "{P}", "人"], "nms": None, "slots": IDENTITY_PLACES,
     "confidence": "attested-pattern",
     "rule_basis": "規則4 判斷句不比「是」：[主語]/[地點]/[身分]"
                   "（教育部課綱臺灣手語；自有實例 P05 我是桃園人→我/桃園/人）"},
    {"tid": "T6", "name": "情態移動句（無時間詞）",
     "chinese": ["我要去{P}。"],
     "gloss": ["我", "去", "{P}", "要"], "nms": None, "slots": PLACES,
     "confidence": "attested-pattern",
     "rule_basis": "規則1 情態詞「要」句尾（母語者例句「我要去買→我/去/買/要」；"
                   "T4 之無時間詞形，自有實例 S14 我要喝水→我/水/喝/要 同屬「要」句尾）"},
    {"tid": "T7", "name": "是非問句（身分）",
     "chinese": ["你是{P}人嗎？"],
     "gloss": ["你", "{P}", "人"], "nms": NMS_YESNO, "slots": IDENTITY_PLACES,
     "confidence": "rule-derived",
     "rule_basis": "規則4（P05 句型）＋規則3（P03 是非問句 NMS）之組合推導，"
                   "自有資料無此句型實例，審核優先"},
    {"tid": "T8", "name": "時間＋否定移動句",
     "chinese": ["我{T}不去{P}。", "{T}我不去{P}。"],
     "gloss": ["{T}", "我", "{P}", "去", "不"], "nms": NMS_NEG,
     "slots": PLACES, "time_slots": TIMES,
     "confidence": "rule-derived",
     "rule_basis": "規則6 否定詞置動詞後/句尾＋規則5 時間詞句首，直接類比 Jane Tsay 2021 "
                   "例句 TODAY-I-SCHOOL-GO-NOT（今天/我/學校/去/不），學校→地名；"
                   "無情態詞故地點在動詞前（v1.2 母語例句歸納）。自有資料無實例，審核優先"},
    {"tid": "T9", "name": "WH 疑問句（上班）",
     "chinese": ["你在哪裡上班？"],
     "gloss": ["你", "上班", "哪裡"], "nms": NMS_WH, "slots": None,
     "confidence": "rule-derived",
     "rule_basis": "規則7 WH 疑問詞置句末＋疑問 NMS（Tai & Tsay 2015），"
                   "類比自有實例 S25 你住哪裡→你/住/哪裡。自有資料無實例，審核優先"},
]


def load_existing():
    """讀既有真實資料：中文句集合（避免合成句與原句重複）＋合法 Gloss 集合。"""
    chinese = set()
    for line in (DATA / "tsl_sentences.jsonl").read_text(encoding="utf-8").splitlines():
        chinese.add(json.loads(line)["chinese"])
    vocab = set(json.load((DATA / "tsl_gloss_vocab.json").open(encoding="utf-8"))["glosses"])
    return chinese, vocab


def expand():
    existing_chinese, vocab = load_existing()
    entries, seen = [], set()
    for t in TEMPLATES:
        slot_vals = t["slots"] or [None]
        time_vals = t.get("time_slots") or [None]
        for p in slot_vals:
            for tm in time_vals:
                gloss = [g.replace("{P}", p or "").replace("{T}", tm or "")
                         for g in t["gloss"]]
                unknown = [g for g in gloss if g not in vocab]
                if unknown:
                    raise ValueError(f"{t['tid']} 產生了詞彙表外 Gloss: {unknown}")
                for vi, ch_tpl in enumerate(t["chinese"]):
                    ch = ch_tpl.replace("{P}", p or "").replace("{T}", tm or "")
                    if ch in existing_chinese:   # 與真實資料重複（如 T1×台北=P01）
                        continue
                    key = (ch, "/".join(gloss))
                    if key in seen:
                        continue
                    seen.add(key)
                    entries.append({
                        "id": f"SYN{len(entries)+1:04d}",
                        "type": "sentence",
                        "chinese": ch,
                        "gloss": gloss,
                        "gloss_text": "/".join(gloss),
                        "nms": t["nms"],
                        "template_id": t["tid"],
                        "template_name": t["name"],
                        "is_paraphrase": vi > 0,
                        "confidence": t["confidence"],
                        "rule_basis": t["rule_basis"],
                        "review_status": "pending",
                        "batch": "規則模板合成",
                    })
    return entries


def write_review_sheet(entries, path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "審核表"
    ws.append(["合成句審核表 — 請手語老師／聾人顧問逐句確認 Gloss 與 NMS "
               "（審核結果填：通過／修正／不通過；rule-derived 句型請優先審核）"])
    header = ["編號", "模板", "信心等級", "中文", "TSL Gloss", "NMS",
              "規則依據", "審核結果", "修正後 Gloss", "備註"]
    ws.append(header)
    for c in ws[2]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDDDDD")
    for e in entries:
        ws.append([e["id"], f"{e['template_id']} {e['template_name']}",
                   e["confidence"], e["chinese"], e["gloss_text"],
                   e["nms"] or "", e["rule_basis"], "", "", ""])
    widths = [10, 22, 16, 26, 26, 30, 60, 10, 18, 18]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = w
    ws.freeze_panes = "A3"
    wb.save(path)


def main():
    OUT.mkdir(exist_ok=True)
    entries = expand()
    jsonl_path = OUT / "tsl_synth.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    write_review_sheet(entries, OUT / "review_sheet.xlsx")

    from collections import Counter
    by_t = Counter(e["template_id"] for e in entries)
    by_c = Counter(e["confidence"] for e in entries)
    print(f"OK: 合成 {len(entries)} 句 → {jsonl_path.relative_to(BASE)}")
    print("各模板:", dict(sorted(by_t.items())))
    print("信心分布:", dict(by_c))
    print(f"審核表: {(OUT / 'review_sheet.xlsx').relative_to(BASE)}")


if __name__ == "__main__":
    main()
