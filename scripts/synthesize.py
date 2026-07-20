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

增量輸出（v2）：
  - 既有 tsl_synth.jsonl 中的句子保留原 SYN 編號不變，新句接續編號。
  - 審核表：首次執行寫 review_sheet.xlsx；之後每次只把「新增句」寫到
    review_sheet_batch{N}.xlsx，不覆蓋老師填寫中的舊審核表。

詞彙兩層制（v2）：
  - 第一層：自有詞彙表 data/tsl_gloss_vocab.json（每個 Gloss 都有自有動作影片）。
  - 第二層：中正大學手語辭典 data/twtsl/twtsl_words.jsonl（scrape_twtsl.py 產出，
    每詞附辭典示範影片網址）。2026-07-20 分工確認：語言模型端只負責翻譯語序，
    影片由下游端自行爬取；entry 的 external_glosses 欄標示「下游需另爬影片」的詞。

輸出：data/synth/tsl_synth.jsonl、data/synth/review_sheet*.xlsx
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

# 外部詞彙槽位（中正大學手語辭典詞條，scrape_twtsl.py 產出後才可用；
# 這些詞有辭典示範影片、自有動作庫尚未收錄，合成句會標 external_glosses）
DRINKS = ["咖啡", "牛奶", "果汁", "紅茶", "珍珠奶茶"]
FOODS = ["水果", "蘋果", "香蕉", "麵包", "蛋糕", "早餐", "便當"]
DESTINATIONS = ["學校", "醫院", "公車站", "火車站"]  # 「車站」為火車站之同義索引名，不另立

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
    # ------------------------------------------------------------ batch2 新句型
    {"tid": "T10", "name": "否定定居句",
     "chinese": ["我不住在{P}。", "我沒有住在{P}。"],
     "gloss": ["我", "{P}", "住", "不"], "nms": NMS_NEG, "slots": PLACES,
     "confidence": "rule-derived",
     "rule_basis": "規則6 否定詞置動詞後（自有實例 S18 我不舒服→我/舒服/不）"
                   "＋規則2 定居句語序（P01）。自有資料無此組合實例，審核優先"},
    {"tid": "T11", "name": "情態移動疑問句",
     "chinese": ["你要去{P}嗎？", "你是不是要去{P}？"],
     "gloss": ["你", "去", "{P}", "要"], "nms": NMS_YESNO, "slots": PLACES,
     "confidence": "rule-derived",
     "rule_basis": "規則1 情態詞「要」句尾（T6／S14）＋規則3 是非問句 NMS 不比「嗎」"
                   "（P03）。自有資料無此組合實例，審核優先"},
    {"tid": "T12", "name": "時間＋情態移動疑問句",
     "chinese": ["你{T}要去{P}嗎？", "{T}你要去{P}嗎？"],
     "gloss": ["{T}", "你", "去", "{P}", "要"], "nms": NMS_YESNO,
     "slots": PLACES, "time_slots": TIMES,
     "confidence": "rule-derived",
     "rule_basis": "規則1＋規則3＋規則5 時間詞句首之三規則組合（P04 語序＋P03 NMS），"
                   "審核優先"},
    {"tid": "T13", "name": "WH 移動疑問句",
     "chinese": ["你要去哪裡？"],
     "gloss": ["你", "去", "哪裡", "要"], "nms": NMS_WH, "slots": None,
     "confidence": "rule-derived",
     "rule_basis": "規則1 情態詞句尾＋規則7 WH 句末；「哪裡」與「要」的相對順序由"
                   "規則7（WH 句末）與規則1（情態句尾）競合，暫依 S14 語序推導"
                   "「要」最尾，務必請老師確認，審核最優先"},
    {"tid": "T14", "name": "時間＋WH 移動疑問句",
     "chinese": ["你{T}要去哪裡？", "{T}你要去哪裡？"],
     "gloss": ["{T}", "你", "去", "哪裡", "要"], "nms": NMS_WH,
     "slots": None, "time_slots": TIMES,
     "confidence": "rule-derived",
     "rule_basis": "T13＋規則5 時間詞句首。與 T13 同屬順序競合句型，審核最優先"},
    {"tid": "T15", "name": "否定移動句（無時間詞）",
     "chinese": ["我不去{P}。"],
     "gloss": ["我", "{P}", "去", "不"], "nms": NMS_NEG, "slots": PLACES,
     "confidence": "rule-derived",
     "rule_basis": "T8 之無時間詞形：規則6 否定詞置動詞後（Jane Tsay 2021 "
                   "TODAY-I-SCHOOL-GO-NOT 去掉時間詞），審核優先"},
    {"tid": "T16", "name": "是非問句（上班）",
     "chinese": ["你在{P}上班嗎？", "你是不是在{P}上班？"],
     "gloss": ["你", "{P}", "上班"], "nms": NMS_YESNO, "slots": PLACES,
     "confidence": "rule-derived",
     "rule_basis": "規則2 定居類語序（P02 上班）＋規則3 是非問句 NMS（P03），"
                   "審核優先"},
    {"tid": "T17", "name": "情態疑問句（喝水）",
     "chinese": ["你要喝水嗎？"],
     "gloss": ["你", "水", "喝", "要"], "nms": NMS_YESNO, "slots": None,
     "confidence": "rule-derived",
     "rule_basis": "自有實例 S14 我要喝水→我/水/喝/要 換主語＋規則3 是非問句 NMS，"
                   "審核優先"},
    {"tid": "T18", "name": "情態疑問句（廁所）",
     "chinese": ["你要去廁所嗎？"],
     "gloss": ["你", "廁所", "去", "要"], "nms": NMS_YESNO, "slots": None,
     "confidence": "rule-derived",
     "rule_basis": "自有實例 S15 我要上廁所→我/廁所/去/要 換主語＋規則3，審核優先"},
    {"tid": "T19", "name": "WH 價錢句",
     "chinese": ["這個多少錢？", "這多少錢？"],
     "gloss": ["這", "多少錢"], "nms": NMS_WH, "slots": None,
     "confidence": "rule-derived",
     "rule_basis": "自有實例 S30 我要這個→我/這/要（這）＋S27 多少錢＋規則7 WH 句末，"
                   "審核優先"},
    # ------------------------------------------ batch2 外部詞彙模板（twtsl 辭典詞）
    {"tid": "T20", "name": "情態飲食句（喝）",
     "chinese": ["我要喝{P}。"],
     "gloss": ["我", "{P}", "喝", "要"], "nms": None, "slots": DRINKS,
     "confidence": "rule-derived",
     "rule_basis": "自有實例 S14 我要喝水→我/水/喝/要 之飲料槽位替換（規則1 情態句尾）；"
                   "槽位詞取自中正手語辭典，尚無自有影片，審核優先"},
    {"tid": "T21", "name": "情態飲食疑問句（喝）",
     "chinese": ["你要喝{P}嗎？"],
     "gloss": ["你", "{P}", "喝", "要"], "nms": NMS_YESNO, "slots": DRINKS,
     "confidence": "rule-derived",
     "rule_basis": "T20 換主語＋規則3 是非問句 NMS（P03），審核優先"},
    {"tid": "T22", "name": "情態飲食句（吃）",
     "chinese": ["我要吃{P}。"],
     "gloss": ["我", "{P}", "吃", "要"], "nms": None, "slots": FOODS,
     "confidence": "rule-derived",
     "rule_basis": "S14 句型之動詞（喝→吃）與賓語槽位類推（規則1）；「吃」與槽位詞"
                   "均取自中正手語辭典，尚無自有影片，審核優先"},
    {"tid": "T23", "name": "情態飲食疑問句（吃）",
     "chinese": ["你要吃{P}嗎？"],
     "gloss": ["你", "{P}", "吃", "要"], "nms": NMS_YESNO, "slots": FOODS,
     "confidence": "rule-derived",
     "rule_basis": "T22 換主語＋規則3 是非問句 NMS（P03），審核優先"},
    {"tid": "T24", "name": "情態移動句（場所）",
     "chinese": ["我要去{P}。"],
     "gloss": ["我", "去", "{P}", "要"], "nms": None, "slots": DESTINATIONS,
     "confidence": "rule-derived",
     "rule_basis": "T6 之場所槽位擴充（規則1）；「學校」見 Jane Tsay 2021 例句，"
                   "槽位詞取自中正手語辭典，尚無自有影片，審核優先"},
    {"tid": "T25", "name": "時間＋否定移動句（場所）",
     "chinese": ["我{T}不去{P}。", "{T}我不去{P}。"],
     "gloss": ["{T}", "我", "{P}", "去", "不"], "nms": NMS_NEG,
     "slots": DESTINATIONS, "time_slots": TIMES,
     "confidence": "rule-derived",
     "rule_basis": "T8 之場所槽位版；「今天/我/學校/去/不」即 Jane Tsay 2021 "
                   "TODAY-I-SCHOOL-GO-NOT 原例句，其餘場所為類推，審核優先"},
]


def load_existing():
    """讀既有真實資料：中文句集合（避免合成句與原句重複）＋兩層合法 Gloss 集合。

    回傳 (中文句集合, 自有詞彙集合, 外部詞彙集合)。
    外部詞彙＝中正大學手語辭典詞條（scrape_twtsl.py 產出），有辭典示範影片
    但自有動作庫尚未收錄；檔案不存在時為空集合。
    """
    chinese = set()
    for line in (DATA / "tsl_sentences.jsonl").read_text(encoding="utf-8").splitlines():
        chinese.add(json.loads(line)["chinese"])
    vocab = set(json.load((DATA / "tsl_gloss_vocab.json").open(encoding="utf-8"))["glosses"])
    external = set()
    twtsl = DATA / "twtsl" / "twtsl_words.jsonl"
    if twtsl.exists():
        for line in twtsl.read_text(encoding="utf-8").splitlines():
            external.add(json.loads(line)["gloss_text"])
    return chinese, vocab, external


def expand():
    existing_chinese, vocab, external = load_existing()
    entries, seen = [], set()
    for t in TEMPLATES:
        slot_vals = t["slots"] or [None]
        time_vals = t.get("time_slots") or [None]
        for p in slot_vals:
            for tm in time_vals:
                gloss = [g.replace("{P}", p or "").replace("{T}", tm or "")
                         for g in t["gloss"]]
                unknown = [g for g in gloss if g not in vocab and g not in external]
                if unknown:
                    hint = ("（外部詞彙檔 data/twtsl/twtsl_words.jsonl 不存在，"
                            "請先執行 scripts/scrape_twtsl.py）" if not external else "")
                    raise ValueError(f"{t['tid']} 產生了詞彙表外 Gloss: {unknown}{hint}")
                ext_used = [g for g in gloss if g not in vocab]
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
                        "external_glosses": ext_used,
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
              "規則依據", "外部詞彙（尚無自有影片）", "審核結果", "修正後 Gloss", "備註"]
    ws.append(header)
    for c in ws[2]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDDDDD")
    for e in entries:
        ws.append([e["id"], f"{e['template_id']} {e['template_name']}",
                   e["confidence"], e["chinese"], e["gloss_text"],
                   e["nms"] or "", e["rule_basis"],
                   "、".join(e.get("external_glosses") or []), "", "", ""])
    widths = [10, 22, 16, 26, 26, 30, 60, 22, 10, 18, 18]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = w
    ws.freeze_panes = "A3"
    wb.save(path)


def main():
    """增量輸出：既有句保留原 SYN 編號與內容，新句接續編號並另出批次審核表。"""
    OUT.mkdir(exist_ok=True)
    entries = expand()
    jsonl_path = OUT / "tsl_synth.jsonl"

    old = []
    if jsonl_path.exists():
        old = [json.loads(l) for l in
               jsonl_path.read_text(encoding="utf-8").splitlines()]
    old_keys = {(e["chinese"], e["gloss_text"]) for e in old}
    new_entries = [e for e in entries
                   if (e["chinese"], e["gloss_text"]) not in old_keys]

    sheets = [p for p in OUT.glob("review_sheet*.xlsx")
              if not p.name.startswith("~$")]
    batch_no = len(sheets) + 1
    next_num = max((int(e["id"][3:]) for e in old), default=0) + 1
    for e in new_entries:
        e["id"] = f"SYN{next_num:04d}"
        next_num += 1
        if old:
            e["batch"] = f"規則模板合成-batch{batch_no}"

    all_entries = old + new_entries
    with jsonl_path.open("w", encoding="utf-8") as f:
        for e in all_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    sheet_path = OUT / ("review_sheet.xlsx" if not old
                        else f"review_sheet_batch{batch_no}.xlsx")
    write_review_sheet(new_entries if old else all_entries, sheet_path)

    from collections import Counter
    by_t = Counter(e["template_id"] for e in new_entries)
    by_c = Counter(e["confidence"] for e in new_entries)
    ext = sum(1 for e in new_entries if e.get("external_glosses"))
    print(f"OK: 總計 {len(all_entries)} 句（既有 {len(old)}＋新增 {len(new_entries)}）"
          f"→ {jsonl_path.relative_to(BASE)}")
    print("新增各模板:", dict(sorted(by_t.items())))
    print("新增信心分布:", dict(by_c))
    if ext:
        print(f"含外部詞彙句數: {ext}（Gloss 尚無自有動作影片，見審核表欄位）")
    print(f"本批審核表: {sheet_path.relative_to(BASE)}")


if __name__ == "__main__":
    main()
