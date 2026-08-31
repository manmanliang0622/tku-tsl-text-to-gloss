#!/usr/bin/env python3
"""伴隨句雙數標記：中文「我跟X…」→ Gloss 補上「我們兩個」。

為什麼要有這一支（2026-08-31）：
    「我跟媽媽去吃飯」線上吐出 `我/媽媽/吃飯`，少了雙數收攏詞。查下來不是
    模型學不會，是**候選清單裡根本沒有那個詞**——k=40 的候選檢索對這句
    撈不到「我們兩個」，約束解碼就不可能吐出它（實測：
    `我跟朋友去吃飯` 撈得到「兩人去」，`我跟媽媽去吃飯` 兩個都撈不到）。

    修候選端會動到訓練分布。本專案已經有三個否定結果（n_syn／n_sem／
    pin_core，見 sign_candidates.candidates 的說明）指出固定 k 之下加通道
    是零和且淨負的，故這裡走**解碼後補寫**，完全不碰候選與訓練契約。

語料依據（data/tslcorpus/parallel.jsonl，5,272 句）：
  - 「我們兩個」49 次，是「我＋某人」的標準收攏詞，位置在**兩個人名詞之後**：
      TC00444 我和太太在找…      → 我/太太/我們兩個/公園/大/找/…
      TC01211 我和朋友找工作…    → 我/朋友/我們兩個/找/工作/…
      TC00849 我跟孩子聊天…      → 孩子/我們兩個/聊天/…
  - 「兩人去」只有 3 次（＋「二人去」1 次），且**全部是真的有移動**
    （去博物館／去公園），它是「去」的分類詞動詞而非數量詞，故本規則不補它。

為什麼不是「看到跟就補」：
    全庫 122 句含「跟」，其中 39 句是「跟X說」這種單向言談（語料用有方向性
    的「告訴」，不收攏）、13 句是「跟著／跟隨／跟團」、1 句是「腳跟」。
    無差別補寫會弄壞這 53 句。故排除條件是這支程式的主體，不是裝飾。

只用標準函式庫；serve_model.py 與離線評估共用。
"""
from __future__ import annotations

import re

# 收攏詞本身。sign_id 由呼叫端用 gloss→id 索引解析，這裡只認 gloss。
DUAL_GLOSS = "我們兩個"

# 已經有這些詞就不補：模型自己收攏過了，再加一個是重複。
_ALREADY = {"我們兩個", "我們兩人", "我們", "我們三人", "兩人", "兩人去", "二人去",
            "兩個人", "兩個人一起", "他們兩個", "她們兩個"}

# 「跟」不是伴隨的用法，逐條對應語料庫實例。命中任一條就整句放棄。
_EXCLUDE = [
    r"腳跟",                                  # TC00154 腳跟底部
    r"跟(著|隨|團|上|不上|進)",                # TC00720 我也跟著回頭看 → 我/一樣/回頭看
    # 單向言談：語料用有方向性的「告訴」，不收攏（TC01942 我跟你說 → 我/告訴）
    r"(跟|和|與|同)[^，。！？]{0,8}?(說|講|告訴|報告|提起|抱怨|表達|反應|道歉|道謝|求婚|請假)",
    # 比較句：TC00329 這個味道跟媽媽煮的一樣 → …/一模一樣/…
    r"(跟|和|與|同)[^，。！？]{0,10}?(一樣|不一樣|相同|不同|差不多|相比|比起來|比較)",
    # 取得類單向動作：TC00545 我只好跟聽人同學借筆記 → 我/只好/聽人/同學/筆記/借/學
    r"(跟|和|與|同)[^，。！？]{0,8}?(借|要錢|討|拿|學藝|買)",
]
_EXCLUDE_RE = [re.compile(p) for p in _EXCLUDE]

# 三人以上就不是「兩個」。命中就放棄（語料另有「我們三人」「他們三人」，
# 但判斷人數要數 NP，規則做不準，寧可不補）。TC01302 我和朋友三個人視訊聊天。
# ⚠️ 只能拿去比對**伴隨對象那一段**，不能比對整句：中文沒有詞界，`們` 這條
# 對整句比會被句首的「我們」自己命中，規則等於永遠不觸發。
_PLURAL_RE = re.compile(r"(三|四|五|六|七|八|九|十|幾|好幾|數|多)\s*(個|位|人|名)|們|大家|各位")

# 人物名詞：只有伴隨對象是人，「我們兩個」才成立。
# 「我跟讀書和打工之間」這種不能補（TC00441）。
_PERSON = [
    "媽媽", "爸爸", "母親", "父親", "爸", "媽", "哥哥", "姐姐", "姊姊", "弟弟", "妹妹",
    "哥", "姐", "姊", "弟", "妹", "兒子", "女兒", "孫子", "孫女", "阿公", "阿嬤",
    "爺爺", "奶奶", "外公", "外婆", "叔叔", "阿姨", "舅舅", "姑姑", "伯伯", "表哥",
    "表姐", "堂哥", "堂姐", "先生", "太太", "老公", "老婆", "丈夫",
    "妻子", "男友", "女友", "男朋友", "女朋友", "小孩", "孩子", "寶寶",
    "朋友", "同學", "同事", "室友", "學長", "學姐", "學弟", "學妹", "老師", "教授",
    "醫生", "醫師", "護理師", "翻譯員", "手語翻譯員", "老闆", "主管", "客人", "鄰居",
    "聾人", "聽人", "聾友", "同伴", "夥伴", "隊友", "他", "她", "你", "妳", "您",
]
# 刻意不收：家人／親戚／大家／同事們——人數不定，語料庫也沒把它們收攏成雙數
# （TC03600 我和家人一起去旅行 → 我/家人/一起/旅行）。
# 長詞優先，避免「男朋友」被「朋友」搶先切走造成位置判斷偏移。
_PERSON.sort(key=len, reverse=True)

# 伴隨介詞。「同」單獨太容易誤命中（同學、同事、相同），只在「同」後面直接
# 接人物名詞時才算，交給下面的比對順序處理。
_WITH_RE = re.compile(r"(我們|我)\s*(跟|和|與|同)\s*([^，。！？、]{1,12})")


def _person_in(chunk: str) -> str | None:
    """chunk 開頭是不是人物名詞。回傳命中的詞，否則 None。"""
    for p in _PERSON:
        if chunk.startswith(p):
            return p
    # 「我跟我的高中同學」：中間夾修飾語，退一步找整段裡的人物名詞（TC00408）。
    for p in _PERSON:
        if p in chunk:
            return p
    return None


def detect(chinese: str) -> str | None:
    """句子是不是「我＋一個人」的伴隨句。是就回傳那個人物名詞，否則 None。"""
    text = str(chinese or "")
    if not text:
        return None
    for rx in _EXCLUDE_RE:
        if rx.search(text):
            return None
    m = _WITH_RE.search(text)
    if not m:
        return None
    chunk = m.group(3)
    if _PLURAL_RE.search(chunk):        # 只看伴隨對象，理由見 _PLURAL_RE
        return None
    return _person_in(chunk)


def apply(chinese: str, glosses: list[str], max_signs: int | None = None) -> tuple[list[str], bool]:
    """在 gloss 串列裡補上「我們兩個」。回傳 (新串列, 是否有改動)。

    只有在**模型自己已經把兩個人都打出來**時才補——這條是最強的守門條件：
    模型沒產出主語就沒有東西可以收攏，硬補只會生出無主語的怪句。
    位置依語料庫慣例放在兩個人物 gloss 中較後面那個之後。
    """
    toks = list(glosses)
    if DUAL_GLOSS in toks or _ALREADY & set(toks):
        return toks, False
    if max_signs is not None and len(toks) >= max_signs:
        return toks, False
    person = detect(chinese)
    if not person:
        return toks, False
    if "我" not in toks or person not in toks:
        return toks, False
    pos = max(toks.index("我"), toks.index(person))
    toks.insert(pos + 1, DUAL_GLOSS)
    return toks, True


def apply_ids(chinese: str, sign_ids: list[str], glosses: list[str],
              gloss_to_id: dict, max_signs: int | None = None):
    """同時維護 sign_ids 與 gloss 兩份平行串列。回傳 (sign_ids, glosses, 是否改動)。

    gloss_to_id 查不到「我們兩個」時一律不動——寧可少補一個詞，也不能塞一個
    下游查不到影片的 ID 進去（可播放率是新格式的硬指標）。
    """
    new_g, changed = apply(chinese, glosses, max_signs=max_signs)
    if not changed:
        return list(sign_ids), list(glosses), False
    sid = gloss_to_id.get(DUAL_GLOSS)
    if not sid:
        return list(sign_ids), list(glosses), False
    pos = new_g.index(DUAL_GLOSS)      # apply() 保證此時全串列只有一個
    return list(sign_ids[:pos]) + [sid] + list(sign_ids[pos:]), new_g, True
