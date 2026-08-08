"""訓練與評估共用的 prompt 格式與 Gloss 解析。

與 Stage A（run_baseline.py）的 TASK_DESC／RULES 保持一致，
確保 Stage A 基線與 Stage B 微調在相同提示條件下可直接比較（計畫 6.4）。
"""
import re

TASK_DESC = (
    "你是臺灣手語（TSL）翻譯助手。請把輸入的中文句子翻譯成臺灣手語 Gloss。"
    "Gloss 是手語動作的文字標記，以「/」分隔，例如：我/台北/住。"
    "只輸出一行 Gloss，不要輸出任何解釋或其他文字。"
)

# 7 條規則＝計畫 3.3 節（張榮興2008、Tai & Tsay 2015、Jane Tsay 2021、
# 教育部課綱、專案母語者例句歸納）
RULES = (
    "臺灣手語語法規則：\n"
    "1. 有情態詞「要」時，語序為 [時間]/[主語]/動詞/[地點或活動]/要（「要」放句尾）。\n"
    "2. 無情態詞、動詞是「住」「上班」等定居類動詞時，語序為 [主語]/地點/動詞（地點在動詞前）。\n"
    "3. 是非問句不翻出「嗎」，改以臉部表情（眉毛上揚）表達，Gloss 中不出現「嗎」。\n"
    "4. 判斷句不翻出「是」，直接 [主語]/[地點]/[身分]，例如「我是桃園人」→ 我/桃園/人。\n"
    "5. 時間詞（今天、明天等）一律放句首。\n"
    "6. 否定詞放動詞後或句尾，例如「我今天不去學校」→ 今天/我/學校/去/不。\n"
    "7. 疑問詞（什麼、哪裡、幾點等）放句末。"
)


# 微調用精簡 prompt：只給任務描述＋待翻譯句，不含 7 條規則。
# 理由：(1) 微調後模型從訓練資料學會語法，規則是 Stage A 未微調模型的拐杖；
# (2) 規則佔 ~260 token，會讓每筆序列變長、Gemma 262k 詞彙的 logits 記憶體暴增，
#     在共用 GPU 上留不住安全餘裕；(3) 訓練與評估同用此函式，仍保持一致可比。
INCLUDE_RULES = False


def build_user_prompt(chinese: str, examples=None, context: str = "") -> str:
    """訓練與推理共用的 user turn 內容。

    examples：[(中文, 目標), ...] 檢索到的相似例句，**放在同一個 user turn 內**。

    ⚠️ 2026-08-07 實測：把例句做成「多輪對話」注入會破壞格式遵循——v8 是以
    單輪（一個 user turn → JSON）微調的，多輪屬分布外輸入，短句會讓模型
    直接回聲輸入而不輸出 JSON（例：「謝謝」→ raw 就是「謝謝」）。
    故改為單輪內嵌，維持與訓練相同的對話結構。
    """
    head = TASK_DESC if not INCLUDE_RULES else f"{TASK_DESC}\n\n{RULES}"
    # 上下文（SCOPE 路線）：手語會沿用前文已建立的話題，故前文影響語序與省略。
    # 實測 22.3% 的參考 Gloss token 無法從本句中文推得，需前文才能還原。
    if context:
        head = f"{head}\n前文：{context}"
    if examples:
        ref = "\n".join(f"中文：{zh}\nGloss：{tgt}" for zh, tgt in examples)
        head = f"{head}\n\n以下是資料庫中的相似例句，可作參考：\n{ref}"
    return f"{head}\n中文：{chinese}\nGloss："


def build_messages(chinese: str, gloss_text: str = None, examples=None,
                   context: str = "") -> list:
    """conversational 格式；gloss_text=None 時只給 user turn（推理用）。

    examples：[(中文, 目標), ...]，推理時放入檢索到的相似例句（RAG），
    內嵌於同一個 user turn（見 build_user_prompt 的說明）。
    訓練不使用此參數，故不影響訓練/推理一致性。
    """
    msgs = [{"role": "user", "content": build_user_prompt(chinese, examples, context)}]
    if gloss_text is not None:
        msgs.append({"role": "assistant", "content": gloss_text})
    return msgs


def parse_gloss(raw: str) -> str:
    """從模型輸出取 Gloss：優先含「/」的行，否則最後一個非空行。"""
    def clean(line):
        line = line.strip().strip("`").strip()
        line = re.sub(r"^(Gloss|gloss|手語|TSL)[：:]\s*", "", line)
        line = line.replace("／", "/").replace(" ", "").strip("「」\"'")
        line = re.sub(r"[（(][^（）()]*[）)]$", "", line)
        return line.rstrip("。．.!?！？")

    lines = [clean(l) for l in raw.strip().splitlines()]
    lines = [l for l in lines if l]
    if not lines:
        return ""
    for line in lines:
        if "/" in line:
            return line
    return lines[-1]
