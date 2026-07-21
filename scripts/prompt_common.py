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


def build_user_prompt(chinese: str) -> str:
    """訓練與推理共用的 user turn 內容（TASK_DESC + RULES + 待翻譯句）。"""
    return f"{TASK_DESC}\n\n{RULES}\n\n中文：{chinese}\nGloss："


def build_messages(chinese: str, gloss_text: str = None) -> list:
    """conversational 格式；gloss_text=None 時只給 user turn（推理用）。"""
    msgs = [{"role": "user", "content": build_user_prompt(chinese)}]
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
