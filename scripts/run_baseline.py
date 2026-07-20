#!/usr/bin/env python3
"""Stage A：未微調 Gemma 4 的提示法基線（計畫第 5 節 Stage A）。

三種提示策略（設計依據）：
  zero    0-shot 任務描述          — SignAlignLM 的 0-shot prompt（ACL 2025 Findings）
  rules   ＋臺灣手語語法規則        — SignAlignLM 的 rule-based prompt；規則內容為
                                     計畫 3.3 節之 7 條已查證規則
  fewshot ＋規則＋專家示例（ICL）   — CCL24-Eval Task 10 的三段式提示詞
                                     （任務描述＋人類專家示例＋翻譯任務）

評測集：data/tsl_sentences.jsonl 全部 35 句（目前唯一經人工確認的資料）。
few-shot 示例採 leave-one-out：示例池排除當前測試句，避免答案洩漏。

後端：本機 Ollama（http://localhost:11434），temperature=0。
結果逐句即時寫入 results/，中斷可 --resume 續跑。

用法：
  python3 scripts/run_baseline.py                    # 全部 3 策略 × 35 句
  python3 scripts/run_baseline.py --limit 3          # 冒煙測試
  python3 scripts/run_baseline.py --strategies zero  # 只跑某策略
"""
import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

import metrics

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
OLLAMA = "http://localhost:11434/api/chat"
MODEL_DEFAULT = "gemma4:e4b"

TASK_DESC = (
    "你是臺灣手語（TSL）翻譯助手。請把輸入的中文句子翻譯成臺灣手語 Gloss。"
    "Gloss 是手語動作的文字標記，以「/」分隔，例如：我/台北/住。"
    "只輸出一行 Gloss，不要輸出任何解釋或其他文字。"
)

# 7 條規則＝計畫 3.3 節（張榮興2008、Tai & Tsay 2015、Jane Tsay 2021、
# 教育部課綱、專案母語者例句歸納），與合成模板共用同一套規則
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

# few-shot 示例池：涵蓋定居句/是非問/情態要/身分句/否定/WH/程度詞等句型
EXEMPLAR_IDS = ["P01", "P03", "P04", "P05", "S01", "S09", "S14", "S21", "S23", "S28"]


def load_sentences():
    sents = []
    for line in (BASE / "data" / "tsl_sentences.jsonl").read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if not e["is_template"]:  # S24/S26 含佔位符，不宜直接評測
            sents.append(e)
    return sents


def build_prompt(strategy, item, pool):
    parts = [TASK_DESC]
    if strategy in ("rules", "fewshot"):
        parts.append(RULES)
    if strategy == "fewshot":
        ex_lines = ["以下是人類專家編寫的翻譯示例："]
        for ex in pool:
            if ex["id"] == item["id"]:  # leave-one-out
                continue
            ex_lines.append(f"中文：{ex['chinese']}\nGloss：{ex['gloss_text']}")
        parts.append("\n".join(ex_lines))
    parts.append(f"中文：{item['chinese']}\nGloss：")
    return "\n\n".join(parts)


def call_ollama(model, prompt, timeout=600):
    # think=False：Gemma 4 為思考型模型，思考內容會佔滿 num_predict 導致正式輸出
    # 為空（done_reason=length）。本機 CPU 跑不起完整思考鏈（數百 token/句），
    # 故基線統一關閉思考模式，此設定需在報告中註明。
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 128},
    }).encode()
    req = urllib.request.Request(OLLAMA, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["message"]["content"]


def parse_gloss(raw: str) -> str:
    """優先取含「/」的行（最像 Gloss），否則取最後一個非空行（單詞句無分隔符）。

    正規化：全形／→半形/、去空白、去前綴（Gloss：）、去引號與句尾標點、
    去尾端括號註解（如「（規則2）」）。
    """
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--strategies", nargs="+",
                    default=["zero", "rules", "fewshot"],
                    choices=["zero", "rules", "fewshot"])
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 句（冒煙測試）")
    ap.add_argument("--resume", action="store_true", help="略過已有結果的句子")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    sents = load_sentences()
    if args.limit:
        sents = sents[:args.limit]
    pool = [s for s in load_sentences() if s["id"] in EXEMPLAR_IDS]
    vocab = set(json.load((BASE / "data" / "tsl_gloss_vocab.json").open(encoding="utf-8"))["glosses"])

    tag = args.model.replace(":", "_")
    for strat in args.strategies:
        out_path = RESULTS / f"baseline_{tag}_{strat}.jsonl"
        done = set()
        if args.resume and out_path.exists():
            done = {json.loads(l)["id"] for l in out_path.read_text(encoding="utf-8").splitlines()}
        with out_path.open("a", encoding="utf-8") as f:
            for i, item in enumerate(sents):
                if item["id"] in done:
                    continue
                prompt = build_prompt(strat, item, pool)
                t0 = time.time()
                raw = call_ollama(args.model, prompt)
                pred = parse_gloss(raw)
                rec = {"id": item["id"], "chinese": item["chinese"],
                       "ref": item["gloss_text"], "pred": pred,
                       "raw": raw.strip(), "seconds": round(time.time() - t0, 1)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{strat} {i+1}/{len(sents)}] {item['id']} "
                      f"{item['chinese']} → {pred}  ({rec['seconds']}s)", flush=True)

        # 策略跑完立即算指標
        recs = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines()]
        m = metrics.evaluate([r["ref"] for r in recs], [r["pred"] for r in recs], vocab)
        print(f"== {strat} == {m}", flush=True)
        summary_path = RESULTS / f"summary_{tag}.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        summary[strat] = m
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
