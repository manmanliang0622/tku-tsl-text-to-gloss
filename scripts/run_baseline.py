#!/usr/bin/env python3
"""Stage A：未微調 Gemma 4 的提示法基線（計畫第 5 節 Stage A）。

三種提示策略（設計依據）：
  zero    0-shot 任務描述          — SignAlignLM 的 0-shot prompt（ACL 2025 Findings）
  rules   ＋臺灣手語語法規則        — SignAlignLM 的 rule-based prompt；規則內容為
                                     計畫 3.3 節之 7 條已查證規則
  fewshot ＋規則＋專家示例（ICL）   — CCL24-Eval Task 10 的三段式提示詞
                                     （任務描述＋人類專家示例＋翻譯任務）

評測集：預設 data/tsl_sentences.jsonl 全部 35 句（核心集）；
`--split test_corpus|test_papers` 可改跑 data/splits/ 的留存測試集。

⚠️ **核心 35 句僅供 Stage A/B 歷史對照，對外報告不得引用**（見 NEXT_STEPS.md
「宣稱界線」）。要回答「微調到底有沒有幫助」，必須跑 `--split test_corpus`
與 `--split test_papers`——那才是與訓練集零重疊的誠實測試集。

few-shot 示例採 leave-one-out：示例池排除當前測試句，避免答案洩漏。
示例池來源（`--exemplar-source`）：
  core   固定 10 句核心集示例（預設，僅適用於跑核心集時）
  train  依固定亂數種子從 data/splits/train.jsonl 分層抽樣（跑留存測試集時的預設）

  為什麼跑留存測試集時必須換池：原本的 EXEMPLAR_IDS 全來自核心 35 句家族
  （問候語為主、句子短），拿去示範 test_corpus 的長對話句等於用另一個分布
  的示例，比較不公平。改為從 train 依 Gloss 長度分層抽樣（≤4／5–7／≥8 各佔
  三分之一），與 --length-balance 的分桶一致。

後端：Ollama，temperature=0。端點可用 --endpoint 或環境變數 TSL_OLLAMA_URL
覆寫（預設 http://localhost:11434/api/chat）——基線通常要在有 Ollama 的 VM
上跑，而不是開發機。

⚠️ **推論堆疊不同，報告必須註明**：本腳本走 Ollama＋`think:False`（見
`call_ollama` 註解），微調模型走 transformers GPU greedy（`eval_json_model.py`）。
同一個底模跑兩套堆疊，基線若落後會有「是不是輸在推論設定」的疑問。

結果逐句即時寫入 results/，中斷可 --resume 續跑。

用法：
  python3 scripts/run_baseline.py                          # 核心 35 句 × 3 策略
  python3 scripts/run_baseline.py --split test_corpus      # 留存語料庫長句 167 句
  python3 scripts/run_baseline.py --split test_papers      # 論文例句 143 句
  python3 scripts/run_baseline.py --limit 3                # 冒煙測試
  python3 scripts/run_baseline.py --strategies zero        # 只跑某策略
"""
import argparse
import json
import os
import random
import re
import time
import urllib.request
from pathlib import Path

import metrics

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
SPLITS = BASE / "data" / "splits"
OLLAMA_DEFAULT = "http://localhost:11434/api/chat"
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


def load_split(name):
    path = SPLITS / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(
            f"找不到 {path.relative_to(BASE)}。切分不入版控，請先重生：\n"
            "  python3 scripts/split_data.py --use-all --length-balance "
            "--papers-as-test --corpus-test-ratio 0.12 --corpus-test-min-len 6")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


# few-shot 示例池要排除的 NMS／註記寫法（見 build_train_pool 說明三）
NMS_IN_GLOSS = re.compile(r"表情|搖頭|揚眉|點頭|[（(]")


def n_gloss(e):
    return len([t for t in e["gloss_text"].split("/") if t])


def build_train_pool(size, seed):
    """從 train 依 Gloss 長度分層抽樣，桶界與 --length-balance 一致（≤4／5–7／≥8）。

    三件必須做的事，否則示例池會有系統性偏差：

    1. **先去重**：train 因長度平衡過取樣有 3,645 列是複製（8,992 列 → 5,347
       個相異句對），不去重的話長句被抽中的機率是三倍。
    2. **分層**：train 短句佔多數，隨機抽會抽出一池短句，拿去示範 test_corpus
       的長對話句等於暗示模型「輸出要短」——那正是 v10 之前的已知偏差。
    3. **濾掉退化示例**：語料庫含大量對話片段（「一條乾的，」→「乾/一」、
       「為什麼？」→「為什麼」）。它們是合法語料，但當示例會教出壞習慣，
       故要求至少 2 個 Gloss 且中文至少 5 字。
       同理排除含 NMS 標記者（`表情(點頭微笑)/…`）——這類只佔 train 相異句
       的 1.3%、test_corpus 1.2%、test_papers 0%，10 句示例裡出現 1 句就等於
       放大 8 倍，會誘導基線輸出參考答案幾乎不含的標記。
    """
    uniq = {}
    for e in load_split("train"):
        uniq.setdefault((e["chinese"], e["gloss_text"]), e)
    cand = [e for e in uniq.values()
            if n_gloss(e) >= 2
            and len(e["chinese"].strip("，。、？?！!")) >= 5
            and not NMS_IN_GLOSS.search(e["gloss_text"])]

    buckets = {"short": [], "mid": [], "long": []}
    for e in cand:
        n = n_gloss(e)
        buckets["short" if n <= 4 else "mid" if n <= 7 else "long"].append(e)

    rng = random.Random(seed)
    names = ("short", "mid", "long")
    # 餘數平均分給前幾桶，不用隨機補齊（補齊會破壞分層）
    quota = {n: size // 3 + (1 if i < size % 3 else 0) for i, n in enumerate(names)}
    pool = []
    for name in names:
        rows = sorted(buckets[name], key=lambda e: e["id"])   # 排序後再抽，確保可重現
        pool.extend(rng.sample(rows, min(quota[name], len(rows))))
    return sorted(pool, key=lambda e: n_gloss(e))


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


def call_ollama(model, prompt, endpoint, timeout=600):
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
    req = urllib.request.Request(endpoint, data=body,
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
    ap.add_argument("--split", default=None,
                    help="改跑 data/splits/ 的留存測試集，如 test_corpus、test_papers。"
                         "省略則跑核心 35 句（僅供歷史對照，對外報告不得引用）")
    ap.add_argument("--exemplar-source", choices=["core", "train"], default=None,
                    help="few-shot 示例池來源。預設：跑核心集用 core、跑留存測試集用 train")
    ap.add_argument("--pool-size", type=int, default=10, help="few-shot 示例句數")
    ap.add_argument("--pool-seed", type=int, default=42, help="train 示例池抽樣種子")
    ap.add_argument("--endpoint", default=os.environ.get("TSL_OLLAMA_URL", OLLAMA_DEFAULT),
                    help="Ollama /api/chat 端點，亦可用環境變數 TSL_OLLAMA_URL")
    ap.add_argument("--dry-run", action="store_true",
                    help="不呼叫模型，只印出設定與第一句的完整提示詞。"
                         "用於在沒有 Ollama 的機器上檢查示例池與提示詞是否正確")
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    sents = load_split(args.split) if args.split else load_sentences()
    if args.limit:
        sents = sents[:args.limit]

    src = args.exemplar_source or ("train" if args.split else "core")
    if src == "core":
        pool = [s for s in load_sentences() if s["id"] in EXEMPLAR_IDS]
    else:
        pool = build_train_pool(args.pool_size, args.pool_seed)
    # 留存測試集與 train 已驗證零重疊，仍保留 leave-one-out 作為便宜的防呆
    overlap = {e["id"] for e in pool} & {e["id"] for e in sents}

    vocab = set(json.load((BASE / "data" / "tsl_gloss_vocab.json").open(encoding="utf-8"))["glosses"])

    suffix = f"_{args.split}" if args.split else ""
    print(f"評測集：{args.split or '核心 35 句'}（{len(sents)} 句）")
    print(f"示例池：{src}，{len(pool)} 句"
          + (f"（種子 {args.pool_seed}）" if src == "train" else "")
          + (f"  ⚠ 與測試集重疊 {len(overlap)} 句，將由 leave-one-out 排除" if overlap else ""))
    print(f"端點　：{args.endpoint}")
    if not args.split:
        print("⚠ 核心 35 句僅供 Stage A/B 歷史對照，對外報告不得引用；"
              "要比較微調成效請加 --split test_corpus 或 --split test_papers")

    if args.dry_run:
        for e in pool:
            print(f"  示例 {e['id']:<10}{n_gloss(e):>2} 詞  {e['chinese']} → {e['gloss_text']}")
        for strat in args.strategies:
            print(f"\n{'='*70}\n{strat} 的提示詞（第一句）\n{'='*70}")
            print(build_prompt(strat, sents[0], pool))
        return

    tag = args.model.replace(":", "_")
    for strat in args.strategies:
        out_path = RESULTS / f"baseline_{tag}{suffix}_{strat}.jsonl"
        done = set()
        # 空檔視同不存在：連線失敗時會留下 0 byte 檔（以 "a" 開檔的副作用），
        # 否則下一次重跑會被自己的閘門擋住
        if out_path.exists() and out_path.stat().st_size == 0:
            out_path.unlink()
        if out_path.exists():
            if not args.resume:
                # 舊版沒有這個閘門：結果是以 "a" 模式開檔，重跑會把新結果附加在
                # 既有結果後面，同一個 id 出現兩次，指標靜靜地算錯。
                raise SystemExit(
                    f"{out_path.relative_to(BASE)} 已存在。加 --resume 續跑，"
                    "或先刪除／改名該檔再重跑（本腳本以附加模式寫檔，"
                    "直接重跑會產生重複列）。")
            done = {json.loads(l)["id"] for l in out_path.read_text(encoding="utf-8").splitlines()}
        with out_path.open("a", encoding="utf-8") as f:
            for i, item in enumerate(sents):
                if item["id"] in done:
                    continue
                prompt = build_prompt(strat, item, pool)
                t0 = time.time()
                raw = call_ollama(args.model, prompt, args.endpoint)
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
        summary_path = RESULTS / f"summary_{tag}{suffix}.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        # 連同設定一起存：推論堆疊與示例池會影響結果，報告要能追溯
        summary[strat] = dict(m, _config={
            "split": args.split or "core35", "model": args.model,
            "backend": "ollama", "think": False, "temperature": 0,
            "exemplar_source": src,
            "pool_ids": [e["id"] for e in pool] if strat == "fewshot" else None,
            "pool_seed": args.pool_seed if src == "train" else None,
        })
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
