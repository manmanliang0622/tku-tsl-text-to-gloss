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

後端（`--backend`）：

  transformers  **預設**。直接載入未微調的 base model，走與 `eval_json_model.py`
                完全相同的推論堆疊：同一個 4-bit bnb 量化設定、同一個
                `load_model(..., ple_on_gpu)`、同一個 chat template、
                `do_sample=False` greedy 無 beam。差別只有「有沒有掛 adapter」。
  ollama        舊路徑，保留供重現既有的核心 33 句結果。走 Ollama＋`think:False`
                （見 `call_ollama` 註解），與微調模型不是同一套堆疊。

為什麼預設改成 transformers：基線的用途是回答「微調到底有沒有幫助」。若基線
走 Ollama CPU、微調走 transformers GPU，兩者差的就不只是 adapter，基線落後時
無法排除「輸在推論設定」。改用同一套堆疊後，唯一的變因就是 adapter。

⚠️ `--ple gpu` 很重要：`device_map` 只要含任何 "cpu" 項目，accelerate 就會為
整個模型掛 offload hook，每個 token 慢到約 35 秒（全放 GPU 是 0.06 秒/token，
快約 580 倍）。詳見 `train_qlora.load_model` 的說明。

提示詞與 Gloss 解析一律從 `prompt_common` 匯入，與微調端共用同一份定義，
避免兩邊各改一份而悄悄失去可比性。

結果逐句即時寫入 results/，中斷可 --resume 續跑。

用法：
  python3 scripts/run_baseline.py --split test_corpus --ple gpu   # 留存語料庫長句 167 句
  python3 scripts/run_baseline.py --split test_papers --ple gpu   # 論文例句 143 句
  python3 scripts/run_baseline.py --split test_corpus --dry-run   # 不載模型，只看提示詞
  python3 scripts/run_baseline.py --backend ollama                # 舊路徑（核心 35 句）
  python3 scripts/run_baseline.py --limit 3                       # 冒煙測試
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
# 提示詞與 Gloss 解析與微調端共用同一份定義（原本兩邊各有一份逐字相同的副本，
# 任一邊改動都會悄悄破壞 Stage A 與 Stage B 的可比性）
from prompt_common import RULES, TASK_DESC, parse_gloss

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
SPLITS = BASE / "data" / "splits"
OLLAMA_DEFAULT = "http://localhost:11434/api/chat"
OLLAMA_MODEL_DEFAULT = "gemma4:e4b"
HF_MODEL_DEFAULT = "google/gemma-4-E4B-it"

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


def build_hf_runner(model_id, ple, max_new):
    """載入未微調的 base model，回傳 infer(prompt) -> raw。

    刻意與 `eval_json_model.py` 走同一條路徑：同一個 bnb 4-bit 設定、同一個
    `train_qlora.load_model`、同一個 chat template、`do_sample=False`。
    唯一差別是**不掛 adapter**——這樣「基線 vs 微調」的變因才只有 adapter 一項。
    """
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig

    from train_qlora import can_fit_ple_on_gpu, load_model

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    ple_on_gpu = {"auto": None, "gpu": True, "cpu": False}[ple]
    if ple_on_gpu is None:
        ple_on_gpu = can_fit_ple_on_gpu()
    print(f"[baseline] PLE 放置：{'GPU（快）' if ple_on_gpu else 'CPU（慢，約 35 秒/token）'}",
          flush=True)
    model = load_model(model_id, bnb, ple_on_gpu=ple_on_gpu)
    model.eval()

    def infer(prompt):
        msgs = [{"role": "user", "content": prompt}]
        inputs = tok.apply_chat_template(
            msgs, add_generation_prompt=True,
            return_tensors="pt", return_dict=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    return infer


def load_vocab():
    """與 `eval_json_model.py` 用同一份詞彙表，InVocab% 才可以並排比較。

    eval_vocab.json 不入版控（可由 build_eval_vocab.py 再生，依賴 splits）；
    找不到時退回 tsl_gloss_vocab.json，並回報實際用了哪一份——兩份的內率
    不可互相比較，報告必須標明。
    """
    p = BASE / "data" / "vocab" / "eval_vocab.json"
    if p.exists():
        return set(json.load(p.open(encoding="utf-8"))["renderable"]), "eval_vocab.renderable"
    p = BASE / "data" / "tsl_gloss_vocab.json"
    return set(json.load(p.open(encoding="utf-8"))["glosses"]), "tsl_gloss_vocab.glosses"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["transformers", "ollama"], default="transformers",
                    help="transformers＝與 eval_json_model 同一套推論堆疊（預設）；"
                         "ollama＝舊路徑，僅供重現既有核心 33 句結果")
    ap.add_argument("--model", default=None,
                    help=f"預設隨 backend：transformers→{HF_MODEL_DEFAULT}、"
                         f"ollama→{OLLAMA_MODEL_DEFAULT}")
    ap.add_argument("--ple", choices=["auto", "gpu", "cpu"], default="gpu",
                    help="僅 transformers。gpu＝全模型放 GPU（快約 580 倍）。"
                         "auto 在顯存剛好卡門檻時會誤退回慢速模式，故預設 gpu")
    ap.add_argument("--max-new", type=int, default=128,
                    help="僅 transformers。與 ollama 路徑的 num_predict=128 對齊，"
                         "兩個後端的輸出長度預算才一致。"
                         "不可再調低：test_corpus 最長參考答案 20 個 Gloss／56 字元，"
                         "截斷會讓基線莫名失分——那正是本次要消除的不公平")
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
    if args.model is None:
        args.model = HF_MODEL_DEFAULT if args.backend == "transformers" else OLLAMA_MODEL_DEFAULT

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

    vocab, vocab_name = load_vocab()

    suffix = f"_{args.split}" if args.split else ""
    print(f"評測集：{args.split or '核心 35 句'}（{len(sents)} 句）")
    print(f"示例池：{src}，{len(pool)} 句"
          + (f"（種子 {args.pool_seed}）" if src == "train" else "")
          + (f"  ⚠ 與測試集重疊 {len(overlap)} 句，將由 leave-one-out 排除" if overlap else ""))
    print(f"詞彙表：{vocab_name}（{len(vocab)} 詞）")
    if args.backend == "transformers":
        print(f"後端　：transformers {args.model}（未微調 base model，"
              f"ple={args.ple}、greedy、max_new={args.max_new}）")
    else:
        print(f"後端　：ollama {args.model} @ {args.endpoint}（think:False）")
        print("⚠ ollama 與微調模型不是同一套推論堆疊，基線落後時無法排除"
              "「輸在推論設定」；報告須註明，或改用 --backend transformers")
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

    if args.backend == "transformers":
        infer = build_hf_runner(args.model, args.ple, args.max_new)
    else:
        def infer(prompt):
            return call_ollama(args.model, prompt, args.endpoint)

    tag = re.sub(r"[^A-Za-z0-9._-]", "_", args.model.replace(":", "_"))
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
                raw = infer(prompt)
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
        cfg = {
            "split": args.split or "core35", "model": args.model,
            "backend": args.backend, "vocab": vocab_name,
            "exemplar_source": src,
            "pool_ids": [e["id"] for e in pool] if strat == "fewshot" else None,
            "pool_seed": args.pool_seed if src == "train" else None,
        }
        if args.backend == "transformers":
            cfg.update({"ple": args.ple, "do_sample": False, "num_beams": 1,
                        "max_new_tokens": args.max_new, "adapter": None})
        else:
            cfg.update({"think": False, "temperature": 0, "num_predict": 128})
        summary[strat] = dict(m, _config=cfg)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
