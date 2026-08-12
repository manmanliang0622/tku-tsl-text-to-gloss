#!/usr/bin/env python3
"""評估「JSON 目標」模型（--target json 訓出者）並與純 Gloss 模型公平對照。

公平性關鍵：JSON 模型輸出的是一整包 JSON，需先取出其中的 `gloss` 欄位、
把空白分隔換回「/」，才能與純 Gloss 模型用同一套 metrics 比較。
除 Gloss 準確度外，另量它獨有的能力：疑問句判斷、否定判斷、NMS 是否輸出。

**分維度評估（2026-08-13 補上）**：教授要求語意／選詞／語序／漏詞／亂加詞／
未知詞／NMS 分開看，不能只有 EM／BLEU。原本主評估只有整批層級的
BLEU／ROUGE-L／EM／內率，逐句的 GER 與錯誤分類只存在於 `eval_three_tier.py`，
一般評估看不到。現改為一律接上 `eval_metrics_ext`，並輸出逐句 CSV：

  語序 → 語序錯誤        選詞 → Gloss替換錯誤     漏詞 → 漏Gloss
  亂加詞 → 多餘Gloss     未知詞 → OOV/未知Gloss   NMS → Question/Negation/Nonmanual
  語意 → **自動指標答不了，需母語者 5 分制人工評分（計畫 6.2）**

⚠️ OOV 判定基準用 **詞彙總表 ∪ 訓練詞彙**（`gloss_fallback.load_vocab`），
不是訓練詞彙。用訓練詞彙當基準會把「合法但訓練沒出現的手語詞」（畫家、
幼稚園、目不轉睛）誤判成模型造詞——實測高估 67%（45 個 vs 實際 27 個）。
註：`eval_three_tier.py` 仍用訓練詞彙為基準，其錯誤分類帶此已知偏差。

用法（VM）：
  python3 scripts/eval_json_model.py --adapter outputs/qlora_e4b_v8_json/checkpoint-763 \
      --tag v8_json_ep1
"""
import argparse
import csv
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

import eval_metrics_ext as ext
import metrics
import prompt_common as pc

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"


def parse_json_output(raw):
    """從模型輸出取出 JSON；失敗時回傳 None（並記錄為無效輸出）。"""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/gemma-4-E4B-it")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--test-file", default="test.jsonl",
                    help="data/splits_json/ 下的測試檔：test.jsonl（核心33句）／"
                         "test_corpus.jsonl（語料庫長句留存）／test_papers.jsonl（論文例句）")
    ap.add_argument("--max-new", type=int, default=160)   # JSON 目標較長
    ap.add_argument("--ple", choices=["auto", "gpu", "cpu"], default="auto",
                    help="PLE 放置；gpu=全模型放 GPU（快約 300 倍）。"
                         "auto 在可用顯存剛好卡門檻時會誤退回慢速模式，評估建議明指 gpu")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    from train_qlora import load_model as load_base, can_fit_ple_on_gpu
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    ple_on_gpu = {"auto": None, "gpu": True, "cpu": False}[args.ple]
    if ple_on_gpu is None:
        ple_on_gpu = can_fit_ple_on_gpu()
    print(f"[eval] PLE 放置：{'GPU（快）' if ple_on_gpu else 'CPU（慢，約 35 秒/token）'}", flush=True)
    model = load_base(args.base, bnb, ple_on_gpu=ple_on_gpu)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # 參考答案取自 JSON 版 test（與訓練同格式），可同時比 gloss 與語法欄位
    test = [json.loads(l) for l in (BASE / "data/splits_json" / args.test_file)
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[eval] 測試集 {args.test_file}：{len(test)} 句", flush=True)

    recs, invalid = [], 0
    for i, item in enumerate(test):
        ref_obj = json.loads(item["output"])
        # 上下文翻譯模型（--context 訓練者）必須在推論時也帶入前文，
        # 否則訓練/推論格式不一致，會嚴重低估其表現。
        msgs = pc.build_messages(item["input"], context=item.get("context", ""))
        inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                         return_tensors="pt", return_dict=True).to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=False)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        obj = parse_json_output(gen)
        if obj is None:
            invalid += 1
            obj = {}
        pred_gloss = "/".join(str(obj.get("gloss", "")).split())
        recs.append({
            "id": f"T{i:03d}", "chinese": item["input"],
            "ref": "/".join(ref_obj["gloss"].split()), "pred": pred_gloss,
            "ref_question": ref_obj.get("question_type"), "pred_question": obj.get("question_type"),
            "ref_negation": ref_obj.get("negation"), "pred_negation": obj.get("negation"),
            "ref_nonmanual": ref_obj.get("nonmanual"), "pred_nonmanual": obj.get("nonmanual"),
            "has_context": bool(item.get("context")),
            "valid_json": obj != {}, "raw": gen.strip(), "seconds": round(time.time() - t0, 2),
        })
        print(f"[{i+1}/{len(test)}] {item['input']} → {pred_gloss}", flush=True)

    out_path = RESULTS / f"{args.tag}_test.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    vocab = json.load((BASE / "data/vocab/eval_vocab.json").open(encoding="utf-8"))
    m = metrics.evaluate([r["ref"] for r in recs], [r["pred"] for r in recs],
                         set(vocab["renderable"]))
    n = len(recs)

    # --- 分維度評估：逐句 GER 與錯誤分類（見檔頭說明）---
    from gloss_fallback import load_vocab
    legal_vocab, _ = load_vocab()          # 合法詞＝詞彙總表 ∪ 訓練詞彙
    per_sent = [ext.score_pair(r["ref"], r["pred"], legal_vocab) for r in recs]
    agg = ext.aggregate(per_sent)
    # 兩套 EM 應一致（同為 token 完全相符）；不一致代表切詞規則漂移，要查
    if agg["ExactMatch%"] != m["ExactMatch%"]:
        m["ExactMatch%_ext"] = agg["ExactMatch%"]
        print(f"⚠ EM 不一致：metrics {m['ExactMatch%']} vs ext {agg['ExactMatch%']}，"
              "請檢查兩邊的 tokenize 規則", flush=True)
    for k in ("TokenPrecision", "TokenRecall", "TokenF1", "GER",
              "AvgEditDistance", "ErrorTypes", "ErrorTypes%"):
        m[k] = agg[k]
    m["OOVBasis"] = f"gloss_master ∪ train（{len(legal_vocab)} 詞）"

    csv_path = RESULTS / f"{args.tag}_per_sentence.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "chinese", "ref", "pred"]
                           + list(per_sent[0].keys()))
        w.writeheader()
        for r, s in zip(recs, per_sent):
            w.writerow({"id": r["id"], "chinese": r["chinese"],
                        "ref": r["ref"], "pred": r["pred"], **s})
    print(f"逐句結果 → {csv_path.relative_to(BASE)}", flush=True)
    m["ValidJSON%"] = round(sum(r["valid_json"] for r in recs) / n * 100, 2)
    m["QuestionAcc%"] = round(sum(r["ref_question"] == r["pred_question"] for r in recs) / n * 100, 2)
    m["NegationAcc%"] = round(sum(r["ref_negation"] == r["pred_negation"] for r in recs) / n * 100, 2)
    m["NonmanualNonNone%"] = round(
        sum(1 for r in recs if r["pred_nonmanual"] not in (None, "none")) / n * 100, 2)
    m["WithContext%"] = round(sum(r["has_context"] for r in recs) / n * 100, 2)
    (RESULTS / f"summary_{args.tag}.json").write_text(
        json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n== {args.tag} ==")
    print(json.dumps(m, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
