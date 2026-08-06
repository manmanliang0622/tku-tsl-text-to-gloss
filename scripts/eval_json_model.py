#!/usr/bin/env python3
"""評估「JSON 目標」模型（--target json 訓出者）並與純 Gloss 模型公平對照。

公平性關鍵：JSON 模型輸出的是一整包 JSON，需先取出其中的 `gloss` 欄位、
把空白分隔換回「/」，才能與純 Gloss 模型用同一套 metrics 比較。
除 Gloss 準確度外，另量它獨有的能力：疑問句判斷、否定判斷、NMS 是否輸出。

用法（VM）：
  python3 scripts/eval_json_model.py --adapter outputs/qlora_e4b_v8_json/checkpoint-763 \
      --tag v8_json_ep1
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

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
    ap.add_argument("--max-new", type=int, default=160)   # JSON 目標較長
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    from train_qlora import load_model as load_base, can_fit_ple_on_gpu
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    model = load_base(args.base, bnb, ple_on_gpu=can_fit_ple_on_gpu())
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # 參考答案取自 JSON 版 test（與訓練同格式），可同時比 gloss 與語法欄位
    test = [json.loads(l) for l in (BASE / "data/splits_json/test.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]

    recs, invalid = [], 0
    for i, item in enumerate(test):
        ref_obj = json.loads(item["output"])
        msgs = pc.build_messages(item["input"])
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
    m["ValidJSON%"] = round(sum(r["valid_json"] for r in recs) / n * 100, 2)
    m["QuestionAcc%"] = round(sum(r["ref_question"] == r["pred_question"] for r in recs) / n * 100, 2)
    m["NegationAcc%"] = round(sum(r["ref_negation"] == r["pred_negation"] for r in recs) / n * 100, 2)
    m["NonmanualNonNone%"] = round(
        sum(1 for r in recs if r["pred_nonmanual"] not in (None, "none")) / n * 100, 2)
    (RESULTS / f"summary_{args.tag}.json").write_text(
        json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n== {args.tag} ==")
    print(json.dumps(m, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
