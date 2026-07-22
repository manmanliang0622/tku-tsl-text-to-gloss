#!/usr/bin/env python3
"""Stage B 評估：微調後 Gemma 4 在 test set（33 句真實句）上的表現。

與 Stage A 用相同的 33 句、相同指標（BLEU-4/ROUGE-L/ExactMatch/詞彙表內率），
可直接比較（計畫 6.4）。輸出 results/finetuned_<tag>_test.jsonl 與 summary。

用法（VM venv 內）：
  python scripts/eval_model.py --adapter outputs/qlora_e4b
  python scripts/eval_model.py --adapter outputs/qlora_e4b --base google/gemma-4-E4B-it
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

import metrics
import prompt_common as pc

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"


def load_vocab():
    own = set(json.load((BASE / "data" / "tsl_gloss_vocab.json").open(encoding="utf-8"))["glosses"])
    union = set(own)
    twtsl = BASE / "data" / "twtsl" / "twtsl_words.jsonl"
    if twtsl.exists():
        for l in twtsl.read_text(encoding="utf-8").splitlines():
            e = json.loads(l)
            union.add(e.get("name") or e["chinese"])
            union.update(e.get("aliases", []))
            union.update(e.get("gloss", []))
    return own, union


def load_model(base, adapter, four_bit):
    # 與 train_qlora 相同的 PLE CPU-offload 載入法（避免 E4B 的 5.6GB PLE 表 OOM）
    from train_qlora import load_model as load_base
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True)
    model = load_base(base, bnb)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/gemma-4-E4B-it")
    ap.add_argument("--adapter", default=None, help="LoRA adapter 目錄；不給則評估未微調基礎模型")
    ap.add_argument("--four-bit", action="store_true", default=True)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    tag = args.tag or ("finetuned_e4b" if args.adapter else "base_e4b_hf")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = load_model(args.base, args.adapter, args.four_bit)
    own, union = load_vocab()

    test = [json.loads(l) for l in (BASE / "data" / "splits" / "test.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]

    out_path = RESULTS / f"{tag}_test.jsonl"
    recs = []
    with out_path.open("w", encoding="utf-8") as f:
        for i, item in enumerate(test):
            msgs = pc.build_messages(item["chinese"])
            inputs = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt",
                return_dict=True).to(model.device)
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new,
                                     do_sample=False)
            gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True)
            pred = pc.parse_gloss(gen)
            rec = {"id": item["id"], "chinese": item["chinese"],
                   "ref": item["gloss_text"], "pred": pred,
                   "raw": gen.strip(), "seconds": round(time.time() - t0, 1)}
            recs.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i+1}/{len(test)}] {item['id']} {item['chinese']} → {pred}",
                  flush=True)

    refs = [r["ref"] for r in recs]
    hyps = [r["pred"] for r in recs]
    m = metrics.evaluate(refs, hyps, own)
    m["InVocab%(自有85)"] = m.pop("InVocab%")
    m["InVocab%(聯集)"] = metrics.evaluate(refs, hyps, union)["InVocab%"]
    summary_path = RESULTS / f"summary_{tag}.json"
    summary_path.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print("==", tag, "==", json.dumps(m, ensure_ascii=False))


if __name__ == "__main__":
    main()
