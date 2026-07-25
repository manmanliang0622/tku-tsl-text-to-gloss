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
    ap.add_argument("--test-file", default="test.jsonl",
                    help="data/splits/ 下的 test 檔名（預設 test.jsonl 核心 33 句；"
                         "擴大真實 test 集用 test_corpus.jsonl）")
    ap.add_argument("--resume", action="store_true",
                    help="沿用已存在的結果，略過已評估過的 id（長 test 集中斷可續跑）")
    ap.add_argument("--bootstrap-samples", type=int, default=1000,
                    help="BLEU 95%% CI 的 group bootstrap 次數（0=不計；預設1000）")
    ap.add_argument("--bootstrap-seed", type=int, default=42)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    tag = args.tag or ("finetuned_e4b" if args.adapter else "base_e4b_hf")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = load_model(args.base, args.adapter, args.four_bit)
    own, union = load_vocab()

    test = [json.loads(l) for l in (BASE / "data" / "splits" / args.test_file)
            .read_text(encoding="utf-8").splitlines() if l.strip()]

    out_path = RESULTS / f"{tag}_test.jsonl"
    recs = []
    done = set()
    if args.resume and out_path.exists():
        for l in out_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                recs.append(r)
                done.add(r["id"])
        print(f"resume：已有 {len(done)} 筆，續跑剩餘")
    with out_path.open("a" if args.resume else "w", encoding="utf-8") as f:
        for i, item in enumerate(test):
            if item["id"] in done:
                continue
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
                   "raw": gen.strip(), "seconds": round(time.time() - t0, 1),
                   "group": item.get("group") or f"row:{item['id']}"}
            recs.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i+1}/{len(test)}] {item['id']} {item['chinese']} → {pred}",
                  flush=True)

    test_by_id = {item["id"]: item for item in test}
    if len(test_by_id) != len(test):
        raise ValueError("test file 含重複 ID")
    rec_by_id = {}
    for rec in recs:
        if rec["id"] in rec_by_id:
            raise ValueError(f"結果含重複 ID：{rec['id']}")
        rec_by_id[rec["id"]] = rec
    unexpected = sorted(set(rec_by_id) - set(test_by_id))
    missing = sorted(set(test_by_id) - set(rec_by_id))
    if unexpected or missing:
        raise ValueError(
            f"結果 ID 與 test 不一致：unexpected={unexpected[:5]} missing={missing[:5]}")

    # 依 test 檔順序重排；舊版 resume 結果沒有 group 時由當前 test 補上。
    recs = [rec_by_id[item["id"]] for item in test]
    for rec in recs:
        rec.setdefault(
            "group", test_by_id[rec["id"]].get("group") or f"row:{rec['id']}")
    refs = [r["ref"] for r in recs]
    hyps = [r["pred"] for r in recs]
    groups = [r["group"] for r in recs]
    m = metrics.evaluate(
        refs, hyps, own, groups=groups,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed)
    m["InVocab%(自有85)"] = m.pop("InVocab%")
    m["InVocab%(聯集)"] = metrics.evaluate(refs, hyps, union)["InVocab%"]
    m["test_file"] = args.test_file
    m["tag"] = tag
    summary_path = RESULTS / f"summary_{tag}.json"
    summary_path.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print("==", tag, "==", json.dumps(m, ensure_ascii=False))


if __name__ == "__main__":
    main()
