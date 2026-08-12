#!/usr/bin/env python3
"""三層診斷評估：分辨模型是「沒學會」還是「學會了但泛化不足」。

  A. Seen        訓練集原句（100 句，固定種子）
  B. Paraphrase  同一批句子、中文換句話說
  C. Unseen      完全沒見過、且用詞多半學過的句子

沿用專案既有元件，不另建模型載入或提示流程：
  train_qlora.load_model      模型載入（含 PLE 放置策略）
  prompt_common.build_messages 提示格式（與訓練同一份，確保訓練/推論一致）
  eval_json_model.parse_json_output  JSON 輸出解析
  metrics.corpus_bleu         BLEU-4
  eval_metrics_ext            逐句 P/R/F1、GER、編輯距離、錯誤分類

用法（VM，venv 內）：
  python3 scripts/eval_three_tier.py \
      --adapter outputs/qlora_e4b_v12_context/checkpoint-1124 --ple gpu

輸出：evaluation_results/{seen,paraphrase,unseen}_results.csv 與 summary.json
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

import metrics
import eval_metrics_ext as ext
import prompt_common as pc
from eval_json_model import parse_json_output

BASE = Path(__file__).resolve().parent.parent
TIERS = BASE / "data" / "eval_tiers"
OUT = BASE / "evaluation_results"

CSV_FIELDS = ["test_type", "idx", "chinese_input", "chinese_original",
              "expected_gloss", "predicted_gloss", "exact_match",
              "precision", "recall", "f1", "ger", "edit_distance",
              "error_type", "oov_tokens", "transform",
              "needs_semantic_mapping", "source", "seconds"]


def train_vocab():
    """訓練集出現過的 Gloss 詞——OOV 判定的依據。"""
    v = set()
    path = BASE / "data" / "splits_json" / "train.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            v.update(json.loads(json.loads(line)["output"])["gloss"].split())
    return v


def load_tier(name):
    p = TIERS / f"{name}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def run_tier(rows, model, tok, args, vocab):
    """對一層資料跑推論並逐句評分。"""
    out = []
    for i, r in enumerate(rows, 1):
        # 與訓練完全相同的提示組法（含上下文），避免訓練/推論不一致而低估模型
        msgs = pc.build_messages(r["chinese"], context=r.get("context", ""))
        inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                         return_tensors="pt", return_dict=True).to(model.device)
        t0 = time.time()
        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=False)
        raw = tok.decode(gen_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        obj = parse_json_output(raw) or {}
        pred = str(obj.get("gloss", "")).strip()

        s = ext.score_pair(r["expected_gloss"], pred, vocab)
        out.append({
            "test_type": r["test_type"], "idx": r.get("idx", i),
            "chinese_input": r["chinese"],
            "chinese_original": r.get("chinese_original", ""),
            "expected_gloss": r["expected_gloss"], "predicted_gloss": pred,
            "transform": r.get("transform", ""),
            "needs_semantic_mapping": r.get("needs_semantic_mapping", ""),
            "source": r.get("source", ""), "seconds": round(time.time() - t0, 2),
            **{k: s[k] for k in ("exact_match", "precision", "recall", "f1",
                                 "ger", "edit_distance", "error_type", "oov_tokens")},
            "_ref_len": s["ref_len"],
        })
        print(f"  [{i}/{len(rows)}] {r['chinese'][:24]} → {pred[:36]}", flush=True)
    return out


def summarize(rows):
    """整層統計；BLEU 沿用專案既有的 metrics.corpus_bleu 以與其他報告可比。"""
    if not rows:
        return {}
    agg = ext.aggregate([{**r, "ref_len": r["_ref_len"]} for r in rows])
    refs = [metrics.tokenize(r["expected_gloss"].replace(" ", "/")) for r in rows]
    hyps = [metrics.tokenize(r["predicted_gloss"].replace(" ", "/")) for r in rows]
    agg["BLEU-4"] = round(metrics.corpus_bleu(refs, hyps), 2)
    return agg


def degraded(base, other, drop):
    """某層相對 Seen 層是否明顯退化。

    ⚠️ 必須同時看 TokenF1 與 GER，不能只看 F1。
    token 層 P/R/F1 是**多重集合比對，對語序完全無感**——把答案整個顛倒過來，
    F1 仍是 1.000。實測（本檔煙霧測試）：完全亂序的預測 F1=1.000 但 GER=0.901。
    只看 F1 會把「語序全崩」誤判為沒有退化，而語序正是 TSL 翻譯的核心能力。
    故 F1 掉超過比例、或 GER 絕對值上升超過門檻，任一成立即視為退化。
    """
    f1_drop = other.get("TokenF1", 0) < base.get("TokenF1", 0) * (1 - drop)
    ger_rise = other.get("GER", 0) > base.get("GER", 0) + drop
    return f1_drop or ger_rise


def diagnose(s_seen, s_para, s_unseen, thresholds):
    """依三層結果給初步判斷（對應四種情況）。

    判準用 ExactMatch 與 TokenF1／GER 一起看：EM 對長句過於嚴苛，
    本專案先前已量出逐句翻譯的 EM 理論上限僅約 23%，單看 EM 會誤判。
    """
    lo_seen, drop = thresholds
    seen_em = s_seen.get("ExactMatch%", 0)

    notes = []
    if seen_em < lo_seen:
        case = "情況 1：連訓練過的句子都做不好"
        action = ("優先懷疑模型根本沒學會。依序檢查：label masking 是否把答案也遮掉、"
                  "prompt/template 訓練與推論是否一致、tokenizer 是否正確、"
                  "LoRA 是否真的掛上目標模組、learning rate 是否過低、資料格式是否錯位。")
    elif degraded(s_seen, s_para, drop):
        case = "情況 2：訓練句好，但換句話說就掉"
        action = ("優先懷疑模型在記憶中文字面而非學會轉換。可做：中文端改寫增強、"
                  "同義詞替換增強、降低對特定表面詞的依賴。")
    elif degraded(s_seen, s_unseen, drop):
        case = "情況 3：訓練句與改寫都還行，但未見句子掉"
        action = ("優先懷疑句型多樣性與 composition 能力不足。可做：擴充句型覆蓋、"
                  "增加長句與複雜結構、檢查訓練資料的句型分布是否偏斜。")
    else:
        case = "情況 4：三層都在可接受範圍"
        action = ("模型基本具備泛化能力。接著應轉向 TSL 語法細節與人工評估"
                  "（見 scripts/make_human_eval_sheet.py 的盲測設計）。")

    if s_seen.get("ErrorTypes%", {}).get("語序錯誤", 0) > 15:
        notes.append("語序錯誤佔比偏高：選詞正確但排序錯，屬 TSL 語序規則未學牢，"
                     "非詞彙問題。")
    if s_unseen.get("ErrorTypes%", {}).get("OOV/未知Gloss", 0) > 15:
        notes.append("未見層 OOV 偏高：模型在造訓練沒出現過的詞，"
                     "可能是照抄中文而非檢索學過的 Gloss。")
    return {"判斷": case, "建議": action, "補充": notes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/gemma-4-E4B-it")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--ple", choices=["auto", "gpu", "cpu"], default="gpu",
                    help="PLE 放置；gpu＝全模型放 GPU（快約 300 倍），評估建議明指")
    ap.add_argument("--max-new", type=int, default=160)
    ap.add_argument("--tiers", nargs="+", default=["seen", "paraphrase", "unseen"])
    ap.add_argument("--seen-em-threshold", type=float, default=40.0,
                    help="Seen 層 ExactMatch 低於此值即判為『沒學會』")
    ap.add_argument("--drop-threshold", type=float, default=0.25,
                    help="相對 Seen 層 TokenF1 掉超過此比例即判為該層不足")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    vocab = train_vocab()
    print(f"[三層評估] 訓練 Gloss 詞彙 {len(vocab)} 詞")

    from train_qlora import load_model as load_base, can_fit_ple_on_gpu
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    ple_on_gpu = {"auto": None, "gpu": True, "cpu": False}[args.ple]
    if ple_on_gpu is None:
        ple_on_gpu = can_fit_ple_on_gpu()
    print(f"[三層評估] PLE 放置：{'GPU（快）' if ple_on_gpu else 'CPU（慢）'}", flush=True)
    model = load_base(args.base, bnb, ple_on_gpu=ple_on_gpu)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    summary = {"adapter": args.adapter, "tiers": {}}
    results = {}
    for tier in args.tiers:
        rows = load_tier(tier)
        if not rows:
            print(f"  略過 {tier}（找不到資料，請先跑 build_eval_tiers.py）")
            continue
        print(f"\n=== {tier}（{len(rows)} 句）===", flush=True)
        recs = run_tier(rows, model, tok, args, vocab)
        results[tier] = recs
        with (OUT / f"{tier}_results.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(recs)
        summary["tiers"][tier] = summarize(recs)

    # 改寫層另拆出「需語意映射」子群：這群最能區辨理解與背誦
    para = results.get("paraphrase")
    if para:
        hard = [r for r in para if r.get("needs_semantic_mapping") is True]
        easy = [r for r in para if r.get("needs_semantic_mapping") is not True]
        if hard:
            summary["tiers"]["paraphrase_需語意映射"] = summarize(hard)
        if easy:
            summary["tiers"]["paraphrase_字面仍有依據"] = summarize(easy)

    t = summary["tiers"]
    if all(k in t for k in ("seen", "paraphrase", "unseen")):
        summary["診斷"] = diagnose(t["seen"], t["paraphrase"], t["unseen"],
                                   (args.seen_em_threshold, args.drop_threshold))
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 62)
    print(f"{'層級':<26}{'EM%':>7}{'TokenF1':>9}{'GER':>7}{'BLEU':>7}")
    print("-" * 62)
    for name, m in t.items():
        print(f"{name:<26}{m.get('ExactMatch%', 0):>7.2f}{m.get('TokenF1', 0):>9.3f}"
              f"{m.get('GER', 0):>7.3f}{m.get('BLEU-4', 0):>7.2f}")
    if "診斷" in summary:
        print("\n" + summary["診斷"]["判斷"])
        print("→ " + summary["診斷"]["建議"])
        for n in summary["診斷"]["補充"]:
            print("• " + n)
    print(f"\n結果 → {OUT.relative_to(BASE)}/")


if __name__ == "__main__":
    main()
