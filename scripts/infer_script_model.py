#!/usr/bin/env python3
"""用訓練好的 adapter 對「手語腳本」格式（tsl-script-v1）產生預測。

為什麼要另寫一支：`eval_json_model.py` 是走 `prompt_common` 組舊格式 prompt 的，
新格式的 messages 本身就是完整的 system/user/assistant，**不能也不該再包一層**。

**prompt 逐字一致**是這支腳本的核心設計：它直接取資料檔裡的 messages[0]（system）
與 messages[1]（user），原封不動套 chat template，不自行組字串。既有教訓——
2026-08-20 因為自組 prompt 與評估腳本不同，33 句 test 的 EM 與 ValidJSON 全部
掛零，看起來像模型壞掉，其實模型是好的。把 prompt 的來源綁死在資料檔就不會再犯。

**約束解碼與線上一致**（2026-08-31）：預設套用 `constrained_decode.py`，
與 `serve_model.py` import 同一份實作。在此之前這支沒有接約束，導致 v17cd
（線上部署的組態）在 repo 內無法離線重現——教授審查意見 4.3 指出的缺口。
要重現 v17 及更早的無約束數字，加 `--no-constrained`。

輸出每行含 `eval_script_format.py` 需要的欄位：
    id / raw（模型原始輸出）/ candidate_ids / ref_sign_ids
    / ref_candidate_coverage_risk / p_needs_review
另寫一份 `results/<tag>_<split>.manifest.json` 記錄 commit、adapter、資料夾與
其雜湊、解碼設定與完整命令列，讓每組數字都能追回產生它的組態。

用法：
    python3 scripts/infer_script_model.py \\
        --adapter ~/outputs/qlora_e4b_v17script_k40sem/checkpoint-558 \\
        --data data/splits_script_k40sem --split test --tag v17cd
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data" / "splits_script"
RESULTS = BASE / "results"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import constrained_decode  # noqa: E402  與 serve_model.py 同一份實作
import script_schema  # noqa: E402


def candidate_ids(user_content: str) -> list[str]:
    """取出該題的候選 ID。相容兩種序列化：'TSL_今天'（語義 ID）與 'TSL_01084=今天'。"""
    u = json.loads(user_content)
    out = []
    for c in u.get("candidates", []):
        if isinstance(c, dict):
            out.append(str(c.get("sign_id")))
        else:
            out.append(str(c).split("=", 1)[0])
    return out


def needs_review_prob(tok, seq, scores):
    """回傳模型在覆蓋風險旗標那個位置給 true 的機率（相對 false 正規化）。

    為什麼要這個（2026-08-21）：貪婪解碼只給硬 true/false，而 v14 在 test_corpus
    的表現是 precision 0.941／recall 0.118——它一開口幾乎都對，就是太少開口。
    這是典型的門檻過保守，用機率＋在 **dev** 上調門檻即可校準，不必重訓。
    ⚠️ 門檻只能在 dev 上調，拿 test_corpus 調就是對測試集調參。

    只在 true/false 兩個結果之間正規化，不看其他 token——那些與這個決策無關，
    納入只會讓機率被序列長度稀釋。找不到該位置時回 None（不猜）。
    """
    import torch
    text = ""
    for i, tid in enumerate(seq.tolist()):
        piece = tok.decode([tid])
        low = piece.strip().lower()
        if any(k in text for k in script_schema.FLAG_KEYS) and low[:4] in ("true", "fals"):
            if i >= len(scores):
                return None
            probs = torch.softmax(scores[i][0].float(), dim=-1)
            top = torch.topk(probs, 50)
            pt = pf = 0.0
            for val, idx in zip(top.values.tolist(), top.indices.tolist()):
                w = tok.decode([idx]).strip().lower()
                if w.startswith("true"):
                    pt += val
                elif w.startswith("fals"):
                    pf += val
            return round(pt / (pt + pf), 6) if (pt + pf) > 0 else None
        text += piece
    return None


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(BASE), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:                     # noqa: BLE001  沒有 git 不該擋住評估
        return None


def _sha256(path: Path) -> str | None:
    """檔案內容雜湊。資料夾就取其中 *.jsonl 依檔名排序後的串接雜湊。"""
    h = hashlib.sha256()
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    if not files:
        return None
    for f in files:
        if not f.is_file():
            return None
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def write_manifest(split: str, args, out_path: Path, n_rows: int) -> None:
    """把「這組數字是誰、用什麼組態產生的」寫在數字旁邊。

    審查意見 4.3 要求評估產物記錄 commit、模型與候選資料雜湊、解碼設定與
    完整命令。少了這個，results/ 裡的 jsonl 事後無法判斷是哪一版候選、
    有沒有開約束——v17 與 v17cd 的差別正是後者。
    """
    adapter = Path(args.adapter).expanduser()
    data_dir = Path(args.data)
    mf = {
        "tag": args.tag,
        "split": split,
        "n_rows": n_rows,
        "git_commit": _git_commit(),
        "adapter": str(adapter),
        "adapter_sha256": _sha256(adapter / "adapter_model.safetensors"),
        "data_dir": str(data_dir),
        "data_sha256": _sha256(data_dir / f"{split}.jsonl"),
        "constrained_decode": bool(args.constrained),
        "max_new": args.max_new,
        "max_len_ctx": args.max_len_ctx,
        "limit": args.limit or None,
        "command": " ".join(sys.argv),
    }
    if args.constrained:
        mf["constraint_guards"] = {"MAX_RUN": constrained_decode.MAX_RUN,
                                   "MAX_SIGNS": constrained_decode.MAX_SIGNS}
    mp = out_path.with_suffix(".manifest.json")
    mp.write_text(json.dumps(mf, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"寫出 {mp}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(Path.home() / "0813/model_service/base_model"))
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--split", nargs="+", default=["test"],
                    help="可一次給多個切分：同一程序內共用已載入的模型，"
                         "切分之間不釋放 GPU——2026-08-25 教訓：逐切分開新程序"
                         "會在模型重載的空窗被看門狗搶走 GPU，之後整批掉進"
                         "CPU offload 慢 70 倍")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", default=str(DATA),
                    help="切分資料夾。⚠️ 必須與訓練用的同一份（候選 k 不同就是"
                         "不同的任務分布），v15 起用 data/splits_script_k60")
    ap.add_argument("--max-new", type=int, default=256,
                    help="腳本目標比舊 JSON 長（sign_ids 陣列），預設放寬到 256")
    ap.add_argument("--max-len-ctx", type=int, default=768,
                    help="載入時的 max_seq_length，需 >= 訓練時的值")
    ap.add_argument("--limit", type=int, default=0)
    # 預設開啟＝與線上服務同組態（serve_model.py 的 CONSTRAINED_DECODE 預設也是開）。
    # v17 之前的報告數字是無約束產生的，要對照請明確加 --no-constrained。
    ap.add_argument("--constrained", dest="constrained", action="store_true", default=True,
                    help="解碼時把 sign_id 鎖在候選清單內（預設開，與線上一致）")
    ap.add_argument("--no-constrained", dest="constrained", action="store_false",
                    help="關閉約束解碼，用於重現 v17 及更早的無約束數字")
    args = ap.parse_args()

    import torch
    # 用 Unsloth 的 FastModel 載入，與 train_unsloth.py 同一條路徑。
    # 不要改用 transformers+PeftModel：unsloth venv 裡的 torchao 是 0.12.0，
    # peft 的 torchao 分派器要求 >0.16.0，會直接 ImportError（2026-08-21 實測）。
    # 載入方式與訓練一致本來就比較安全，順便繞開這個版本衝突。
    from unsloth import FastModel

    model, tok = FastModel.from_pretrained(
        model_name=args.adapter,          # adapter 目錄，Unsloth 會自己接回 base
        max_seq_length=args.max_len_ctx,
        load_in_4bit=True,
        full_finetuning=False,
    )
    FastModel.for_inference(model)
    # processor 與 tokenizer 的屬性位置不同，兩種都試
    _tk = getattr(tok, "tokenizer", tok)
    pad_id = _tk.pad_token_id if _tk.pad_token_id is not None else _tk.eos_token_id

    RESULTS.mkdir(exist_ok=True)
    for split in args.split:
        run_split(split, args, model, tok, pad_id)
    return 0


def run_split(split, args, model, tok, pad_id):
    import torch
    src = Path(args.data) / f"{split}.jsonl"
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    out_path = RESULTS / f"{args.tag}_{split}.jsonl"
    print(f"=== {split} ===", flush=True)
    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            msgs = r["messages"]
            # 原封不動用資料檔裡的 system+user，不自行組 prompt
            prompt = tok.apply_chat_template(msgs[:2], tokenize=False,
                                             add_generation_prompt=True)
            # 必須指名 text=：Gemma 4 的是多模態 processor，第一個位置參數是
            # images，直接傳字串會被當成圖片而 text 變成 None（TypeError）。
            enc = tok(text=prompt, return_tensors="pt").to(model.device)
            cand_ids = candidate_ids(msgs[1]["content"])
            # 約束是**逐題**的：候選清單每句不同，prefix_fn 不能跨題重用。
            prefix_fn = (constrained_decode.constrained_prefix_fn(
                tok, enc["input_ids"].shape[1], cand_ids) if args.constrained else None)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new,
                                     do_sample=False, pad_token_id=pad_id,
                                     prefix_allowed_tokens_fn=prefix_fn,
                                     return_dict_in_generate=True, output_scores=True)
            seq = gen.sequences[0][enc["input_ids"].shape[1]:]
            raw = tok.decode(seq, skip_special_tokens=True)
            p_nr = needs_review_prob(tok, seq, gen.scores)
            ref = json.loads(msgs[2]["content"])
            f.write(json.dumps({
                "id": r.get("metadata", {}).get("id"),
                "raw": raw,
                "candidate_ids": cand_ids,
                "ref_sign_ids": ref.get("sign_ids") or [],
                # 2026-08-31 正名（審查意見 4.2）。讀取端（eval_script_format._ref_flag、
                # nr_threshold）新舊鍵名都收，所以歷史結果檔不受影響。
                # 結構欄位的參考值。沒有這些，eval 的 StructureFields 就沒東西可比，
                # 而「沒有指標在看」正是 clause_breaks 死了很久沒人發現的原因。
                "ref_clause_breaks": ref.get("clause_breaks") or [],
                "ref_compounds": ref.get("compounds") or [],
                "ref_reduplicated": ref.get("reduplicated") or [],
                "ref_candidate_coverage_risk": script_schema.read_flag(ref),
                # 旗標那個位置的機率，供事後調門檻。鍵名維持 p_needs_review：
                # 它是純機率的管線鍵，results/ 既有檔與 nr_threshold 都靠它，
                # 改名的收益不抵風險。
                "p_needs_review": p_nr,
            }, ensure_ascii=False) + "\n")
            if i % 20 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}  {time.time()-t0:.0f}s", flush=True)
    print(f"寫出 {out_path}")
    write_manifest(split, args, out_path, len(rows))


if __name__ == "__main__":
    raise SystemExit(main())
