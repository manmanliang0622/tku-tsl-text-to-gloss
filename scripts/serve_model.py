#!/usr/bin/env python3
"""中文 → TSL Gloss 推論 API（供前端 /手語前端優化 測試用）。

僅用標準函式庫的 http.server，避免在共用機安裝額外套件。
模型只載入一次常駐記憶體；每次請求做一次 greedy 生成。

啟動（VM 上）：
  python3 scripts/serve_model.py --adapter outputs/qlora_e4b_v6_all/checkpoint-XXX --port 8018

前端（Mac）經 SSH 通道連入：
  ssh -p 2288 -N -L 8018:localhost:8018 b310ai@<VM>

API：
  GET  /health              → {"status":"ok","adapter":...}
  POST /translate           → body {"text":"我要喝水","context":"前一句（選填）"}
                              回 {"chinese":...,"gloss":["我","水","喝","要"],
                                  "gloss_text":"我/水/喝/要","seconds":1.2}

CORS：前端以 file:// 開啟時 Origin 為 null，故一律回 Access-Control-Allow-Origin: *
（本服務只在 SSH 通道內對本機開放，不對外網暴露）。
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

import prompt_common as pc

BASE = Path(__file__).resolve().parent.parent

STATE = {"model": None, "tokenizer": None, "adapter": None, "model_name": None, "max_new": 64,
         "target": "gloss", "retriever": None, "rag_k": 0, "rag_min": 0.05,
         "json_targets": {}}


def load(base_model, adapter, ple_on_gpu=None):
    """ple_on_gpu=None 時自動判斷（顯存夠就放 GPU，快約 30 倍）。"""
    from train_qlora import load_model as load_base, can_fit_ple_on_gpu
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True)
    tok = AutoTokenizer.from_pretrained(base_model)
    if ple_on_gpu is None:
        ple_on_gpu = can_fit_ple_on_gpu()
    print(f"[serve] PLE 放置：{'GPU（加速）' if ple_on_gpu else 'CPU（省顯存，較慢）'}",
          flush=True)
    model = load_base(base_model, bnb, ple_on_gpu=ple_on_gpu)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    STATE.update(model=model, tokenizer=tok, adapter=str(adapter))
    print(f"[serve] 模型就緒 adapter={adapter}", flush=True)


def _parse_json_output(raw):
    """JSON 目標模型的輸出：取出整包 JSON。失敗回 None。"""
    import re
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _clean_key(text):
    """與 scripts/build_json_targets.py 的 clean_chinese 一致的正規化。"""
    return str(text or "").strip().strip("，。！？；：,.!?;: ")


def _rag_examples(text):
    """檢索訓練資料中最相似的例句（RAG）。回傳 [(中文, 目標字串), ...]。"""
    r = STATE.get("retriever")
    if not r or STATE.get("rag_k", 0) <= 0:
        return [], []
    hits = r.search(text, k=STATE["rag_k"], min_score=STATE.get("rag_min", 0.05))
    pairs, info = [], []
    for score, row in hits:
        # 例句的目標格式要與模型訓練目標一致
        if STATE["target"] == "json":
            # build_json_targets 會清掉句尾標點，查表時需用同樣的正規化
            tgt = (STATE["json_targets"].get(row["chinese"])
                   or STATE["json_targets"].get(_clean_key(row["chinese"])))
            if not tgt:
                continue
        else:
            tgt = row["gloss_text"]
        pairs.append((row["chinese"], tgt))
        info.append({"score": score, "chinese": row["chinese"],
                     "gloss_text": row["gloss_text"]})
    return pairs, info


def _apply_fallback(toks):
    """把表外 Gloss 修成可播放的詞，修不了的標為指拼（見 scripts/gloss_fallback.py）。

    回傳 (修復後詞串列, 需指拼的原詞串列)。後者放進回應的 `unknown` 欄位，
    讓下游知道哪些詞得走指拼，而不是收到一個查不到的假 Gloss 就默默失敗。
    """
    if not STATE.get("fallback"):
        return toks, []
    import gloss_fallback as fb
    vocab, rend = fb.load_vocab()
    fixed, unknown = [], []
    for t in toks:
        parts, rule = fb.repair_token(t, vocab, rend)
        if rule == "fingerspell":
            unknown.append(t)
        fixed.extend(parts)
    return fixed, unknown


def translate(text, context=""):
    tok, model = STATE["tokenizer"], STATE["model"]
    ex_pairs, ex_info = _rag_examples(text)
    # v12 以段落前文訓練；呼叫端若提供 context，推論必須用同一格式送入。
    msgs = pc.build_messages(text, examples=ex_pairs, context=context)
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True).to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=STATE["max_new"], do_sample=False)
    gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    secs = round(time.time() - t0, 2)

    if STATE["target"] == "json":
        obj = _parse_json_output(gen) or {}
        toks = [t for t in str(obj.get("gloss", "")).split() if t.strip()]
        toks, unknown = _apply_fallback(toks)
        return {
            "chinese": text, "gloss": toks, "gloss_text": "/".join(toks),
            # 下游虛擬人需要的語法資訊（計畫第 1 節：表情/頭部/身體同步）
            "question_type": obj.get("question_type", "none"),
            "negation": obj.get("negation", False),
            "nonmanual": obj.get("nonmanual", "none"),
            "topic": obj.get("topic"), "verb": obj.get("verb"),
            "time": obj.get("time"),
            # 相容欄位（0804try 前端用 glosses/question）
            "glosses": toks, "question": obj.get("question_type", "none"),
            "source": "gemma", "model": STATE["model_name"], "unknown": unknown,
            "context": context,
            "raw": gen.strip(), "seconds": secs, "rag": ex_info,
        }

    gloss_text = pc.parse_gloss(gen)
    toks = [t for t in gloss_text.split("/") if t.strip()]
    toks, unknown = _apply_fallback(toks)
    gloss_text = "/".join(toks)
    return {"chinese": text, "gloss": toks, "gloss_text": gloss_text,
            "glosses": toks, "unknown": unknown,
            "source": "gemma", "model": STATE["model_name"],
            "context": context,
            "raw": gen.strip(), "seconds": secs, "rag": ex_info}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, {"status": "ok" if STATE["model"] else "loading",
                             "adapter": STATE["adapter"],
                             "model": STATE["model_name"],
                             "target": STATE["target"]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/translate"):
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            text = (data.get("text") or "").strip()
            context = (data.get("context") or "").strip()
            if not text:
                self._send(400, {"error": "text 不可為空"})
                return
            if STATE["model"] is None:
                self._send(503, {"error": "模型尚未載入完成"})
                return
            self._send(200, translate(text, context=context))
        except Exception as e:  # 回傳錯誤而非讓連線中斷，前端才能顯示原因
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        print(f"[serve] {self.address_string()} {fmt % args}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="google/gemma-4-E4B-it")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--port", type=int, default=8018)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--model-name", default=None,
                    help="API 顯示名稱；預設由 adapter 的上層目錄推斷")
    ap.add_argument("--target", choices=["gloss", "json"], default="gloss",
                    help="模型的輸出格式；json 版會另外回傳 question_type/negation/nonmanual")
    ap.add_argument("--rag", type=int, default=0,
                    help="推理時檢索 N 筆訓練集相似例句放進 prompt（0=關閉）。"
                         "依據 CCL24-Eval 與工研院 ITRI 的 RAG/ICL 做法")
    ap.add_argument("--rag-min", type=float, default=0.05,
                    help="檢索相似度下限，低於此值不放入（避免不相關例句干擾）")
    ap.add_argument("--no-fallback", action="store_true",
                    help="關閉表外 Gloss 修復。預設開啟：修不了的詞標為指拼放進 "
                         "unknown 欄位，避免下游收到查不到的假 Gloss 而默默失敗")
    ap.add_argument("--ple", choices=["auto", "gpu", "cpu"], default="auto",
                    help="PLE 放置：auto 依顯存自動判斷（預設）；gpu 強制加速；cpu 省顯存")
    args = ap.parse_args()
    STATE["max_new"] = args.max_new
    STATE["target"] = args.target
    STATE["fallback"] = not args.no_fallback
    if STATE["fallback"]:
        import gloss_fallback as _fb
        _v, _r = _fb.load_vocab()
        print(f"[serve] 表外 Gloss 修復已啟用：合法詞 {len(_v)}、可播放 {len(_r)}；"
              f"修不了的會標為指拼並列入 unknown 欄位", flush=True)
    adapter_path = Path(args.adapter.rstrip("/"))
    STATE["model_name"] = args.model_name or adapter_path.parent.name or adapter_path.name
    if args.target == "json" and args.max_new < 160:
        STATE["max_new"] = 160        # JSON 目標較長，實測最多 182 token
    STATE["rag_k"], STATE["rag_min"] = args.rag, args.rag_min
    if args.rag > 0:
        from rag_retrieve import Retriever
        STATE["retriever"] = Retriever()
        if args.target == "json":
            # JSON 模式的示範也要是 JSON，否則格式不一致會誤導模型
            import json as _json
            STATE["json_targets"] = {
                _json.loads(l)["input"]: _json.loads(l)["output"]
                for l in (BASE / "data/splits_json/train.jsonl")
                .read_text(encoding="utf-8").splitlines() if l.strip()}
            print(f"[serve] JSON 目標索引 {len(STATE['json_targets'])} 筆", flush=True)
        print(f"[serve] RAG 已啟用：每次檢索 {args.rag} 筆（相似度 ≥ {args.rag_min}）",
              flush=True)
    print(f"[serve] 載入模型中…（adapter={args.adapter}）", flush=True)
    load(args.base, args.adapter,
         ple_on_gpu={"auto": None, "gpu": True, "cpu": False}[args.ple])
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[serve] 監聽 127.0.0.1:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
