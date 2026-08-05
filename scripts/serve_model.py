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
  POST /translate           → body {"text":"我要喝水"}
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

STATE = {"model": None, "tokenizer": None, "adapter": None, "max_new": 64}


def load(base_model, adapter):
    from train_qlora import load_model as load_base
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True)
    tok = AutoTokenizer.from_pretrained(base_model)
    model = load_base(base_model, bnb)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    STATE.update(model=model, tokenizer=tok, adapter=str(adapter))
    print(f"[serve] 模型就緒 adapter={adapter}", flush=True)


def translate(text):
    tok, model = STATE["tokenizer"], STATE["model"]
    msgs = pc.build_messages(text)
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True).to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=STATE["max_new"], do_sample=False)
    gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    gloss_text = pc.parse_gloss(gen)
    toks = [t for t in gloss_text.split("/") if t.strip()]
    return {"chinese": text, "gloss": toks, "gloss_text": gloss_text,
            "raw": gen.strip(), "seconds": round(time.time() - t0, 2)}


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
                             "adapter": STATE["adapter"]})
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
            if not text:
                self._send(400, {"error": "text 不可為空"})
                return
            if STATE["model"] is None:
                self._send(503, {"error": "模型尚未載入完成"})
                return
            self._send(200, translate(text))
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
    args = ap.parse_args()
    STATE["max_new"] = args.max_new
    print(f"[serve] 載入模型中…（adapter={args.adapter}）", flush=True)
    load(args.base, args.adapter)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[serve] 監聽 127.0.0.1:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
