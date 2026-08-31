#!/usr/bin/env python3
"""中文 → 臺灣手語腳本（tsl-script-v1）推論 API。

僅用標準函式庫的 http.server，避免在共用機安裝額外套件。
模型只載入一次常駐記憶體；每次請求做一次 greedy 生成——`--target script`
時預設帶約束解碼，把 sign_ids 鎖在該句候選清單內（見「約束解碼」段）。

線上（0821_bundle）由 bundle_server.py 以子行程啟動，實際參數見其 model_cmd()：
  .venv/bin/python3 model_service/scripts/serve_model.py \
    --base model_service/base_model --adapter model_service/checkpoint \
    --target script --max-new 256 --port 8878
現行模型：qlora_e4b_v17script_k40sem／checkpoint-558（2026-08-27 上線）。

API：
  GET  /health              → {"status":"ok","adapter":...,"model":...,"target":...}
  POST /translate           → body {"text":"我想喝水"}
                              回 {"chinese":...,"sign_ids":["TSL_我","TSL_想","TSL_水","TSL_喝"],
                                  "gloss":["我","想","水","喝"],"gloss_text":"我/想/水/喝",
                                  "candidate_coverage_risk":false,
                                  "candidate_coverage_risk_prob":0.026,
                                  "needs_review":false,   # 相容別名，同值
                                  "dropped_ids":[],"candidates_k":40,
                                  "schema_version":"tsl-script-v1","raw":"{...}","seconds":6.1}
  舊的 --target gloss／json 模式仍保留（歷史 adapter 用），欄位見 translate()。
  訓練配方與格式說明見封包 README「語言模型」一節。

CORS：前端以 file:// 開啟時 Origin 為 null，故一律回 Access-Control-Allow-Origin: *
（本服務只在 SSH 通道內對本機開放，不對外網暴露）。
"""
import argparse
import os
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

import prompt_common as pc
import script_schema

# ── 部署相依 ─────────────────────────────────────────────────────────
# 部署到 0821_bundle 時，這些 scripts/ 底下的模組**必須一起帶**，
# 少一個 serve_model 就起不來（或在請求進來時才炸）。
#
# 這不是備忘錄而是契約：scripts/check_bundle_deps.py 會用 AST 算出
# serve_model 的遞移本地相依，與這份清單對帳，不一致就讓 CI 紅。
# 2026-08-31 的教訓——commit 279d35c 一次加了 constrained_decode 與
# script_schema 兩個相依，但註解只提到一個、數量還寫錯成「第四個相依」，
# 那版部署上去會啟動即死。註解不會失敗，清單會。
BUNDLE_MODULES = (
    "comitative",          # 伴隨句雙數收攏（缺檔會 warn 後停用，不擋啟動）
    "constrained_decode",  # 約束解碼；缺檔**故意**讓服務起不來，見下方說明
    "eval_video_coverage",  # sign_candidates 的 gloss 正規化（fold／norm）
    "gloss_fallback",
    "prompt_common",
    "rag_retrieve",
    "script_schema",       # schema 常數與旗標欄位名；模組層 import，缺檔最先炸
    "sign_candidates",
    "train_qlora",         # load_model／can_fit_ple_on_gpu
)

BASE = Path(__file__).resolve().parent.parent

STATE = {"model": None, "tokenizer": None, "adapter": None, "model_name": None, "max_new": 64,
         "target": "gloss", "retriever": None, "rag_k": 0, "rag_min": 0.05,
         "json_targets": {}, "cand_retriever": None, "id2gloss": {}, "k": 40}

# ---- 手語腳本格式 ---------------------------------------------------------
# **這台服務目前載入的 checkpoint 是用哪個 schema 訓練的。**
# v17（部署中）＝ tsl-script-v1，旗標欄位 needs_review。
# 換上 v2 訓練的 checkpoint 時，這一行要跟著改，否則 system prompt 與訓練
# 不一致——2026-08-20 就因為推論端 prompt 與訓練不同，33 句 test 的 EM 與
# ValidJSON 全部掛零，看起來像模型壞掉，其實模型是好的。
DEPLOYED_SCHEMA = script_schema.V1

# 2026-08-31：原本這裡硬寫一份 system 字串，再用 ast 讀 build_script_dataset.py
# 的 SYSTEM 字面值比對，防兩份複本漂移。現在兩邊都從 script_schema 取同一份，
# 複本消失、比對也就不需要了（審查意見 4.3 的「共用模組」同樣適用於 prompt）。
SCRIPT_SYSTEM = script_schema.SYSTEM_BY_SCHEMA[DEPLOYED_SCHEMA]

# 門檻只能在 dev 上重選，不要看 test 的數字調。選法用
# scripts/nr_threshold.py（tku-tsl-text-to-gloss）重跑即可。
#
# 2026-08-27 改選法：舊規則「recall>=0.7 下最大化 precision」偏保守，
# 在 v17 dev 上選到 0.095349（P0.693/R0.710/F1 0.702）。改成直接最大化
# F1 選到 0.039707（P0.617/R0.928/F1 0.741），且在兩個測試集同向更好
# （corpus F1 0.827→0.905、textbook 0.653→0.702），所以不是過擬合 dev。
# 關鍵是漏放行大幅減少：dev 93→23、corpus 35→9、textbook 89→31——
# 這個旗標的錯誤本來就不對稱（漏放行會讓錯句直接送去給虛擬人比出來，
# 誤攔只是多一次人看），所以偏 recall 是對的方向。
NEEDS_REVIEW_THRESHOLD = 0.039707  # v17 dev 上最大化 F1（2026-08-27）


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


def _load_script_assets():
    """候選檢索器與 sign_id→gloss 對照。訓練與上線**必須共用同一支檢索器**，
    否則模型學到的候選分布與線上不同，約束就失效了。"""
    if STATE["cand_retriever"] is not None:
        return
    from sign_candidates import CandidateRetriever
    STATE["cand_retriever"] = CandidateRetriever()
    STATE["id2gloss"] = {r["sign_id"]: r.get("gloss_clean") or r["gloss"]
                         for r in STATE["cand_retriever"].by_id.values()}
    print(f"[serve] 候選檢索器就緒（{len(STATE['id2gloss'])} 個 sign_id）", flush=True)


def _needs_review_prob(tok, seq, scores):
    """覆蓋風險旗標那個位置給 true 的機率（只在 true/false 之間正規化）。"""
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
            return (pt / (pt + pf)) if (pt + pf) > 0 else None
        text += piece
    return None



# ── 約束解碼 ─────────────────────────────────────────────────────────
# 解碼時就把 sign_ids 陣列內的字串鎖在候選清單上，而不是生成後才過濾。
# 文字級狀態機：每步解碼已生成文字，游標在字串常值內時，只允許「能接成
# 某個候選 id」的 token；比對不到（模型用合併 token 帶進怪字首）就整步
# 放行，交回 translate_script 既有的事後過濾兜底——寧可漏擋，不可擋出
# 破 JSON。CONSTRAINED_DECODE=0 可整個關掉。
#
# 2026-08-27 追加退化守衛（MAX_RUN / MAX_SIGNS）。動機：greedy 解碼在離線
# 8 句上崩壞，最嚴重的 TB0296 參考只有 1 個詞、模型吐出 32 個（TSL_二十
# 連續 27 次）。原本的約束擋不住——重複的 id 本身就在候選清單裡，合法。
# **不要改用 no_repeat_ngram_size**：它是 token 級的，而元素分隔符 '", "'
# 每個元素都重複，n-gram 封鎖會直接吐出破 JSON；而且重複在臺灣手語裡是
# 合法的（重疊表複數／強調），參考答案有 5–10% 的句子重複用詞。
# 兩個上限取參考答案實測極值，10,200 句參考驗證過零排除。
# 與 tku-tsl-text-to-gloss/scripts/constrained_decode.py 同一套邏輯，改動要同步。
CONSTRAINED_DECODE = os.environ.get("CONSTRAINED_DECODE", "1") != "0"
# 伴隨句雙數收攏規則（見 comitative.py）。COMITATIVE_DUAL=0 可關掉。
# 只作用在線上服務；離線評估腳本刻意不套，讓 v17 之前的指標保持可比。
COMITATIVE_DUAL = os.environ.get("COMITATIVE_DUAL", "1") != "0"
# 部署到 0821_bundle 時 comitative.py **必須一起帶**（見上方 BUNDLE_MODULES）。這裡在載入時就吵，不要等到請求進來
# 才每次 import 失敗——那會變成「服務看起來正常、規則靜默失效」。
try:
    import comitative
except ImportError:                      # noqa: BLE001 - 缺檔不該讓整個服務起不來
    comitative = None
    print("[serve] ⚠ 找不到 comitative.py，伴隨句雙數收攏規則停用（部署時漏帶？）",
          flush=True)
# 2026-08-31：原本這裡有一份與 scripts/constrained_decode.py 逐行相同的副本，
# 靠 tests/test_serve_parity.py 守著不漂移。改成直接 import 同一份實作——
# 教授審查意見 4.3 要求「離線推論與服務端 import 同一份」。副本消失，
# 漂移的可能性也跟著消失。
#
# ⚠️ 部署到 0821_bundle 時 constrained_decode.py **必須一起帶**（見上方 BUNDLE_MODULES）。
# 這裡刻意**不**做 try/except 兜底：約束解碼是「輸出 ID 一定在候選內」這條
# 保證的來源，靜默停用會讓服務看起來正常、實際卻退回 v14 那種 4.22% 違反率。
# 缺檔就讓服務起不來，比帶病上線好。
try:
    from constrained_decode import constrained_prefix_fn as _constrained_prefix_fn
    from constrained_decode import MAX_RUN, MAX_SIGNS      # noqa: F401  供健康檢查回報
except ImportError as e:                 # noqa: BLE001
    raise SystemExit(
        "[serve] ✗ 找不到 constrained_decode.py——約束解碼是輸出正確性的保證，"
        "不允許靜默停用。請把 scripts/constrained_decode.py 一起部署到 "
        "model_service/scripts/ 後再啟動。") from e

def translate_script(text):
    """tsl-script-v1：跑候選檢索 → 模型從候選挑 sign_id → 對回 gloss 與影片。"""
    _load_script_assets()
    tok, model = STATE["tokenizer"], STATE["model"]
    retr = STATE["cand_retriever"]
    cands = retr.candidates(text, k=STATE["k"])
    user = {"text": text, "candidates": [c["sign_id"] for c in cands]}
    msgs = [{"role": "system", "content": SCRIPT_SYSTEM},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                     return_tensors="pt", return_dict=True).to(model.device)
    t0 = time.time()
    with torch.no_grad():
        prefix_fn = (_constrained_prefix_fn(tok, inputs["input_ids"].shape[1],
                                            [c["sign_id"] for c in cands])
                     if CONSTRAINED_DECODE else None)
        out = model.generate(**inputs, max_new_tokens=STATE["max_new"], do_sample=False,
                             prefix_allowed_tokens_fn=prefix_fn,
                             return_dict_in_generate=True, output_scores=True)
    seq = out.sequences[0][inputs["input_ids"].shape[1]:]
    gen = tok.decode(seq, skip_special_tokens=True)
    secs = round(time.time() - t0, 2)

    obj = _parse_json_output(gen) or {}
    cand_ids = {c["sign_id"] for c in cands}
    raw_ids = [str(x) for x in (obj.get("sign_ids") or [])]
    # 約束在**服務端**強制執行，不只在訓練目標裡。實測 v14 在 test_corpus 仍有
    # 0.7% 的 ID 落在候選外，其中還有總表查無的幻覺 ID（語義 ID 讓「造一個
    # 看起來合理的 ID」變容易）。這一行是新格式可播放率的最後保證。
    sign_ids = [i for i in raw_ids if i in cand_ids and i in STATE["id2gloss"]]
    dropped = [i for i in raw_ids if i not in sign_ids]

    p_nr = _needs_review_prob(tok, seq, out.scores)
    model_nr = script_schema.read_flag(obj)     # v1／v2 欄位名都收
    needs_review = (p_nr >= NEEDS_REVIEW_THRESHOLD) if p_nr is not None else model_nr

    toks = [STATE["id2gloss"][i] for i in sign_ids]

    # 伴隨句雙數收攏（2026-08-31）：「我跟X…」補「我們兩個」。
    # 為什麼在解碼後補而不是修候選：`我跟媽媽去吃飯` 的 k=40 候選裡根本沒有
    # 「我們兩個」，約束解碼不可能吐出它；而加候選通道在固定 k 之下是零和的，
    # 本專案已有三個否定結果（見 sign_candidates.candidates）。詳見 comitative.py。
    if COMITATIVE_DUAL and comitative is not None:
        sign_ids, toks, dual_added = comitative.apply_ids(
            text, sign_ids, toks, STATE["cand_retriever"].index, max_signs=MAX_SIGNS)
    else:
        dual_added = False

    return {
        "chinese": text, "gloss": toks, "gloss_text": "/".join(toks), "glosses": toks,
        "sign_ids": sign_ids,
        "comitative_dual": dual_added,
        # 2026-08-31 正名（審查意見 4.2）：這個旗標只反映「參考詞有沒有全部
        # 進候選」＝檢索覆蓋率風險，偵測不到選錯詞、語序錯、重複遺失、語意
        # 不自然。needs_review* 三個鍵**保留為相容別名**（前端在用），
        # 新的取用端請改讀 candidate_coverage_risk*。
        "candidate_coverage_risk": needs_review,
        "candidate_coverage_risk_prob": round(p_nr, 6) if p_nr is not None else None,
        "candidate_coverage_risk_model": model_nr,
        "needs_review": needs_review,
        "needs_review_prob": round(p_nr, 6) if p_nr is not None else None,
        "needs_review_model": model_nr,
        "oov_items": obj.get("oov_items") or [],
        "dropped_ids": dropped,
        "candidates_k": len(cands),
        "schema_version": obj.get("schema_version", DEPLOYED_SCHEMA),
        "source": "gemma", "model": STATE["model_name"],
        "raw": gen.strip(), "seconds": secs,
    }


def translate(text, context=""):
    if STATE["target"] == "script":
        return translate_script(text)
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
            text_value = data.get("text")
            context_value = data.get("context", "")
            if not isinstance(text_value, str):
                self._send(400, {"error": "text must be a string"})
                return
            if not isinstance(context_value, str):
                self._send(400, {"error": "context must be a string"})
                return
            text = text_value.strip()
            context = context_value.strip()
            if not text:
                self._send(400, {"error": "text 不可為空"})
                return
            if len(text) > 500:
                self._send(400, {"error": "text must not exceed 500 characters"})
                return
            if len(context) > 1000:
                self._send(400, {"error": "context must not exceed 1000 characters"})
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
    ap.add_argument("--target", choices=["gloss", "json", "script"], default="gloss",
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
