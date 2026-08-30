#!/usr/bin/env python3
"""語義向量檢索通道（可選，預設不啟用）。

為什麼要有這一支（2026-08-23）：候選檢索的最大缺口是語義距離遠的對應
（學到了→學習、作畫→繪畫、辦畫展→辦展覽），字面／共現／同義詞表都撈不到。
離線量測（results/semantic_retrieval_probe_2026-08-23.md）顯示向量檢索在
k=60 之外再給 40 個名額可回收約 21% 的缺口；本模組把它做成候選檢索器的
一個通道，由 build_script_dataset.py 以 --n-sem 啟用。

設計上的三個決定：
  - **預設關閉、延遲載入**：sign_candidates.py 不 import 本模組，線上服務
    （共用機，不裝額外依賴）完全不受影響。只有建資料時在 Mac 端啟用。
  - **向量矩陣快取**：17k 筆 gloss 嵌一次存 npz，附 gloss 清單比對，
    總表換了就自動重算。
  - **查詢快取**：每句的排名結果存 json，換配方（名額分配）不必重嵌。

模型：BAAI/bge-small-zh-v1.5（CLS pooling + L2 normalize，官方用法）。
環境：Mac 用 .venv-emb（Intel Mac 只能 torch 2.2.2，故直接用 transformers
而非 sentence-transformers）。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMB_CACHE = BASE / "data" / "signs" / "gloss_emb_bge_small_zh.npz"
QUERY_CACHE = BASE / "data" / "signs" / "semantic_query_cache.json"


def _ngrams(text: str) -> list[str]:
    t = "".join(ch for ch in text if "一" <= ch <= "鿿")
    qs = {text.strip()}
    for n in (2, 3, 4):
        for i in range(len(t) - n + 1):
            qs.add(t[i:i + n])
    # **一定要排序**：set 的迭代順序隨 PYTHONHASHSEED 每個行程都不同，而這個
    # 順序就是 encode() 的批次順序——批次組成不同會讓 transformer 的浮點結果
    # 差在末位，近似並列的候選排名因此翻面。2026-08-30 實測：同一條建資料指令
    # 連跑兩次，40 句裡有 3 句候選清單不同；固定 PYTHONHASHSEED 後降為 0 句。
    return sorted(q for q in qs if q)


class SemanticRanker:
    def __init__(self, rows: list[dict], top: int = 100,
                 emb_cache: Path = EMB_CACHE, query_cache: Path = QUERY_CACHE):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        self._mod = AutoModel.from_pretrained(MODEL_NAME).eval()
        self.top = top
        self.glosses = [r["gloss"] for r in rows]        # 動作庫原鍵（add() 用）
        texts = [r["gloss_clean"] for r in rows]
        sig = hashlib.sha256(("\n".join(texts) + MODEL_NAME).encode()).hexdigest()[:16]
        self._emb_cache = emb_cache
        if emb_cache.exists():
            z = np.load(emb_cache, allow_pickle=False)
            if str(z["sig"]) == sig:
                self.gmat = z["gmat"]
            else:
                self.gmat = self._build(texts, sig)
        else:
            self.gmat = self._build(texts, sig)
        self._qcache_path = query_cache
        self._qcache: dict[str, list[str]] = {}
        if query_cache.exists():
            self._qcache = json.loads(query_cache.read_text(encoding="utf-8"))
        self._dirty = 0

    def _build(self, texts, sig):
        print(f"[semantic] 嵌入 {len(texts)} 筆 gloss（首次，之後用快取）", flush=True)
        gmat = self.encode(texts, batch_size=256)
        self._emb_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self._emb_cache, gmat=gmat, sig=np.array(sig))
        return gmat

    def encode(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        torch = self._torch
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                b = self._tok(texts[i:i + batch_size], padding=True, truncation=True,
                              max_length=128, return_tensors="pt")
                h = self._mod(**b).last_hidden_state[:, 0]
                outs.append(torch.nn.functional.normalize(h, dim=-1).numpy())
        return np.concatenate(outs, axis=0)

    def rank(self, text: str) -> list[str]:
        """回傳與句子（整句＋n-gram）語義最近的 gloss 原鍵，最多 top 筆。"""
        key = text.strip()
        if key in self._qcache:
            return self._qcache[key]
        qmat = self.encode(_ngrams(key))
        sims = (self.gmat @ qmat.T).max(axis=1)
        # kind="stable"：分數完全相同時依 gloss 在總表的順序決定，不看排序法內部實作
        order = np.argsort(-sims, kind="stable")[: self.top]
        ranked = [self.glosses[i] for i in order]
        self._qcache[key] = ranked
        self._dirty += 1
        if self._dirty >= 200:
            self.flush()
        return ranked

    def flush(self) -> None:
        if self._dirty:
            self._qcache_path.write_text(
                json.dumps(self._qcache, ensure_ascii=False), encoding="utf-8")
            self._dirty = 0
