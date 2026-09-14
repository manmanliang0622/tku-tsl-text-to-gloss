#!/usr/bin/env python3
"""語義向量檢索可行性量測（離線原型，不動 VM）。

回答一個問題：k=60 檢索「沒撈到但庫裡有」的參考 token，
語義向量檢索用 M 個額外候選名額能撈回幾個？

方法：
- 詞庫側：17k 筆 gloss_clean 各嵌一次（BGE-small-zh-v1.5）
- 句子側：整句 + 2~4 字元 n-gram 全部當 query，取每個 gloss 對所有
  query 的最高相似度排序
- 排除既有 k=60 候選後取前 M 個，看 miss 有沒有被撈回
- M ∈ {10, 20, 40}；同時記回收案例與 miss 的相似度排名分布
"""
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path("/Users/leo/Documents/語言模型")
sys.path.insert(0, str(BASE / "scripts"))
from sign_candidates import CandidateRetriever  # noqa: E402

SCRATCH = Path(__file__).parent
CACHE = SCRATCH / "gloss_emb_cache.npz"

import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

# Intel Mac 上 torch 只到 2.2.2、新版 sentence-transformers 不相容，
# 改用 transformers 直接做 BGE 編碼：CLS pooling + L2 normalize（官方用法）
_tok = AutoTokenizer.from_pretrained("BAAI/bge-small-zh-v1.5")
_mod = AutoModel.from_pretrained("BAAI/bge-small-zh-v1.5")
_mod.eval()


class _Encoder:
    @staticmethod
    def encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False):
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                b = _tok(texts[i:i + batch_size], padding=True, truncation=True,
                         max_length=128, return_tensors="pt")
                h = _mod(**b).last_hidden_state[:, 0]      # CLS
                if normalize_embeddings:
                    h = torch.nn.functional.normalize(h, dim=-1)
                outs.append(h.numpy())
        return np.concatenate(outs, axis=0)


model = _Encoder()

retr = CandidateRetriever()
rows = retr.rows
gloss_texts = [r["gloss_clean"] for r in rows]
gloss_ids = [r["sign_id"] for r in rows]

if CACHE.exists():
    gmat = np.load(CACHE)["gmat"]
    assert gmat.shape[0] == len(gloss_texts)
else:
    gmat = model.encode(gloss_texts, batch_size=256, show_progress_bar=True,
                        normalize_embeddings=True)
    np.savez_compressed(CACHE, gmat=gmat)
print(f"詞庫向量 {gmat.shape}", flush=True)


def ngrams(text: str):
    t = "".join(ch for ch in text if "一" <= ch <= "鿿")
    qs = {text.strip()}
    for n in (2, 3, 4):
        for i in range(len(t) - n + 1):
            qs.add(t[i:i + n])
    return [q for q in qs if q]


def probe(split: str, k: int = 60):
    data = [json.loads(l) for l in
            (BASE / "data" / "splits" / f"{split}.jsonl").read_text().splitlines()
            if l.strip()]
    Ms = (10, 20, 40)
    total_miss = 0
    recovered = {m: 0 for m in Ms}
    examples = []
    rank_hist = []
    for row in data:
        text = str(row.get("chinese", "")).strip()
        tokens = [t.strip() for t in str(row.get("gloss_text", "")).split("/") if t.strip()]
        cands = retr.candidates(text, k=k, exclude_id=row.get("id"))
        cand_ids = {c["sign_id"] for c in cands}
        miss = []
        for tok in tokens:
            sid = retr.resolve(tok)
            if sid is not None and sid not in cand_ids:
                miss.append((tok, sid))
        if not miss:
            continue
        total_miss += len(miss)
        qs = ngrams(text)
        qmat = model.encode(qs, batch_size=128, normalize_embeddings=True)
        sims = (gmat @ qmat.T).max(axis=1)          # 每個 gloss 的最佳相似度
        order = np.argsort(-sims)
        extra = []
        for idx in order:
            sid = gloss_ids[idx]
            if sid in cand_ids or sid in extra:
                continue
            extra.append(sid)
            if len(extra) >= max(Ms):
                break
        pos = {sid: i for i, sid in enumerate(extra)}
        for tok, sid in miss:
            p = pos.get(sid)
            rank_hist.append(p if p is not None else -1)
            if p is not None:
                for m in Ms:
                    if p < m:
                        recovered[m] += 1
                if len(examples) < 15 and p < 20:
                    examples.append(f"{text[:18]} … {tok} → rank {p+1}")
    out = {
        "split": split, "retrieval_miss_tokens": total_miss,
        **{f"recovered@+{m}": f"{recovered[m]} ({recovered[m]/total_miss:.1%})"
           for m in Ms},
        "miss_not_in_top40": sum(1 for r in rank_hist if r == -1),
    }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print("回收案例：")
    for e in examples:
        print("  ", e)
    return out


for split in ("dev", "test_corpus"):
    probe(split)
