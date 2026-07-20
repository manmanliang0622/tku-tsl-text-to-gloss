#!/usr/bin/env python3
"""Gloss 序列評估指標（計畫 6.1 節）。

- BLEU-4：Papineni et al. 2002，corpus-level，n≥2 采 add-1 平滑，含 brevity penalty
- ROUGE-L：Lin 2004，LCS 為基礎的 F1（β=1），句級平均
- Exact Match：Gloss 序列完全一致比例
- In-Vocabulary Rate（詞彙表內率，本專案自訂）：輸出 token 落在 Gloss 詞彙總表
  （data/tsl_gloss_vocab.json）內的比例；表外 Gloss 下游動作庫檢索不到

Token 化：Gloss 字串以「/」切分（與標記表格式一致），不做其他正規化。
"""
import math
from collections import Counter


def tokenize(gloss_text: str) -> list:
    return [t for t in gloss_text.replace("／", "/").split("/") if t.strip()]


def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(references: list, hypotheses: list, max_n: int = 4) -> float:
    """references/hypotheses: list of token lists（單一參考譯文）。"""
    assert len(references) == len(hypotheses)
    p_num, p_den = [0] * max_n, [0] * max_n
    ref_len = hyp_len = 0
    for ref, hyp in zip(references, hypotheses):
        ref_len += len(ref)
        hyp_len += len(hyp)
        for n in range(1, max_n + 1):
            hyp_ng, ref_ng = _ngrams(hyp, n), _ngrams(ref, n)
            p_num[n - 1] += sum(min(c, ref_ng[g]) for g, c in hyp_ng.items())
            p_den[n - 1] += max(sum(hyp_ng.values()), 0)
    if hyp_len == 0:
        return 0.0
    log_p = 0.0
    for n in range(max_n):
        num, den = p_num[n], p_den[n]
        if n >= 1:  # n>=2 add-1 平滑
            num, den = num + 1, den + 1
        if num == 0 or den == 0:
            return 0.0
        log_p += math.log(num / den)
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return bp * math.exp(log_p / max_n) * 100


def _lcs_len(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] \
                else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l_f1(references: list, hypotheses: list) -> float:
    scores = []
    for ref, hyp in zip(references, hypotheses):
        if not ref or not hyp:
            scores.append(0.0)
            continue
        lcs = _lcs_len(ref, hyp)
        p, r = lcs / len(hyp), lcs / len(ref)
        scores.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return sum(scores) / len(scores) * 100 if scores else 0.0


def exact_match(references: list, hypotheses: list) -> float:
    hits = sum(1 for r, h in zip(references, hypotheses) if r == h)
    return hits / len(references) * 100 if references else 0.0


def in_vocab_rate(hypotheses: list, vocab: set) -> float:
    toks = [t for h in hypotheses for t in h]
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in vocab) / len(toks) * 100


def evaluate(refs_text: list, hyps_text: list, vocab: set) -> dict:
    refs = [tokenize(r) for r in refs_text]
    hyps = [tokenize(h) for h in hyps_text]
    return {
        "BLEU-4": round(corpus_bleu(refs, hyps), 2),
        "ROUGE-L": round(rouge_l_f1(refs, hyps), 2),
        "ExactMatch%": round(exact_match(refs, hyps), 2),
        "InVocab%": round(in_vocab_rate(hyps, vocab), 2),
        "n": len(refs),
    }
