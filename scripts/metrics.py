#!/usr/bin/env python3
"""Gloss 序列評估指標（計畫 6.1 節）。

- BLEU-4：Papineni et al. 2002，corpus-level，n≥2 采 add-1 平滑，含 brevity penalty
- ROUGE-L：Lin 2004，LCS 為基礎的 F1（β=1），句級平均
- Exact Match：Gloss 序列完全一致比例
- In-Vocabulary Rate（詞彙表內率，本專案自訂）：輸出 token 落在 Gloss 詞彙表內的比例。

  ⚠️ 2026-08-04 修正（重要）：此指標必須與**參考答案的天花板**一起看。
  診斷（v4 擴大 584 句 test）發現：模型內率 69.74%、**參考答案內率僅 69.29%**，
  即完美複製標準答案也只有 69.29%——低內率反映的是「詞彙表收錄不足」，
  不是「模型亂造詞」。專案 coverage.json 佐證：文化部語料庫 Gloss token
  僅 70.7% 在辭典內。故 evaluate() 一律回報 InVocabRef%（天花板）與
  InVocabGap（模型−天花板，>0 表示模型用詞比標準答案更保守）。

Token 化：Gloss 字串以「/」切分（與標記表格式一致）。
詞彙比對正規化（normalize_gloss）：臺→台、去重複記號 ++、去 _N/_B 轉寫後綴、
去括號註、去頭尾標點。僅用於「詞彙表內率」比對，不影響 BLEU/ROUGE/EM 的字面比較。
"""
import math
import random
import re
from collections import Counter


def tokenize(gloss_text: str) -> list:
    return [t for t in gloss_text.replace("／", "/").split("/") if t.strip()]


def normalize_gloss(token: str) -> str:
    """詞彙表比對用的正規化（不改變 BLEU/ROUGE/EM 的字面比較）。

    處理實測發現的系統性落差，例如「臺灣」因辭典收「台灣」而被誤判表外、
    語料庫轉寫的重複記號（颱風++）與句末標點（什麼?）。
    """
    t = str(token).replace("臺", "台")
    t = re.sub(r"\+\+$", "", t)          # 語料庫重複動作記號
    t = re.sub(r"_[A-Za-z]+$", "", t)    # 辭典轉寫後綴 _N/_B
    t = re.sub(r"[（(][^）)]*[）)]", "", t)  # 括號註
    return t.strip(" ?？。，、!！") or str(token)


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


def in_vocab_rate(hypotheses: list, vocab: set, normalize: bool = True) -> float:
    """token 落在詞彙表內的比例；預設做 normalize_gloss 正規化後比對。"""
    toks = [t for h in hypotheses for t in h]
    if not toks:
        return 0.0
    if normalize:
        v = {normalize_gloss(x) for x in vocab}
        hit = sum(1 for t in toks if normalize_gloss(t) in v)
    else:
        hit = sum(1 for t in toks if t in vocab)
    return hit / len(toks) * 100


def ngram_count(sequences: list, n: int) -> int:
    return sum(max(len(tokens) - n + 1, 0) for tokens in sequences)


def _percentile(sorted_values: list, q: float) -> float:
    if not sorted_values:
        return 0.0
    pos = (len(sorted_values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight


def bootstrap_bleu_ci(references: list, hypotheses: list, groups: list,
                      n_samples: int = 1000, seed: int = 42) -> list:
    """以 group 為抽樣單位計算 corpus BLEU-4 的 percentile 95% CI。

    同一對話的句子高度相關，故擴大真實 test 不以逐句 bootstrap，而以 37 個
    seg_uuid 對話群組整組重抽樣；核心 test 每句自成一組。
    """
    assert len(references) == len(hypotheses) == len(groups)
    if not references or n_samples <= 0:
        return [0.0, 0.0]
    by_group = {}
    for i, group in enumerate(groups):
        by_group.setdefault(group, []).append(i)
    units = list(by_group)
    rng = random.Random(seed)
    scores = []
    for _ in range(n_samples):
        indices = []
        for _ in units:
            indices.extend(by_group[rng.choice(units)])
        scores.append(corpus_bleu(
            [references[i] for i in indices],
            [hypotheses[i] for i in indices],
        ))
    scores.sort()
    return [
        round(_percentile(scores, 0.025), 2),
        round(_percentile(scores, 0.975), 2),
    ]


def evaluate(refs_text: list, hyps_text: list, vocab: set, groups: list = None,
             bootstrap_samples: int = 0, bootstrap_seed: int = 42) -> dict:
    refs = [tokenize(r) for r in refs_text]
    hyps = [tokenize(h) for h in hyps_text]
    if groups is None:
        groups = [f"row:{i}" for i in range(len(refs))]
    assert len(groups) == len(refs)
    result = {
        "BLEU-4": round(corpus_bleu(refs, hyps), 2),
        "ROUGE-L": round(rouge_l_f1(refs, hyps), 2),
        "ExactMatch%": round(exact_match(refs, hyps), 2),
        # 內率一律附上參考答案天花板：低內率多半是詞彙表收錄不足，非模型亂造詞
        "InVocab%": round(in_vocab_rate(hyps, vocab), 2),
        "InVocabRef%": round(in_vocab_rate(refs, vocab), 2),
        "InVocabGap": round(in_vocab_rate(hyps, vocab)
                            - in_vocab_rate(refs, vocab), 2),
        "InVocab%(raw)": round(in_vocab_rate(hyps, vocab, normalize=False), 2),
        "n": len(refs),
        "n_groups": len(set(groups)),
        "Reference4Grams": ngram_count(refs, 4),
        "Hypothesis4Grams": ngram_count(hyps, 4),
    }
    if bootstrap_samples > 0:
        result["BLEU-4_95%CI"] = bootstrap_bleu_ci(
            refs, hyps, groups, bootstrap_samples, bootstrap_seed)
        result["BLEU-bootstrap-samples"] = bootstrap_samples
        result["BLEU-bootstrap-seed"] = bootstrap_seed
        result["BLEU-bootstrap-unit"] = "group"
    return result
