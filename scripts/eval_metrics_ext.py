#!/usr/bin/env python3
"""三層評估用的擴充指標與錯誤分類（既有 metrics.py 的補充，不取代它）。

metrics.py 已有 BLEU-4／ROUGE-L／Exact Match／詞彙表內率，那些是「整批語料」
層級的指標。三層診斷需要的是**逐句**指標與**錯誤型態**，才能回答
「模型是沒學會，還是學會了但不會泛化」，故另立此模組：

  - token 層 Precision／Recall／F1（多重集合為基礎，容許重複詞）
  - GER（Gloss Error Rate）：仿 WER，(替換+刪除+插入)/參考詞數
  - 編輯距離：token 層 Levenshtein
  - 錯誤分類：語序／漏詞／多詞／替換／OOV／完全錯誤

**為什麼 P/R/F1 用多重集合而非集合**：Gloss 允許重複詞（如「等 等」表持續），
用 set 會把重複詞的漏譯算成正確。故以 Counter 取交集。

**錯誤分類的判定順序**（先判定者優先，理由見各分支註解）：
  正確 → 語序錯誤 → 完全錯誤 → OOV → 漏詞／多詞／替換（取主要編輯操作）
"""
from collections import Counter


def tokenize(gloss_text):
    """與 metrics.tokenize 相同的切詞規則（「/」或空白分隔皆可）。

    訓練目標的 JSON 內 gloss 以空白分隔，評估結果檔以「/」分隔，
    兩種都要能吃，故一律先把「/」換成空白再切。
    """
    t = str(gloss_text or "").replace("／", "/").replace("/", " ")
    return [x for x in t.split() if x]


def edit_ops(ref, hyp):
    """token 層 Levenshtein，回傳 (替換, 刪除, 插入, 距離)。

    刪除＝參考有但預測沒有（漏詞）；插入＝預測多出來的詞（多詞）。
    需要逐項操作數而不只是距離，才能做錯誤分類。
    """
    n, m = len(ref), len(hyp)
    # dp[i][j] = (距離, 替換數, 刪除數, 插入數)
    dp = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            d_sub, s, d, ins = dp[i - 1][j - 1]
            sub_c = (d_sub + 1, s + 1, d, ins)
            d_del, s2, d2, ins2 = dp[i - 1][j]
            del_c = (d_del + 1, s2, d2 + 1, ins2)
            d_ins, s3, d3, ins3 = dp[i][j - 1]
            ins_c = (d_ins + 1, s3, d3, ins3 + 1)
            dp[i][j] = min(sub_c, del_c, ins_c, key=lambda x: x[0])
    dist, sub, dele, ins = dp[n][m]
    return sub, dele, ins, dist


def token_prf(ref, hyp):
    """多重集合 token P/R/F1（允許重複詞，見檔頭說明）。"""
    if not ref and not hyp:
        return 1.0, 1.0, 1.0
    if not ref or not hyp:
        return 0.0, 0.0, 0.0
    overlap = sum((Counter(ref) & Counter(hyp)).values())
    p = overlap / len(hyp)
    r = overlap / len(ref)
    f = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return p, r, f


def gloss_error_rate(ref, hyp):
    """GER＝(替換+刪除+插入)/參考詞數，仿 WER。參考為空時回傳 0 或 1。"""
    if not ref:
        return 0.0 if not hyp else 1.0
    sub, dele, ins, _ = edit_ops(ref, hyp)
    return (sub + dele + ins) / len(ref)


def classify_error(ref, hyp, train_vocab=None):
    """回傳 (錯誤類型, 說明用細節 dict)。

    train_vocab：訓練集出現過的 Gloss 詞集合；用來認出模型自創/未學過的詞。
    傳 None 則不做 OOV 判定。
    """
    train_vocab = train_vocab or set()
    detail = {}
    if ref == hyp:
        return "正確", detail

    sub, dele, ins, dist = edit_ops(ref, hyp)
    detail.update(sub=sub, missing=dele, extra=ins, distance=dist)

    # 語序錯誤：詞完全相同（含重複次數）只是順序不同 —— 這是 TSL 最關鍵的能力，
    # 必須與「用錯詞」分開統計，否則看不出模型是不會選詞還是不會排序。
    if Counter(ref) == Counter(hyp):
        return "語序錯誤", detail

    if not hyp:
        return "完全錯誤", detail
    overlap = sum((Counter(ref) & Counter(hyp)).values())
    if overlap == 0:
        return "完全錯誤", detail

    # OOV：預測用了訓練集沒出現過的詞。優先於漏/多/替換回報，因為成因不同——
    # 前者是模型自創或照抄中文，後者是學過的詞用錯位置。
    oov = [t for t in hyp if t not in train_vocab] if train_vocab else []
    if oov:
        detail["oov_tokens"] = oov
        return "OOV/未知Gloss", detail

    # 其餘取主要編輯操作。並列時偏向回報「替換」，因為替換同時牽涉選詞與位置。
    if sub >= dele and sub >= ins:
        return "Gloss替換錯誤", detail
    if dele > ins:
        return "漏Gloss", detail
    return "多餘Gloss", detail


def score_pair(ref_text, hyp_text, train_vocab=None):
    """單句完整評分，供 CSV 逐列輸出。"""
    ref, hyp = tokenize(ref_text), tokenize(hyp_text)
    p, r, f = token_prf(ref, hyp)
    _, _, _, dist = edit_ops(ref, hyp)
    etype, detail = classify_error(ref, hyp, train_vocab)
    return {
        "exact_match": int(ref == hyp),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f, 4),
        "ger": round(gloss_error_rate(ref, hyp), 4),
        "edit_distance": dist,
        "error_type": etype,
        "ref_len": len(ref),
        "pred_len": len(hyp),
        "oov_tokens": " ".join(detail.get("oov_tokens", [])),
    }


def aggregate(rows):
    """整層平均。P/R/F1 取句平均（macro），GER 取語料層（總編輯數/總參考詞數）。

    GER 用語料層而非句平均，理由同 WER 慣例：句平均會讓短句的單一錯誤
    被過度放大（3 詞句錯 1 詞＝33%，10 詞句錯 1 詞＝10%）。
    """
    n = len(rows)
    if not n:
        return {}
    total_dist = sum(r["edit_distance"] for r in rows)
    total_ref = sum(r["ref_len"] for r in rows)
    types = Counter(r["error_type"] for r in rows)
    return {
        "n": n,
        "ExactMatch%": round(sum(r["exact_match"] for r in rows) / n * 100, 2),
        "TokenPrecision": round(sum(r["precision"] for r in rows) / n, 4),
        "TokenRecall": round(sum(r["recall"] for r in rows) / n, 4),
        "TokenF1": round(sum(r["f1"] for r in rows) / n, 4),
        "GER": round(total_dist / total_ref, 4) if total_ref else None,
        "GER_sentence_avg": round(sum(r["ger"] for r in rows) / n, 4),
        "AvgEditDistance": round(total_dist / n, 2),
        "ErrorTypes": dict(types.most_common()),
        "ErrorTypes%": {k: round(v / n * 100, 2) for k, v in types.most_common()},
    }
