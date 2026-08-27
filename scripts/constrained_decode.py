#!/usr/bin/env python3
"""約束解碼（tsl-script-v1）：sign_ids 陣列內的字串在解碼時就鎖在候選清單上。

與 0821/model_service/scripts/serve_model.py 內嵌的版本同一套邏輯（2026-08-26），
抽成模組供 infer_script_model.py（離線評估）使用。之後若要改，兩處要一起改，
或把 serve_model 改成 import 這裡。

做法：文字級狀態機。每步把已生成文字解碼，游標在 "sign_ids": [...] 的字串
常值內時，只允許「能接成某個候選 id」的 token（詞表按首字索引，避免全表掃描）；
比對不到任何候選（模型用合併 token 帶進怪字首）就整步放行，交回呼叫端既有的
事後過濾兜底——寧可漏擋，不可擋出破 JSON。

2026-08-27 追加退化守衛（MAX_RUN / MAX_SIGNS）。動機：greedy 解碼在 8 句上
崩壞，最嚴重的 TB0296 參考只有 1 個詞、模型吐出 32 個（TSL_二十 連續 27 次）。
原本的約束擋不住這種錯——重複的 id 本身就在候選清單裡，完全合法。

**不要改用 no_repeat_ngram_size。** 兩個原因：
  1. 它是 token 級的，而輸出是 JSON——元素分隔符 '", "' 每個元素都會重複，
     n-gram 封鎖會直接吐出破 JSON。
  2. 重複在臺灣手語裡是合法的（重疊表複數／強調）。參考答案有 5–10% 的句子
     重複用詞，最極端的是「人 ×7」＝很多人。

兩個上限都取參考答案的實測極值，所以不會排除任何一句合法的參考：
  MAX_RUN=6   參考中「連續相同」最長就是 6（train 的 人/人/人/人/人/人）
  MAX_SIGNS=18  參考最長 15 個 sign（p99=12），留 3 個緩衝
"""
import re

MAX_RUN = 6        # 同一個 sign_id 最多連續出現幾次
MAX_SIGNS = 18     # sign_ids 陣列最多幾個元素

_VOCAB_BY_FIRST = None
_ELEMENT = re.compile(r'"([^"]*)"')


def _vocab_index(tok):
    global _VOCAB_BY_FIRST
    tok = getattr(tok, "tokenizer", tok)
    if _VOCAB_BY_FIRST is None:
        idx = {}
        for tid, piece in enumerate(tok.convert_ids_to_tokens(list(range(len(tok))))):
            if piece is None or (piece.startswith("<") and piece.endswith(">")):
                continue          # special / byte-fallback token 不參與
            text = piece.replace("▁", " ")
            if text:
                idx.setdefault(text[0], []).append((tid, text))
        _VOCAB_BY_FIRST = idx
    return _VOCAB_BY_FIRST


def _trailing_run(emitted):
    """已完成元素尾端「連續相同」的長度。"""
    if not emitted:
        return 0, None
    last = emitted[-1]
    run = 1
    for prev in reversed(emitted[:-1]):
        if prev != last:
            break
        run += 1
    return run, last


def constrained_prefix_fn(tok, prompt_len, cand_ids):
    """回傳給 model.generate(prefix_allowed_tokens_fn=...) 用的函式。

    tok 可以是 tokenizer 或 Processor（Gemma4Processor 之類）——後者自動解包。"""
    tok = getattr(tok, "tokenizer", tok)
    all_ids = list(range(len(tok)))
    index = _vocab_index(tok)
    cands = sorted(cand_ids)

    def fn(_batch, input_ids):
        gen = tok.decode(input_ids[prompt_len:], skip_special_tokens=True)
        m = gen.rfind('"sign_ids"')
        if m < 0:
            return all_ids
        seg = gen[m + 10:]
        b = seg.find("[")
        if b < 0 or "]" in seg[b:]:
            return all_ids                      # 還沒進陣列，或陣列已關閉
        seg = seg[b + 1:]

        # 已經完成的元素（成對引號才算；結尾未閉合的那個不列入）
        emitted = _ELEMENT.findall(seg)

        if seg.count('"') % 2 == 0:
            # 不在字串常值內。長度到頂就收掉陣列——但只在沒有逗號待補時，
            # 否則會生出 ["a", ] 這種尾逗號破 JSON。
            if len(emitted) >= MAX_SIGNS and not seg.rstrip().endswith(","):
                close = sorted({tid for tid, text in index.get("]", [])
                                if text.startswith("]")})
                if close:
                    return close
            return all_ids

        partial = seg[seg.rfind('"') + 1:]

        # 連續重複到頂：把那個 id 從這一步的候選裡拿掉，逼模型改選別的。
        run, last = _trailing_run(emitted)
        banned = {last} if last is not None and run >= MAX_RUN else set()

        allowed = set()
        for c in cands:
            if c in banned or not c.startswith(partial):
                continue
            rest = c[len(partial):]
            if not rest:                        # 候選打完了 → 只准關引號
                for tid, _text in index.get('"', []):
                    allowed.add(tid)
                continue
            for tid, text in index.get(rest[0], []):
                if rest.startswith(text) or text.startswith(rest + '"'):
                    allowed.add(tid)
        return sorted(allowed) if allowed else all_ids
    return fn
