#!/usr/bin/env python3
"""constrained_decode 的狀態機測試（不需要載入真模型）。

用一個字元級假 tokenizer 驅動 prefix_allowed_tokens_fn，檢查三件事：
  1. 候選約束照舊生效
  2. 連續重複到 MAX_RUN 會被擋（TB0296 的退化模式）
  3. 長度守衛不會製造破 JSON（尾逗號的情況必須放行）
"""
import os
import string
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import constrained_decode as cd


class FakeTok:
    """每個 token 就是一個字串；id = 在 vocab 裡的位置。"""

    def __init__(self, pieces):
        self.pieces = list(pieces)

    def __len__(self):
        return len(self.pieces)

    def convert_ids_to_tokens(self, ids):
        return [self.pieces[i] for i in ids]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.pieces[i] for i in ids)

    def encode(self, text):
        """貪婪最長匹配，只用來組測試前綴。"""
        out, i = [], 0
        while i < len(text):
            longest = max(len(x) for x in self.pieces)
            for length in range(min(longest, len(text) - i), 0, -1):
                piece = text[i:i + length]
                if piece in self.pieces:
                    out.append(self.pieces.index(piece))
                    i += length
                    break
            else:
                raise AssertionError(f"假 tokenizer 缺 token: {text[i]!r}")
        return out


CANDS = ["TSL_二十", "TSL_三十", "TSL_我", "TSL_你"]
PIECES = sorted({
    '"', ",", " ", "[", "]", "{", "}", ":", "TSL_", "二十", "三十", "我", "你",
    '"sign_ids"', '", "', '"]',
} | set("TSL_二十三我你") | set(string.ascii_letters) | set("_")
    # v3 的 compounds／reduplicated 是索引陣列，測那兩個欄位需要數字
    | set(string.digits))


def test_v3_fields_do_not_reengage_constraint():
    """schema v3 在 sign_ids 之後多了 compounds／reduplicated 兩個索引陣列。

    這是刻意的設計選擇——用索引陣列而不是巢狀物件，就是為了讓約束解碼
    完全不必改：它只管 `"sign_ids": [...]` 內的字串常值，陣列一關閉就整步
    放行。這裡把它釘住，尤其是 compounds 的 `[[` 巢狀括號不會讓狀態機
    誤判成又進了 sign_ids 陣列。
    """
    tok = FakeTok(PIECES)
    total = len(tok.pieces)
    for prefix, want_free in [
        ('"sign_ids": ["TSL_', False),                      # 陣列內：受約束
        ('"sign_ids": ["TSL_我"], "compounds": [[', True),   # 巢狀括號：放行
        ('"sign_ids": ["TSL_我"], "compounds": [[0, 1]], "reduplicated": [', True),
        ('"sign_ids": ["TSL_我"], "compounds": [], "reduplicated": [1], "oov_items": ["', True),
    ]:
        got = allowed_after(prefix)
        free = len(got) == total
        assert free == want_free, (
            f"前綴 {prefix!r}：預期{'全放行' if want_free else '受約束'}，"
            f"實得 {len(got)}/{total}")
    print("✓ v3 的 compounds／reduplicated 不會重新觸發約束（巢狀括號也安全）")


def allowed_after(prefix, cands=CANDS):
    tok = FakeTok(PIECES)
    cd._VOCAB_BY_FIRST = None                       # 每次重建索引
    fn = cd.constrained_prefix_fn(tok, 0, cands)
    ids = tok.encode(prefix)
    return {tok.pieces[i] for i in fn(0, ids)}


def texts(allowed):
    return sorted(allowed)


def test_candidate_constraint_still_applies():
    got = allowed_after('"sign_ids": ["TSL_')
    # 只該放行能接成候選的 token，不該整表放行
    assert len(got) < len(PIECES), f"候選約束失效，整表放行: {len(got)}"
    assert "二十" in got and "三十" in got, texts(got)
    print("✓ 候選約束照舊生效 →", texts(got))


def test_run_cap_blocks_degenerate_repeat():
    # 連續 6 次 TSL_二十（= MAX_RUN），第 7 次應被擋
    elems = '", "'.join(["TSL_二十"] * cd.MAX_RUN)
    prefix = f'"sign_ids": ["{elems}", "TSL_'
    got = allowed_after(prefix)
    assert "二十" not in got, f"連續 {cd.MAX_RUN} 次後仍放行 二十: {texts(got)}"
    assert "三十" in got, f"擋過頭了，合法候選也被拿掉: {texts(got)}"
    print(f"✓ 連續 {cd.MAX_RUN} 次後擋掉重複，仍留其他候選 →", texts(got))

    # 少一次（MAX_RUN-1）時還不該擋
    elems = '", "'.join(["TSL_二十"] * (cd.MAX_RUN - 1))
    got = allowed_after(f'"sign_ids": ["{elems}", "TSL_')
    assert "二十" in got, f"未到上限就擋: {texts(got)}"
    print(f"✓ 連續 {cd.MAX_RUN - 1} 次時不干預")


def test_run_cap_is_consecutive_not_total():
    # 交錯出現不算連續，合法（參考答案有「我/他/我/他/我」這種）
    elems = '", "'.join(["TSL_我", "TSL_你"] * 4)
    got = allowed_after(f'"sign_ids": ["{elems}", "TSL_')
    assert "我" in got, f"交錯重複被誤擋: {texts(got)}"
    print("✓ 交錯重複不受影響（重疊表複數／強調是合法的）")


def test_length_cap_closes_array():
    elems = '", "'.join(["TSL_我"] * cd.MAX_SIGNS)
    got = allowed_after(f'"sign_ids": ["{elems}"')      # 剛關掉引號，沒有尾逗號
    assert got and all(t.startswith("]") for t in got), texts(got)
    print(f"✓ 滿 {cd.MAX_SIGNS} 個元素且無尾逗號時只准收陣列 →", texts(got))


def test_length_cap_never_makes_trailing_comma():
    elems = '", "'.join(["TSL_我"] * cd.MAX_SIGNS)
    got = allowed_after(f'"sign_ids": ["{elems}", ')    # 逗號已經出去了
    assert len(got) == len(PIECES), "尾逗號狀態下強收陣列會生出破 JSON"
    print("✓ 已有尾逗號時整步放行（寧可漏擋，不可擋出破 JSON）")


def test_under_cap_untouched():
    elems = '", "'.join(["TSL_我"] * 3)
    got = allowed_after(f'"sign_ids": ["{elems}"')
    assert len(got) == len(PIECES), "未到長度上限就干預"
    print("✓ 未達長度上限時不干預")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{len(fns)} 項全過（MAX_RUN={cd.MAX_RUN} MAX_SIGNS={cd.MAX_SIGNS}）")
