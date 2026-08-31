#!/usr/bin/env python3
"""子句邊界：定義、對齊與失敗時的退路（教授審查意見 3.2）。

修正前的實測狀態是「欄位完全是死的」——8,915 列訓練資料的 clause_breaks
**全部**是空陣列，模型只學會固定輸出 []。三個疊在一起的錯：

  1. 先用 `/` 切 gloss_text 再找 `//`，但 `//` 在切分那步就沒了。
  2. 只認獨立的 `++` token，實際資料是 `買++` 這種後綴。
  3. 把 `++` 與 `//` 當同一件事。它們來自不相交的兩個來源、意思也不同：
     `++` 只在文化部語料庫（303 筆）＝重複；`//` 只在中正辭典例句（40 筆）
     ＝子句邊界。兩者從未共存於同一筆。

修正後的來源是上游本來就有的 `clauses` 欄位，但只採 **token 計數**、
不採字串——人工校訂改過 gloss_text 卻沒同步 clauses，10 筆已經對不起來。

這支釘住的重點：
  - 索引空間是 **sign_ids**，不是 gloss token（OOV 的詞不會進 sign_ids）
  - 對不齊時退回空陣列，不猜
  - `++` 永遠不是子句邊界
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))

from build_script_dataset import clause_breaks


def test_professor_cases():
    """審查意見裡點名的兩個實測案例。"""
    # 我/好//你/好 —— 上游 clauses 已切好，邊界應落在第 2 個 sign
    breaks, reason = clause_breaks(["我/好", "你/好"], ["我", "好", "你", "好"],
                                   [0, 1, 2, 3])
    assert (breaks, reason) == ([2], "ok"), f"得到 {breaks} / {reason}"

    # 我/買++/東西 —— ++ 是重複記號，不是子句邊界；語料庫也沒有 clauses
    breaks, reason = clause_breaks(None, ["我", "買++", "東西"], [0, 1, 2])
    assert (breaks, reason) == ([], "no_annotation"), f"得到 {breaks} / {reason}"
    print("✓ 審查意見的兩個案例都正確")


def test_plus_plus_is_never_a_boundary():
    """就算 ++ 出現在有 clauses 的句子裡，也只由 clauses 決定邊界。"""
    breaks, _ = clause_breaks(["我/買++", "東西/好"], ["我", "買++", "東西", "好"],
                              [0, 1, 2, 3])
    assert breaks == [2], f"++ 不該自己製造邊界，得到 {breaks}"
    print("✓ ++ 不會被當成子句邊界")


def test_single_clause_has_no_break():
    for clauses in (None, [], ["我/好/你"]):
        breaks, reason = clause_breaks(clauses, ["我", "好", "你"], [0, 1, 2])
        assert breaks == [], f"{clauses!r} 不該有邊界，得到 {breaks}"
        assert reason == "no_annotation"
    print("✓ 單子句與無標註都回空陣列")


def test_index_space_is_sign_ids_not_gloss_tokens():
    """OOV 的詞不進 sign_ids，邊界要跟著往前挪。

    gloss: 我 / 缺詞 / 好 // 你 / 好      （4 個進 sign_ids，第 2 個是 OOV）
    sign_ids:  [我, 好, 你, 好]
    邊界在 gloss 索引 3，但 sign_ids 索引應該是 2。
    """
    breaks, reason = clause_breaks(["我/缺詞/好", "你/好"],
                                   ["我", "缺詞", "好", "你", "好"],
                                   [0, None, 1, 2, 3])
    assert (breaks, reason) == ([2], "ok"), f"得到 {breaks} / {reason}"
    print("✓ 邊界索引在 sign_ids 空間，OOV 已扣掉")


def test_unaligned_falls_back_to_empty():
    """clauses 的 token 總數與 gloss_text 對不上時不猜。

    真實情境：2026-08-21 人工校訂把 `蟑螂媽媽` 拆成 `蟑螂/媽媽`，
    但 clauses 欄位沒跟著改。
    """
    breaks, reason = clause_breaks(["那/蟑螂媽媽", "那/怕"],
                                   ["那", "蟑螂", "媽媽", "那", "怕"],
                                   [0, 1, 2, 3, 4])
    assert (breaks, reason) == ([], "unaligned"), f"得到 {breaks} / {reason}"
    print("✓ token 數對不上時退回空陣列並標 unaligned")


def test_boundary_at_edges_is_dropped():
    """整個子句被 OOV 吃光時，邊界會落到頭或尾——那切不出兩段，要丟掉。"""
    # 第一個子句全部 OOV → 邊界落在 0
    breaks, reason = clause_breaks(["缺一/缺二", "你/好"],
                                   ["缺一", "缺二", "你", "好"],
                                   [None, None, 0, 1])
    assert (breaks, reason) == ([], "collapsed"), f"得到 {breaks} / {reason}"
    # 第二個子句全部 OOV → 邊界落在尾端
    breaks, reason = clause_breaks(["我/好", "缺一/缺二"],
                                   ["我", "好", "缺一", "缺二"],
                                   [0, 1, None, None])
    assert (breaks, reason) == ([], "collapsed"), f"得到 {breaks} / {reason}"
    print("✓ 落在頭尾的邊界會被丟掉並標 collapsed")


def test_three_clauses_and_dedupe():
    breaks, reason = clause_breaks(["我/好", "你/好", "他/好"],
                                   ["我", "好", "你", "好", "他", "好"],
                                   [0, 1, 2, 3, 4, 5])
    assert (breaks, reason) == ([2, 4], "ok"), f"得到 {breaks} / {reason}"
    # 中間子句整段掉光 → 兩個邊界疊在同一個位置，去重後剩一個
    breaks, reason = clause_breaks(["我/好", "缺一/缺二", "他/好"],
                                   ["我", "好", "缺一", "缺二", "他", "好"],
                                   [0, 1, None, None, 2, 3])
    assert (breaks, reason) == ([2], "ok"), f"得到 {breaks} / {reason}"
    print("✓ 三子句正確，且疊在一起的邊界會去重")


def test_breaks_are_valid_slice_points():
    """產出的邊界必須能真的把 sign_ids 切成非空的幾段。"""
    n = 6
    breaks, _ = clause_breaks(["a/b", "c/d", "e/f"], list("abcdef"), list(range(n)))
    prev, segs = 0, []
    for b in breaks + [n]:
        segs.append((prev, b))
        prev = b
    assert all(a < z for a, z in segs), f"切出空段：{segs}"
    assert breaks == sorted(set(breaks)), "邊界必須嚴格遞增且不重複"
    print("✓ 邊界都是合法的切點（嚴格遞增、切出的每段非空）")


def main():
    test_professor_cases()
    test_plus_plus_is_never_a_boundary()
    test_single_clause_has_no_break()
    test_index_space_is_sign_ids_not_gloss_tokens()
    test_unaligned_falls_back_to_empty()
    test_boundary_at_edges_is_dropped()
    test_three_clauses_and_dedupe()
    test_breaks_are_valid_slice_points()
    print("\n子句邊界檢查全過")


if __name__ == "__main__":
    main()
