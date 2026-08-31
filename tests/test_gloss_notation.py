#!/usr/bin/env python3
"""Gloss 標記的解析：重複貌 ++ 與複合 +（教授審查意見 3.1）。

修正前 `norm()` 把兩種標記都當雜訊丟掉：

    買++    →  買       重複貌不見了
    樹+見   →  樹       「見」整個消失

實測 train 有 478 列帶 `++`、447 列帶複合 `+`，第二段被直接刪除 664 次。
刪掉的多是 手機(58)／這(51)／杯子(26)／腳(20) 這類分類詞或指涉對象——
不是雜訊，是句子的一部分。

兩個查證結論決定了設計，寫在這裡免得日後被「簡化」掉：

1. **複合是兩個手語不是一個**。384 個複合 token 裡，整串本身是動作庫鍵的
   有 **0 個**；拆開後每段都查得到的有 275 個（71.6%）。所以攤平成連續
   sign_id 是對的。
2. **`++` 是重複貌不是次數**。語料庫標記慣例見 scrape_tslcorpus_full.clean_token。
   345 個 `++`、1 個 `+++`，沒有任何資訊指出重複幾次，所以回布林而不是
   repeat 整數——編一個數字等於替標註者宣稱他沒寫的事。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))

from eval_video_coverage import parse_token, norm


def test_reduplication():
    assert parse_token("買++") == (["買"], True)
    assert parse_token("吃++") == (["吃"], True)
    assert parse_token("買") == (["買"], False)
    # +++ 也是重複貌，不代表「更多次」——語料只有 1 個
    assert parse_token("走+++") == (["走"], True)
    print("✓ 重複貌只回布林，不編造次數")


def test_compound_is_flattened():
    assert parse_token("樹+見") == (["樹", "見"], False)
    assert parse_token("腳踝+這") == (["腳踝", "這"], False)
    # 三段的也有（實測 14 個）
    assert parse_token("船+車+登船") == (["船", "車", "登船"], False)
    print("✓ 複合攤平成連續詞段，第二段不再被刪掉")


def test_compound_with_reduplication():
    """兩種標記並存，實測 8 個。"""
    assert parse_token("樹+開花++") == (["樹", "開花"], True)
    assert parse_token("船+車+登船++") == (["船", "車", "登船"], True)
    print("✓ 複合＋重複貌並存時兩者都保留")


def test_parenthetical_is_stripped_before_splitting():
    """括號要先剝再拆 +，順序反了 `(包+包)交換` 會被拆壞。"""
    assert parse_token("(包+包)交換") == (["交換"], False)
    assert parse_token("成(成人)") == (["成"], False)
    assert parse_token("告訴(他)") == (["告訴"], False)
    # 整串都在括號裡的是註記本身，不能整個刪掉
    assert parse_token("(到處)") == (["到處"], False)
    print("✓ 括號註記先剝離，括號內的 + 不會被誤拆")


def test_norm_still_returns_first_segment():
    """norm() 的語意刻意不動：它回答的是「這個詞查不查得到影片」，
    只需要第一段。結構化解析是 parse_token 的事，兩者用途不同。"""
    assert norm("樹+見") == "樹"
    assert norm("買++") == "買"
    print("✓ norm() 行為未變（查詢用），結構化交給 parse_token")


def test_empty_and_degenerate():
    assert parse_token("") == ([], False)
    assert parse_token("+") == ([], True)
    assert parse_token("++") == ([], True)
    print("✓ 空字串與退化輸入不會炸")


def main():
    test_reduplication()
    test_compound_is_flattened()
    test_compound_with_reduplication()
    test_parenthetical_is_stripped_before_splitting()
    test_norm_still_returns_first_segment()
    test_empty_and_degenerate()
    print("\nGloss 標記解析檢查全過")


if __name__ == "__main__":
    main()
