#!/usr/bin/env python3
"""常用語句整句對照（scripts/phrasebook.py）。

守三件事：該命中的命中（請再說一次 → 請/再/說/一次，含標點與全半形差異）、
**不是整句的不命中**（請你再說一次 要交回模型）、演不出來的標準答案不收。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import phrasebook as pb


def _fake():
    index = {"請": "TSL_請", "再": "TSL_再", "說": "TSL_說", "一次": "TSL_一次",
             "你": "TSL_你", "住": "TSL_住", "宜蘭": "TSL_宜蘭", "好": "TSL_好"}
    eligible = set(index.values()) - {"TSL_宜蘭"}     # 宜蘭 影片不合格
    rows = [
        {"id": "S07", "chinese": "請再說一次", "gloss": ["請", "再", "說", "一次"]},
        {"id": "S24", "chinese": "我叫＿", "gloss": ["我", "名字", "＿"], "is_template": True},
        {"id": "P03", "chinese": "你住在宜蘭嗎？", "gloss": ["你", "宜蘭", "住"]},
        {"id": "X01", "chinese": "你好", "gloss": ["你", "好"]},
        {"id": "X02", "chinese": "你好！", "gloss": ["好"]},          # 同句不同答案
    ]
    return pb.build(rows, index.get, eligible)


def test_hits_with_punct_and_width():
    table, _ = _fake()
    for text in ("請再說一次", "請再說一次。", "請再說一次？", " 請再說一次 ", "請再說一次!",
                 "請　再說一次！"):
        hit = pb.lookup(table, text)
        assert hit and hit["sign_ids"] == ["TSL_請", "TSL_再", "TSL_說", "TSL_一次"], text
    print("✓ 請再說一次 含標點／全半形／空白皆命中")


def test_not_whole_sentence_misses():
    table, _ = _fake()
    for text in ("請你再說一次", "再說一次", "請再說一次好嗎", "麻煩再說一次"):
        assert pb.lookup(table, text) is None, text
    print("✓ 非整句一律不命中")


def test_skips():
    table, skipped = _fake()
    why = dict(skipped)
    assert "S24" in why and "模板" in why["S24"]
    assert "P03" in why and "宜蘭" in why["P03"]
    assert pb.lookup(table, "你住在宜蘭嗎") is None
    assert pb.lookup(table, "你好") is None, "同句兩個答案應兩個都不採"
    print("✓ 模板句、演不出來、同句歧義都不收")


def test_real_data():
    """用 repo 的 tsl_sentences.jsonl 與真的候選檢索器：S07 必須收進表。"""
    try:
        from sign_candidates import CandidateRetriever
        retr = CandidateRetriever(use_examples=False)
    except (FileNotFoundError, OSError) as e:
        print(f"- 略過真實資料檢查（{e}）")
        return
    table, skipped = pb.load(retr)
    hit = pb.lookup(table, "請再說一次")
    assert hit and hit["id"] == "S07", "S07 請再說一次 沒進對照表"
    assert hit["sign_ids"] == ["TSL_請", "TSL_再", "TSL_說", "TSL_一次"], hit
    eligible = {r["sign_id"] for r in retr.rows}
    for v in table.values():
        assert all(i in eligible for i in v["sign_ids"]), v
    print(f"✓ 真實資料：對照 {len(table)} 句，略過 {[i for i, _ in skipped]}")


def main():
    test_hits_with_punct_and_width()
    test_not_whole_sentence_misses()
    test_skips()
    test_real_data()


if __name__ == "__main__":
    main()
