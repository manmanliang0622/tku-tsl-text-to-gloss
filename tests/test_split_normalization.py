#!/usr/bin/env python3
"""切分的表面形式正規化與去洩漏（教授審查意見 2.4）。

2026-08-31 之前 split_data.py 只比對**原始中文字串**，於是：

  - 核心 33 句有 3 句去標點後就在 train（我住在台北。／我知道／我不知道）
  - dev 與 train 有 6 句原字串完全相同——去重鍵是 (chinese, gloss_text)，
    同一句中文配不同 Gloss 就兩邊都留，而語料庫會把同一段話收在不同對話編號、
    辭典會把同一句例句掛在多個詞條底下

這支測試釘住兩件事：正規化函式本身的行為，以及**實際切分檔**沒有重疊。
後者是真正重要的那個——函式對了但沒接上去，等於沒修。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, "scripts"))

from split_data import normalize_text, normalize_gloss

SPLIT_DIR = os.path.join(BASE, "data", "splits")


def load(name):
    path = os.path.join(SPLIT_DIR, f"{name}.jsonl")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_normalize_text():
    cases = [
        # 標點：教授點名的那三句，去標點後必須與 train 那版視為同一句
        ("我住在台北。", "我住在台北"),
        ("我知道，", "我知道"),
        ("此外，", "此外"),
        # 全形／半形（NFKC）
        ("ＡＢＣ１２３", "ABC123"),
        ("我　知道", "我知道"),
        # 異體字：臺／台、妳／你、牠祂／他
        ("臺北", "台北"),
        ("妳好", "你好"),
        ("牠很可愛", "他很可愛"),
        # 括號與引號
        ("他說「好」！", "他說好"),
    ]
    for raw, want in cases:
        got = normalize_text(raw)
        assert got == want, f"normalize_text({raw!r}) = {got!r}，預期 {want!r}"
    print(f"✓ normalize_text {len(cases)} 組case 全對")


def test_normalize_gloss():
    assert normalize_gloss("我/臺北/住") == "我/台北/住"
    assert normalize_gloss(" 我 / 好 ") == "我/好"
    assert normalize_gloss("我//好") == "我/好"      # 空段落丟掉
    print("✓ normalize_gloss 行為正確")


def test_no_normalized_overlap():
    """實際切分檔：任一測試集／dev 的正規化中文都不得出現在 train。"""
    train = load("train")
    if train is None:
        print("⚠ data/splits/train.jsonl 不存在，略過切分檔檢查"
              "（先跑 scripts/split_data.py）")
        return
    train_norm = {normalize_text(e["chinese"]) for e in train}

    checked = 0
    for name in ("dev", "test", "test_corpus", "test_textbook", "test_papers"):
        rows = load(name)
        if not rows:
            continue
        overlap = sorted({normalize_text(e["chinese"]) for e in rows} & train_norm)
        assert not overlap, (
            f"{name} 有 {len(overlap)} 句正規化中文出現在 train，前 5 句：{overlap[:5]}")
        print(f"✓ {name:14} {len(rows):5} 句，與 train 正規化後零重疊")
        checked += 1
    assert checked, "一個切分檔都沒讀到"


def test_manifest_records_normalization():
    path = os.path.join(SPLIT_DIR, "manifest.json")
    if not os.path.exists(path):
        print("⚠ manifest.json 不存在，略過")
        return
    with open(path, encoding="utf-8") as fh:
        mf = json.load(fh)
    sn = mf.get("surface_normalization")
    assert sn, "manifest 沒有 surface_normalization——切分是舊版程式產生的，請重跑"
    assert sn["train_dev_normalized_chinese_overlap"] == 0, \
        f"manifest 記錄 train/dev 仍有 {sn['train_dev_normalized_chinese_overlap']} 句重疊"
    print(f"✓ manifest 記錄正規化：合併 {sn['text_clusters_merged']} 個 text cluster，"
          f"train/dev 重疊 0")


def main():
    test_normalize_text()
    test_normalize_gloss()
    test_no_normalized_overlap()
    test_manifest_records_normalization()
    print("\n切分正規化檢查全過")


if __name__ == "__main__":
    main()
