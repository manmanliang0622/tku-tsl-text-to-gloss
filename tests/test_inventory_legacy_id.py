#!/usr/bin/env python3
"""`legacy_sign_id` 不得在重建總表時被自我覆寫。

2026-08-31 抓到的潛伏 bug：`build_sign_inventory.load_existing()` 原本回傳
`row["sign_id"]`——也就是**現行**的語義 ID。於是每重建一次總表，
legacy_sign_id 就被現行 ID 蓋掉一次：

    TSL_訪談_2 的 legacy_sign_id: TSL_00001 → TSL_訪談_2

這個欄位存在的唯一目的是把 v11 之前用流水號寫成的訓練資料與下游腳本
對回來，自我覆寫等於把它變成廢欄位。實測一次重建就毀掉 16,628 筆。

正確語意（本測試釘住的）：
  1. 已記錄過 legacy 的，沿用原值——重建幾次都一樣（冪等）。
  2. 沒記錄過的維持沒記錄，**除非 ID 方案真的換了**（serial → semantic），
     那才是這個欄位設計上要接住的情境。
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))

import build_sign_inventory as bsi
from pathlib import Path

ROWS = [
    # 有原始流水號的（v11 之前發出去的 ID）
    {"gloss": "訪談", "sign_id": "TSL_訪談_2", "legacy_sign_id": "TSL_00001"},
    {"gloss": "許久", "sign_id": "TSL_許久_2", "legacy_sign_id": "TSL_00002"},
    # 從來沒有流水號的（語義方案之後才進動作庫的詞）
    {"gloss": "冷氣吹", "sign_id": "TSL_冷氣吹", "legacy_sign_id": None},
]


def _fixture(tmp, scheme):
    inv = Path(tmp) / "sign_inventory.jsonl"
    inv.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ROWS),
                   encoding="utf-8")
    stats = Path(tmp) / "inventory_stats.json"
    stats.write_text(json.dumps({"id_scheme": scheme}), encoding="utf-8")
    return inv, stats


def _run(scheme_prev, scheme_now):
    with tempfile.TemporaryDirectory() as tmp:
        inv, stats = _fixture(tmp, scheme_prev)
        old_inv, old_stats = bsi.INVENTORY, bsi.STATS
        try:
            bsi.INVENTORY, bsi.STATS = inv, stats
            return bsi.load_existing(scheme_now)
        finally:
            bsi.INVENTORY, bsi.STATS = old_inv, old_stats


def test_preserves_recorded_legacy():
    """方案沒換：已記錄的原值必須原封不動。"""
    m = _run("semantic", "semantic")
    assert m.get("訪談") == "TSL_00001", f"訪談 的 legacy 被改成 {m.get('訪談')!r}"
    assert m.get("許久") == "TSL_00002", f"許久 的 legacy 被改成 {m.get('許久')!r}"
    print("✓ 已記錄的 legacy_sign_id 沿用原值（TSL_00001／TSL_00002）")


def test_no_self_overwrite():
    """方案沒換：從未記錄過的不該被填成它自己的現行 ID。"""
    m = _run("semantic", "semantic")
    assert "冷氣吹" not in m, (
        f"冷氣吹 原本沒有 legacy，卻被填成 {m.get('冷氣吹')!r}——這就是自我覆寫")
    print("✓ 沒記錄過的維持沒記錄，不會被自己的 sign_id 填滿")


def test_scheme_change_captures_old_id():
    """方案換了：舊 sign_id 本身就是 legacy，這時才該落回。"""
    m = _run("serial", "semantic")
    assert m.get("冷氣吹") == "TSL_冷氣吹", (
        f"方案由 serial 換成 semantic 時，舊 ID 應被記為 legacy，實得 {m.get('冷氣吹')!r}")
    # 已有原值的仍然優先
    assert m.get("訪談") == "TSL_00001"
    print("✓ ID 方案變更時才落回舊 sign_id，且不覆蓋已有原值")


def main():
    test_preserves_recorded_legacy()
    test_no_self_overwrite()
    test_scheme_change_captures_old_id()
    print("\nlegacy_sign_id 語意檢查全過")


if __name__ == "__main__":
    main()
