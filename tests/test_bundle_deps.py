#!/usr/bin/env python3
"""serve_model 的部署相依宣告必須與程式碼一致。

背景（2026-08-31）：serve_model 部署到 0821_bundle 時要哪些檔案，一直只寫在
註解裡。commit 279d35c 一次加了 constrained_decode 與 script_schema 兩個新
相依，註解只提到一個、數量還寫錯，那版部署上去會啟動即死——而且是
script_schema 先炸（模組層 import，連 try/except 都沒有）。

這支把「必須一起帶」從備忘錄變成契約：AST 算出遞移相依，與
serve_model.BUNDLE_MODULES 對帳。加了 import 忘了更新清單就會紅。

守不到的部分要講清楚：這只保證宣告與程式碼一致，**不保證 VM 上真的有這些檔**。
部署端的檢查指令見 `python3 scripts/check_bundle_deps.py --deploy-check`。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))

import check_bundle_deps as cbd


def test_declaration_matches_code():
    need = cbd.transitive_deps()
    want = set(cbd.declared())
    missing = sorted(need - want)
    extra = sorted(want - need)
    assert not missing, (
        f"serve_model.BUNDLE_MODULES 漏列 {missing}——程式 import 了但宣告沒寫，"
        "部署時會漏帶，服務啟動即死")
    assert not extra, (
        f"serve_model.BUNDLE_MODULES 多列 {extra}——已不再 import，"
        "留著會讓部署清單失真")
    print(f"✓ 部署相依宣告與程式碼一致（{len(need)} 個模組）")


def test_hard_dependencies_are_declared():
    """幾個「缺了就一定壞」的相依，明列出來當回歸錨點。

    刻意寫死而不是從程式推導：推導出來的清單如果整個算錯，上面那個測試
    仍然會通過（兩邊都錯得一樣）。這裡釘住幾個已知必要的。
    """
    want = set(cbd.declared())
    for mod in ("prompt_common", "sign_candidates", "constrained_decode",
                "script_schema", "train_qlora"):
        assert mod in want, f"{mod} 沒列進 BUNDLE_MODULES"
    print("✓ 五個已知必要相依都在宣告裡")


def test_declared_modules_exist():
    for mod in cbd.declared():
        path = os.path.join(cbd.SCRIPTS, f"{mod}.py")
        assert os.path.exists(path), f"BUNDLE_MODULES 列了 {mod} 但 {path} 不存在"
    print(f"✓ 宣告的 {len(cbd.declared())} 個模組檔案都在")


def main():
    test_declaration_matches_code()
    test_hard_dependencies_are_declared()
    test_declared_modules_exist()
    print("\n部署相依檢查全過")


if __name__ == "__main__":
    main()
