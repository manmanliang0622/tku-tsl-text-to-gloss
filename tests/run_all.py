#!/usr/bin/env python3
"""跑完整測試套件。零第三方相依——CI 不必裝任何東西。

  python3 tests/run_all.py

各支負責的事：
  test_constrained_decode   約束解碼狀態機（含 MAX_RUN／MAX_SIGNS 退化守衛）
  test_serve_parity         約束解碼只有一份實作，服務端沒有內嵌副本
  test_metrics              BLEU／ROUGE-L／EM
  test_eval_script_format   評分管線能重現既有的 *_scriptmetrics.json
  test_split_normalization  切分的表面形式正規化與去洩漏
  test_inventory_legacy_id  legacy_sign_id 不被重建覆寫（追溯欄位的冪等性）
  test_comitative           伴隨句雙數收攏規則（重點在不該補的一個都沒補）
  test_bundle_deps          serve_model 的部署相依宣告與程式碼一致
  test_clause_breaks        子句邊界的定義、對齊與失敗退路
  test_gloss_notation       重複貌 ++ 與複合 + 的解析
  test_candidate_parity     候選參數的訓練／上線一致與 cross-fitting 分組
"""
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))

# 有 main() 的走子行程（它們用 print + assert，不是 unittest）
SCRIPT_TESTS = [
    "test_constrained_decode.py",
    "test_serve_parity.py",
    "test_eval_script_format.py",
    "test_split_normalization.py",
    "test_inventory_legacy_id.py",
    "test_comitative.py",
    "test_bundle_deps.py",
    "test_clause_breaks.py",
    "test_gloss_notation.py",
    "test_candidate_parity.py",
]
# unittest 形式的
UNITTEST_MODULES = ["test_metrics"]


def main():
    failed = []
    for name in SCRIPT_TESTS:
        print(f"\n{'=' * 62}\n== {name}\n{'=' * 62}")
        rc = subprocess.run([sys.executable, os.path.join(HERE, name)]).returncode
        if rc != 0:
            failed.append(name)

    print(f"\n{'=' * 62}\n== unittest: {', '.join(UNITTEST_MODULES)}\n{'=' * 62}")
    sys.path.insert(0, HERE)
    suite = unittest.defaultTestLoader.loadTestsFromNames(UNITTEST_MODULES)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        failed.append("test_metrics")

    print(f"\n{'=' * 62}")
    if failed:
        print(f"✗ {len(failed)} 支失敗：{', '.join(failed)}")
        return 1
    print(f"✓ 全部 {len(SCRIPT_TESTS) + len(UNITTEST_MODULES)} 支通過")
    return 0


if __name__ == "__main__":
    sys.exit(main())
