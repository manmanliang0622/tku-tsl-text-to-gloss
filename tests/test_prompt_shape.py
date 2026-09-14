#!/usr/bin/env python3
"""守住 user prompt 的形狀：訓練端與上線端只能經由 script_schema.user_prompt 組裝，
而且 context 鍵「有沒有」由開關決定，不是無條件帶著。

2026-09-08 的實況：v20ctx 為了「形狀固定」把 context 鍵改成永遠存在（空字串）。
無前文重建的 v21 資料因此多了一個 v19 沒有的鍵，dev／test 由逐位元相同變成
548 列全部不同；而線上 v19 是沒這個鍵訓的，改過的 serve_model 一部署就是
training-serving skew——正是教授審查意見 2.1 那一類。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import script_schema

REPO_SERVE = os.path.join(ROOT, "scripts", "serve_model.py")
REPO_BUILD = os.path.join(ROOT, "scripts", "build_script_dataset.py")


def test_context_key_is_conditional():
    """None → 沒有 context 鍵；str（含空字串）→ 有。鍵序固定。"""
    no_ctx = script_schema.user_prompt("你好", ["TSL_你", "TSL_好"])
    assert list(no_ctx) == ["text", "candidates"], list(no_ctx)
    with_empty = script_schema.user_prompt("你好", ["TSL_你"], context="")
    assert list(with_empty) == ["text", "context", "candidates"], list(with_empty)
    assert with_empty["context"] == ""
    with_ctx = script_schema.user_prompt("你好", ["TSL_你"], context="前一句")
    assert with_ctx["context"] == "前一句"
    # 沒前文時的序列化必須與 v19 資料逐字相同（那是線上模型看過的形狀）
    assert json.dumps(no_ctx, ensure_ascii=False) == \
        '{"text": "你好", "candidates": ["TSL_你", "TSL_好"]}'
    print("✓ context 鍵由 context=None/str 決定，鍵序 text→(context)→candidates")


def test_both_sides_use_shared_builder():
    """serve_model 與 build_script_dataset 都必須呼叫 user_prompt，不得手寫 dict。"""
    for path in (REPO_SERVE, REPO_BUILD):
        src = open(path, encoding="utf-8").read()
        assert "script_schema.user_prompt(" in src, f"{os.path.basename(path)} 沒有用 user_prompt"
        assert '"context": str(' not in src and "{\"text\": text," not in src, \
            f"{os.path.basename(path)} 又手寫了 user dict"
    print("✓ 兩端都經由 script_schema.user_prompt 組裝")


def test_serve_gate_is_symmetric():
    """訓練沒前文、服務多送一個 context 鍵，也要被 _verify_candidate_config 擋。"""
    src = open(REPO_SERVE, encoding="utf-8").read()
    assert "bool(trained_ctx) != bool(SERVE_CONTEXT_SENTENCES)" in src, \
        "閘門只擋單向（訓練有、服務沒有），訓練沒有、服務有的那一向沒擋"
    print("✓ context_sentences 閘門雙向")


def test_existing_datasets_match_rule():
    """有資料就對帳：無前文的資料沒有 context 鍵，有前文的才有。沒資料就略過。"""
    checked = 0
    for name, expect in (("v19", False), ("v21", False), ("v20ctx", True)):
        path = os.path.join(ROOT, "data", f"splits_script_{name}", "dev.jsonl")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            row = json.loads(f.readline())
        user = json.loads(row["messages"][1]["content"])
        assert ("context" in user) == expect, f"{name}: 鍵 {list(user)}"
        cfg_path = os.path.join(os.path.dirname(path), "candidate_config.json")
        if os.path.exists(cfg_path):
            cfg = json.load(open(cfg_path, encoding="utf-8"))
            assert bool(cfg.get("context_sentences", 0)) == expect, f"{name}: {cfg}"
        checked += 1
    print(f"✓ 既有資料 {checked} 份與規則一致" if checked else "－ 本機沒有資料，略過對帳")


if __name__ == "__main__":
    test_context_key_is_conditional()
    test_both_sides_use_shared_builder()
    test_serve_gate_is_symmetric()
    test_existing_datasets_match_rule()
