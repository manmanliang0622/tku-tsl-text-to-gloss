#!/usr/bin/env python3
"""候選器的兩個契約：訓練／上線參數一致，以及 train 候選的 cross-fitting。

對應教授審查意見 2.1（training-serving skew）與 2.2（標籤洩漏）。

2.1 的實況：v17 的訓練資料用 n_sem=8 建，線上服務沒載向量模型、實際走
n_sem=0，每句約 8 個候選相異，參考詞可及率由 100% 掉到 99.0/98.5%。
這件事當時只寫在模型卡的「已知限制」，沒有任何機制擋住。

2.2 的實況：對齊表與核心詞是用**完整 train** 建的，替 train 句產生候選時
表裡已含該句自己的答案。`exclude_id` 只擋掉例句遷移，擋不到這兩張統計表。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))

from build_script_dataset import assign_folds


def test_folds_keep_groups_intact():
    """同一個 group 的所有列必須落在同一個 fold。

    這是 2.2 的核心要求：長度平衡會把同一句複製 2–4 份，按列分會讓副本
    散到不同 fold，候選器仍看得到那句的答案。
    """
    rows = [{"group": f"g{i % 7}"} for i in range(100)]
    fold_of = assign_folds(rows, 5)
    seen = {}
    for r, f in zip(rows, fold_of):
        g = r["group"]
        if g in seen:
            assert seen[g] == f, f"group {g} 被拆到 fold {seen[g]} 與 {f}"
        seen[g] = f
    print(f"✓ {len(seen)} 個 group 各自完整落在單一 fold")


def test_folds_are_deterministic():
    """同樣的輸入永遠得到同樣的切法——否則重建資料不可重現。"""
    rows = [{"group": f"g{i % 13}"} for i in range(200)]
    a = assign_folds(rows, 5)
    b = assign_folds(list(rows), 5)
    assert a == b, "同樣輸入得到不同 fold 分配"
    print("✓ fold 分配是決定性的")


def test_folds_are_balanced():
    """貪心裝箱應該讓各 fold 的列數接近。"""
    rows = [{"group": f"g{i % 40}"} for i in range(1000)]
    fold_of = assign_folds(rows, 5)
    sizes = [fold_of.count(f) for f in range(5)]
    assert max(sizes) - min(sizes) <= max(sizes) * 0.2, f"fold 大小失衡：{sizes}"
    print(f"✓ fold 大小均衡：{sizes}")


def test_rows_without_group_are_separated():
    """沒有 group 的列各自成組，不會被綁在一起。"""
    rows = [{"group": None} for _ in range(10)]
    fold_of = assign_folds(rows, 5)
    assert len(set(fold_of)) > 1, "無 group 的列全被塞進同一個 fold"
    print("✓ 無 group 的列各自成組")


def _retriever():
    from sign_candidates import CandidateRetriever
    return CandidateRetriever()


def test_config_reports_every_knob():
    """config() 必須涵蓋所有會影響候選分布的參數。

    漏掉一個，那個參數就能在訓練與上線之間悄悄分歧——正是 n_sem 出事的方式。
    """
    retr = _retriever()
    cfg = retr.config()
    for key in retr.CONFIG_KEYS:
        assert key in cfg, f"config() 沒有回報 {key}"
    # n_sem 是踩過的那個坑，額外釘住
    assert "n_sem" in cfg and "semantic_loaded" in cfg
    print(f"✓ config() 回報全部 {len(retr.CONFIG_KEYS)} 個參數")


def test_config_overrides_take_effect():
    retr = _retriever()
    assert retr.config(k=40)["k"] == 40
    assert retr.config(n_sem=8)["n_sem"] == 8
    # 不在 CONFIG_KEYS 裡的東西不該被塞進去
    assert "bogus" not in retr.config(bogus=1)
    print("✓ config() 的覆寫只接受已知參數")


def test_semantic_channel_defaults_off():
    """預設必須是 n_sem=0——那是線上服務唯一跑得起來的組態。

    語義通道要載向量模型，0821_bundle 的環境載不起來。預設開著就會讓人
    不知不覺地建出一份上線重現不了的訓練資料。
    """
    retr = _retriever()
    cfg = retr.config()
    assert cfg["n_sem"] == 0, f"n_sem 預設應為 0，實得 {cfg['n_sem']}"
    assert cfg["semantic_loaded"] is False
    print("✓ 語義通道預設關閉（與線上服務一致）")


def main():
    test_folds_keep_groups_intact()
    test_folds_are_deterministic()
    test_folds_are_balanced()
    test_rows_without_group_are_separated()
    test_config_reports_every_knob()
    test_config_overrides_take_effect()
    test_semantic_channel_defaults_off()
    print("\n候選器契約檢查全過")


if __name__ == "__main__":
    main()
