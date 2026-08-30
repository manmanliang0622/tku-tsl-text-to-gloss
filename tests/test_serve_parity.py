#!/usr/bin/env python3
"""確認 serve_model.py 內嵌的約束解碼與 scripts/constrained_decode.py 行為一致。

兩份副本是已知的維護風險（v17cd 報告已列）。在還沒合併成單一模組之前，
這個測試至少保證它們不會默默漂移：抽出 serve_model 的狀態機原始碼，
在 stub 命名空間裡 exec（避開 torch/transformers），再用同一組前綴逐一比對。
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import constrained_decode as cd
from test_constrained_decode import CANDS, PIECES, FakeTok

# 要比對的副本。repo 這份一定在（版控），部署那份只在學校主機上才有。
# 2026-08-30 修正：原本只指 ~/0821，但線上服務 8/28 起是從 ~/0821_bundle 起的，
# 等於在守一個沒人跑的目錄——部署副本漂移了也照樣顯示通過。
REPO_SERVE = os.path.join(HERE, "..", "scripts", "serve_model.py")
DEPLOYED_SERVE = os.environ.get(
    "SERVE_MODEL_PATH", "/home/b310ai/0821_bundle/model_service/scripts/serve_model.py")


def load_serve_fn(path):
    src = open(path, encoding="utf-8").read()
    start = src.index("MAX_RUN = 6")
    end = src.index("def translate_script(")
    ns = {"re": re, "os": os}
    exec(compile(src[start:end], path, "exec"), ns)
    return ns


def check(label, path):
    ns = load_serve_fn(path)
    for name in ("MAX_RUN", "MAX_SIGNS"):
        assert ns[name] == getattr(cd, name), \
            f"[{label}] {name} 不一致: serve={ns[name]} module={getattr(cd, name)}"
    print(f"── {label}：{path}")
    print(f"✓ 常數一致 MAX_RUN={ns['MAX_RUN']} MAX_SIGNS={ns['MAX_SIGNS']}")

    prefixes = [
        '"sign_ids": ["TSL_',
        '"sign_ids": ["TSL_我',
        '"sign_ids": ["TSL_我"',
        '"sign_ids": ["TSL_我", ',
        '"sign_ids": ["' + '", "'.join(["TSL_二十"] * cd.MAX_RUN) + '", "TSL_',
        '"sign_ids": ["' + '", "'.join(["TSL_二十"] * (cd.MAX_RUN - 1)) + '", "TSL_',
        '"sign_ids": ["' + '", "'.join(["TSL_我", "TSL_你"] * 4) + '", "TSL_',
        '"sign_ids": ["' + '", "'.join(["TSL_我"] * cd.MAX_SIGNS) + '"',
        '"sign_ids": ["' + '", "'.join(["TSL_我"] * cd.MAX_SIGNS) + '", ',
        '"sign_ids": ["' + '", "'.join(["TSL_我"] * 3) + '"',
    ]

    for i, prefix in enumerate(prefixes, 1):
        tok = FakeTok(PIECES)
        ids = tok.encode(prefix)

        cd._VOCAB_BY_FIRST = None
        mine = set(cd.constrained_prefix_fn(tok, 0, CANDS)(0, ids))

        ns["_VOCAB_BY_FIRST"] = None
        theirs = set(ns["_constrained_prefix_fn"](tok, 0, CANDS)(0, ids))

        shown = prefix if len(prefix) <= 46 else prefix[:20] + " … " + prefix[-20:]
        assert mine == theirs, (
            f"[{label}] 第 {i} 組不一致\n  前綴: {shown}\n"
            f"  只在 module: {sorted(tok.pieces[j] for j in mine - theirs)}\n"
            f"  只在 serve : {sorted(tok.pieces[j] for j in theirs - mine)}")
        print(f"✓ [{i:2}] 一致（{len(mine):3} 個 token 放行）  {shown}")

    print(f"{len(prefixes)} 組前綴逐一比對，行為相同\n")


def main():
    check("repo 副本", REPO_SERVE)
    if os.path.exists(DEPLOYED_SERVE):
        check("部署副本", DEPLOYED_SERVE)
    else:
        # 在沒有部署環境的機器上跑（例如開發用的 Mac）：repo 副本仍已比對過。
        print(f"── 部署副本：{DEPLOYED_SERVE} 不存在，略過")
        print("   （要比對線上那份，請在學校主機上跑，或用 SERVE_MODEL_PATH 指定路徑）")


if __name__ == "__main__":
    main()
