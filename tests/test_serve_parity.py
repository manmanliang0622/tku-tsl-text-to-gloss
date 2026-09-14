#!/usr/bin/env python3
"""守住「約束解碼只有一份實作」。

2026-08-31 之前，serve_model.py 內嵌了一份與 scripts/constrained_decode.py
逐行相同的狀態機，這個測試負責比對兩份不要漂移。現在 serve_model.py 直接
import 那個模組（教授審查意見 4.3），repo 這一側已經沒有第二份可以漂移，
測試的職責因此換成兩件事：

  1. **回歸守衛**：repo 的 serve_model.py 不得再出現內嵌副本，而且必須真的
     從 constrained_decode import。有人日後為了「少一個相依」把它貼回去，
     這裡會擋下來。
  2. **部署副本比對**：學校主機上那份在重新部署之前仍可能是舊的內嵌版，
     所以保留原本的逐前綴行為比對——存在才比，不存在就略過。
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import constrained_decode as cd
from test_constrained_decode import CANDS, PIECES, FakeTok

REPO_SERVE = os.path.join(HERE, "..", "scripts", "serve_model.py")
# 部署那份只在學校主機上才有。2026-08-30 修正：原本只指 ~/0821，但線上服務
# 8/28 起是從 ~/0821_bundle 起的，等於在守一個沒人跑的目錄。
DEPLOYED_SERVE = os.environ.get(
    "SERVE_MODEL_PATH", "/home/b310ai/0821_bundle/model_service/scripts/serve_model.py")

# 內嵌副本的特徵：模組層自己定義狀態機，而不是 import 進來。
_INLINE_MARKERS = ("def _constrained_prefix_fn(", "def _vocab_index(")


def test_repo_imports_shared_module():
    """repo 的 serve_model.py 必須 import，不得內嵌。"""
    src = open(REPO_SERVE, encoding="utf-8").read()
    assert "from constrained_decode import constrained_prefix_fn" in src, \
        "serve_model.py 沒有從 constrained_decode import 約束解碼"
    for marker in _INLINE_MARKERS:
        assert marker not in src, (
            f"serve_model.py 又出現內嵌副本（{marker}）。"
            "約束解碼只能有一份實作，請改回 import constrained_decode。")
    print("✓ repo serve_model.py 使用共用模組，無內嵌副本")


def load_serve_fn(path):
    """從內嵌版的原始碼裡把狀態機挖出來，在 stub 命名空間 exec（避開 torch）。"""
    src = open(path, encoding="utf-8").read()
    start = src.index("MAX_RUN = 6")
    end = src.index("def translate_script(")
    ns = {"re": re, "os": os}
    exec(compile(src[start:end], path, "exec"), ns)
    return ns


def check_deployed(label, path):
    src = open(path, encoding="utf-8").read()
    if "from constrained_decode import constrained_prefix_fn" in src:
        print(f"── {label}：{path}\n✓ 已改用共用模組，無需比對")
        return

    ns = load_serve_fn(path)
    for name in ("MAX_RUN", "MAX_SIGNS"):
        assert ns[name] == getattr(cd, name), \
            f"[{label}] {name} 不一致: serve={ns[name]} module={getattr(cd, name)}"
    print(f"── {label}：{path}（仍是內嵌版，逐前綴比對）")
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


def test_deployed_copy():
    if os.path.exists(DEPLOYED_SERVE):
        check_deployed("部署副本", DEPLOYED_SERVE)
    else:
        # 在沒有部署環境的機器上跑（例如開發用的 Mac）。
        print(f"── 部署副本：{DEPLOYED_SERVE} 不存在，略過")
        print("   （要比對線上那份，請在學校主機上跑，或用 SERVE_MODEL_PATH 指定路徑）")


def main():
    test_repo_imports_shared_module()
    test_deployed_copy()


if __name__ == "__main__":
    main()
