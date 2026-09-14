#!/usr/bin/env python3
"""算出 serve_model.py 部署時必須一起帶的本地模組，並與宣告清單對帳。

為什麼需要這支（2026-08-31）：serve_model 的部署相依一直只寫在註解裡
——「comitative.py 必須一起帶」「constrained_decode.py 必須一起帶」。
註解不會失敗。實際發生的事：commit 279d35c 一次加了兩個新相依
（constrained_decode 與 script_schema），註解只提到其中一個，數量還寫錯
（寫「第四個相依」，實際是第五、第六個）。那版部署到 0821_bundle 會啟動即死，
而且是 script_schema 先炸——它在模組層 import，連 try/except 都沒有。

作法：AST 解析 serve_model.py 及其相依，遞移收斂出所有 scripts/ 底下的模組，
與 serve_model.BUNDLE_MODULES 的宣告對帳。加了新 import 卻忘了更新宣告，
這支就會紅——tests/test_bundle_deps.py 把它掛進 CI。

**注意**：這支只保證「宣告與程式碼一致」，不保證「VM 上真的有這些檔」。
後者要在部署端驗，見 --deploy-check 印出的指令。

用法：
    python3 scripts/check_bundle_deps.py                # 對帳
    python3 scripts/check_bundle_deps.py --list         # 只印檔案清單
    python3 scripts/check_bundle_deps.py --deploy-check # 印部署端的驗證指令
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ENTRY = "serve_model"


def local_modules() -> set[str]:
    return {p.stem for p in SCRIPTS.glob("*.py")}


def direct_deps(module: str, local: set[str]) -> set[str]:
    """module 直接 import 的本地模組。函式內的延遲 import 也算——
    它照樣會在請求進來時炸，只是炸得比較晚、比較難查。"""
    tree = ast.parse((SCRIPTS / f"{module}.py").read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in local:
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            # level>0 是相對 import，本專案的 scripts/ 是扁平的，不會出現
            if node.level == 0 and node.module in local:
                found.add(node.module)
    return found


def transitive_deps(entry: str = ENTRY) -> set[str]:
    local = local_modules()
    seen: set[str] = set()
    stack = [entry]
    while stack:
        mod = stack.pop()
        if mod in seen:
            continue
        seen.add(mod)
        stack.extend(direct_deps(mod, local) - seen)
    seen.discard(entry)
    return seen


def declared() -> tuple[str, ...]:
    """讀 serve_model.BUNDLE_MODULES 的字面值。

    用 ast 而不是 import——import serve_model 會把 torch/transformers 一起拖進來。
    """
    tree = ast.parse((SCRIPTS / f"{ENTRY}.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "BUNDLE_MODULES" for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise SystemExit(f"{ENTRY}.py 找不到 BUNDLE_MODULES 宣告")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="只印必須部署的檔案清單")
    ap.add_argument("--deploy-check", action="store_true",
                    help="印出在部署端驗證檔案是否到齊的指令")
    args = ap.parse_args()

    need = transitive_deps()
    files = sorted(f"{m}.py" for m in need)

    if args.list:
        print("\n".join(files))
        return 0

    if args.deploy_check:
        names = " ".join(files)
        print("# 在部署端確認檔案到齊（缺任何一個，serve_model 啟動即死）")
        print(f"ssh tku-gpu 'cd ~/0821_bundle/model_service/scripts && "
              f"for f in {names} serve_model.py; do "
              f"[ -f \"$f\" ] || echo \"缺 $f\"; done; echo 檢查完畢'")
        return 0

    want = set(declared())
    missing = sorted(need - want)      # 程式有 import，宣告漏了
    extra = sorted(want - need)        # 宣告有，程式已不再 import

    print(f"{ENTRY}.py 的遞移本地相依（{len(need)} 個）：")
    for f in files:
        print(f"  {f}")

    if missing or extra:
        print()
        if missing:
            print(f"✗ BUNDLE_MODULES 漏列：{missing}")
            print("  程式 import 了但宣告沒寫——部署時會漏帶，服務啟動即死。")
        if extra:
            print(f"✗ BUNDLE_MODULES 多列：{extra}")
            print("  已經不再 import，留著會讓部署清單失真。")
        print(f"\n請更新 {ENTRY}.py 的 BUNDLE_MODULES。")
        return 1

    print(f"\n✓ 與 {ENTRY}.BUNDLE_MODULES 宣告一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
