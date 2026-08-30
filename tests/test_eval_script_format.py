#!/usr/bin/env python3
"""回歸測試：eval_script_format 必須重現既有的 *_scriptmetrics.json。

背景：這支評分腳本只 commit 過 2026-08-20 那一版（5362e3d，在未合併的
origin/video-coverage-eval 上），後來加的 Full_reference 與
NeedsReview_calibrated 只存在於跑評估那台 VM 的工作目錄，導致 v14–v17 的
指標一度無法從頭重現。2026-08-27 把兩塊補回來，這支測試就是驗收：
拿 results/v17cd_* 重跑，與既有檔案逐欄比對。

既有檔案是用舊門檻 0.095349 算的，所以測試也用舊門檻——這裡驗的是
「管線忠實」，不是「門檻該取多少」（後者見 scripts/nr_threshold.py）。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, "scripts"))

import eval_script_format as esf

LEGACY_THRESHOLD = 0.095349
CASES = ["v17cd_test", "v17cd_test_corpus", "v17cd_test_textbook", "v17cd_dev"]


def load_rows(stem):
    path = os.path.join(BASE, "results", f"{stem}.jsonl")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def compare(stem):
    rows = load_rows(stem)
    split = esf.guess_split(esf.Path(os.path.join(BASE, "results", f"{stem}.jsonl")))
    assert split and split.exists(), f"{stem}: 推斷不到 split（{split}）"
    got = esf.evaluate(rows, threshold=LEGACY_THRESHOLD,
                       full_ref=esf.load_full_reference(split))

    with open(os.path.join(BASE, "results", f"{stem}_scriptmetrics.json"),
              encoding="utf-8") as fh:
        want = json.load(fh)

    diffs = []
    for key in sorted(set(want) | set(got)):
        if key == "_ref_scope":                 # 純說明字串，不比對
            continue
        a, b = want.get(key), got.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            for sub in sorted(set(a) | set(b)):
                if sub == "decision":
                    continue
                if a.get(sub) != b.get(sub):
                    diffs.append(f"{key}.{sub}: 既有={a.get(sub)} 重算={b.get(sub)}")
        elif a != b:
            diffs.append(f"{key}: 既有={a} 重算={b}")
    return got, want, diffs


def main():
    failed = 0
    for stem in CASES:
        got, want, diffs = compare(stem)
        n_fields = sum(1 + (len(v) if isinstance(v, dict) else 0)
                       for v in want.values())
        if diffs:
            failed += 1
            print(f"✗ {stem}  {len(diffs)} 欄不符")
            for d in diffs:
                print(f"    {d}")
        else:
            print(f"✓ {stem:22} {n_fields:3} 欄全部相同  "
                  f"(BLEU {got['BLEU-4']} / Full {got['Full_reference']['BLEU-4']} / "
                  f"dropped {got['Full_reference']['ref_tokens_dropped%']}%)")
    print()
    if failed:
        print(f"{failed}/{len(CASES)} 個資料集不符——評分管線與既有指標不一致")
        return 1
    print(f"{len(CASES)}/{len(CASES)} 個資料集逐欄相同——評分管線已可從頭重現")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
