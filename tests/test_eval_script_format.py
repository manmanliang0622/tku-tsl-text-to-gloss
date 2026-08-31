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
# 舊鍵名 → 現行鍵名
NEW_KEYS = ("ValidSignID%", "AssetAvailable%", "QualityPlayable%",
            "QualityDetail", "CompositionSuccess%")
LEGACY_KEYS = {
    "NeedsReview": "CandidateCoverageRisk",
    "NeedsReview_calibrated": "CandidateCoverageRisk_calibrated",
}
CASES = ["v17cd_test", "v17cd_test_corpus", "v17cd_test_textbook", "v17cd_dev"]


def load_rows(stem):
    path = os.path.join(BASE, "results", f"{stem}.jsonl")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# v17／v17cd 是在 2026-08-31 修正切分**之前**評估的。三個測試集修正後位元
# 完全相同，但 dev 變了（663→548），所以歷史 dev 數字只能對舊 dev 重算——
# 那份凍結在 data/splits_v17/（見該目錄 README）。
LEGACY_SPLIT_DIR = os.path.join(BASE, "data", "splits_v17")


def resolve_split(stem):
    """找這個結果檔對應的 split 檔，優先用凍結的 v17 那份。

    不能只靠 esf.guess_split()：它是列舉 data/splits/*.jsonl 來比對名字的，
    而那些檔在 .gitignore 裡（可由 split_data.py 再生），所以乾淨 clone 上
    整個目錄沒有 jsonl，guess_split 一律回 None——四個案例會全部誤判成缺料。
    這裡改成直接用已知的 split 名字比對 stem，兩個目錄都找。
    """
    for d in (LEGACY_SPLIT_DIR, os.path.join(BASE, "data", "splits")):
        if not os.path.isdir(d):
            continue
        names = sorted((n[:-6] for n in os.listdir(d) if n.endswith(".jsonl")),
                       key=len, reverse=True)     # test_corpus 要贏 test
        for name in names:
            if stem == name or stem.endswith("_" + name):
                return esf.Path(os.path.join(d, f"{name}.jsonl"))
    return None


def compare(stem):
    rows = load_rows(stem)
    split = resolve_split(stem)
    assert split and split.exists(), f"{stem}: 推斷不到 split（{split}）"
    got = esf.evaluate(rows, threshold=LEGACY_THRESHOLD,
                       full_ref=esf.load_full_reference(split))

    with open(os.path.join(BASE, "results", f"{stem}_scriptmetrics.json"),
              encoding="utf-8") as fh:
        want = json.load(fh)

    # 2026-08-31 指標正名：NeedsReview → CandidateCoverageRisk（審查意見 4.2）。
    # 既有的 *_scriptmetrics.json 是 VM 當時產出的原件，**刻意不重寫**——它們是
    # provenance 錨點。改成比對時把舊鍵名對映到新鍵名，兩邊都不必動。
    got = {LEGACY_KEYS.get(k, k): v for k, v in got.items()}
    want = {LEGACY_KEYS.get(k, k): v for k, v in want.items()}

    # 2026-08-31 Playable% 拆層（審查意見 4.1）：新增的鍵在既有檔案裡沒有，
    # 不算「不符」。舊的 Playable% 仍保留且必須同值——那是回歸的重點。
    for k in NEW_KEYS:
        got.pop(k, None)

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


def available(stem):
    """乾淨 clone 缺料時要能說清楚缺什麼，而不是丟 FileNotFoundError。

    教材集（test_textbook）的授權未查證，刻意不入庫（見 .gitignore），
    所以那一案在公開 clone 上會缺料——這是設計上的取捨，不是壞掉。
    """
    for path in (os.path.join(BASE, "results", f"{stem}.jsonl"),
                 os.path.join(BASE, "results", f"{stem}_scriptmetrics.json")):
        if not os.path.exists(path):
            return False, os.path.relpath(path, BASE)
    split = resolve_split(stem)
    if not (split and split.exists()):
        # 說得出缺的是哪個檔，不要只說「缺 None」。
        name = stem.split("_", 1)[-1] if "_" in stem else stem
        return False, f"data/splits_v17/{name}.jsonl（或 data/splits/）"
    return True, None


def main():
    failed = skipped = 0
    for stem in CASES:
        ok, missing = available(stem)
        if not ok:
            skipped += 1
            print(f"− {stem:22} 略過：缺 {missing}")
            continue
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
    ran = len(CASES) - skipped
    if not ran:
        print("所有資料集都缺料，什麼都沒驗到——不能宣稱可重現")
        return 1
    tail = f"（另 {skipped} 個因缺料略過）" if skipped else ""
    print(f"{ran}/{len(CASES)} 個資料集逐欄相同——評分管線已可從頭重現{tail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
