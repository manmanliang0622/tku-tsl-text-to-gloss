#!/usr/bin/env python3
"""訓練／評估前的資料檢查（教授審查意見 5.1、5.2）。

一支指令回答三個問題：

  1. **三份訓練來源到位了嗎**：檔案在不在、筆數、sha256、人工校訂狀態。
     有 pending（未審）或 hash 與 manifest 不符就 fail closed。
  2. **切分是不是用修正後的程式產生的**：manifest 必須有 surface_normalization，
     且 train/dev 正規化中文重疊為 0（審查意見 2.4）。
  3. **這份 clone 能重現到什麼程度**：缺什麼、為什麼缺、怎麼補。
     刻意不入庫的（授權未查證）與單純沒產生的（可再生）要分開講——
     混在一起會讓「不能重現」聽起來比實際嚴重，也會讓真的缺料被忽略。

用法：
    python3 scripts/data_preflight.py            # 檢查並印報告
    python3 scripts/data_preflight.py --json     # 機器可讀
    python3 scripts/data_preflight.py --write-manifest   # 更新來源 hash 快照

退出碼：0＝可以往下跑；1＝有阻斷性問題。
「可再生但目前沒有」不算阻斷（會標 WARN），「來源缺失／hash 不符／有 pending」算。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SPLITS = BASE / "data" / "splits"
MANIFEST = BASE / "data" / "source_manifest.json"

# 三份**訓練**來源。textbook 與 papers 是純測試集，不在此列——
# 審查意見說「只處理三個來源」，但訓練來源本來就只有三個。
SOURCES = {
    "tslcorpus": BASE / "data" / "tslcorpus" / "parallel.jsonl",
    "twtsl": BASE / "data" / "twtsl" / "twtsl_sentences.jsonl",
    "synth": BASE / "data" / "synth" / "tsl_synth.jsonl",
}

# review_status 的前綴代表「已審」。任何不以這些開頭的都算 pending。
REVIEWED_PREFIXES = ("human-reviewed-", "teacher-reviewed-")

# 缺檔時要說清楚是「刻意不入庫」還是「可再生」。
OPTIONAL = {
    "data/splits/train.jsonl": ("可再生", "python3 scripts/split_data.py --use-all "
                                "--length-balance --no-papers --textbook-as-test "
                                "--corpus-test-ratio 0.12 --corpus-test-min-len 6 --seed 42"),
    "data/splits/dev.jsonl": ("可再生", "同上"),
    "data/splits_script/train.jsonl": ("可再生", "python3 scripts/build_script_dataset.py"),
    "data/video/lexicon.json": ("非公開素材", "scp tku-gpu:'~/0813/recordings/lexicon.json' data/video/"),
    "data/video/entries_final.csv": ("非公開素材", "scp tku-gpu:'~/0813/quality_scan/entries_final.csv' data/video/"),
    "data/splits_v17/test_textbook.jsonl": ("授權未查證·刻意不入庫", "見 .gitignore 說明"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_sources(recorded: dict) -> tuple[list, list, dict]:
    """回傳 (blockers, warnings, snapshot)。"""
    blockers, warnings, snap = [], [], {}
    for name, path in SOURCES.items():
        rel = str(path.relative_to(BASE))
        if not path.exists():
            blockers.append(f"訓練來源缺失：{rel}")
            continue
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        pending = [r for r in rows
                   if not str(r.get("review_status", "")).startswith(REVIEWED_PREFIXES)]
        eligible = [r for r in rows if r.get("train_eligible") is not False]
        digest = sha256(path)
        snap[name] = {"path": rel, "rows": len(rows), "sha256": digest,
                      "train_eligible": len(eligible),
                      "excluded": len(rows) - len(eligible)}
        if pending:
            blockers.append(
                f"{name}：{len(pending)} 筆 review_status 仍是 pending／未標記"
                f"（前 3 個 id：{[r.get('id') for r in pending[:3]]}）")
        prev = (recorded.get("sources") or {}).get(name)
        if prev and prev.get("sha256") != digest:
            blockers.append(
                f"{name}：sha256 與 data/source_manifest.json 記錄不符\n"
                f"      記錄 {prev['sha256'][:16]}… / 現在 {digest[:16]}…\n"
                f"      來源被改過。確認是預期的變更後跑 --write-manifest 更新快照。")
        elif not prev:
            warnings.append(f"{name}：source_manifest.json 尚無記錄（首次執行請跑 --write-manifest）")
    return blockers, warnings, snap


def check_splits() -> tuple[list, list]:
    blockers, warnings = [], []
    mf = SPLITS / "manifest.json"
    if not mf.exists():
        warnings.append("data/splits/manifest.json 不存在——切分尚未產生（可再生）")
        return blockers, warnings
    m = json.loads(mf.read_text(encoding="utf-8"))
    sn = m.get("surface_normalization")
    if not sn:
        blockers.append(
            "data/splits/ 是 2026-08-31 修正**之前**的程式產生的（manifest 無 "
            "surface_normalization）。那一版的去洩漏只比對原始字串，核心 33 句有 3 句、"
            "dev 有 11 句與 train 重疊。請重跑 split_data.py。")
    elif sn.get("train_dev_normalized_chinese_overlap"):
        blockers.append(
            f"train/dev 正規化中文重疊 {sn['train_dev_normalized_chinese_overlap']} 句")
    return blockers, warnings


def check_inventory() -> tuple[list, list]:
    blockers, warnings = [], []
    inv = BASE / "data" / "signs" / "sign_inventory.jsonl"
    if not inv.exists():
        blockers.append("data/signs/sign_inventory.jsonl 不存在")
        return blockers, warnings
    classes = set()
    n = 0
    for line in inv.read_text(encoding="utf-8").splitlines():
        if line.strip():
            n += 1
            classes.add(json.loads(line).get("asset_class"))
    if len(classes) <= 1:
        warnings.append(
            f"sign_inventory 的 asset_class 只有一種值（{classes}）＝沒有品質分層。"
            "重建總表時附上 data/video/entries_final.csv 才有 QualityPlayable% 可算。")
    return blockers, warnings


def check_optional() -> list:
    out = []
    for rel, (why, how) in OPTIONAL.items():
        if not (BASE / rel).exists():
            out.append((rel, why, how))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="輸出機器可讀結果")
    ap.add_argument("--write-manifest", action="store_true",
                    help="把目前的來源 hash 寫進 data/source_manifest.json")
    args = ap.parse_args()

    recorded = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    blockers, warnings, snap = check_sources(recorded)
    for fn in (check_splits, check_inventory):
        b, w = fn()
        blockers += b
        warnings += w
    missing = check_optional()

    if args.write_manifest:
        MANIFEST.write_text(
            json.dumps({"note": "訓練來源的內容雜湊快照。data_preflight.py 用它偵測"
                                "來源被改動；改動屬預期時重跑 --write-manifest 更新。",
                        "sources": snap}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(f"已寫出 {MANIFEST.relative_to(BASE)}")

    if args.json:
        print(json.dumps({"blockers": blockers, "warnings": warnings,
                          "sources": snap,
                          "missing_optional": [
                              {"path": p, "why": w, "how": h} for p, w, h in missing]},
                         ensure_ascii=False, indent=2))
        return 1 if blockers else 0

    print("=" * 66)
    print("資料 preflight")
    print("=" * 66)
    print("\n[訓練來源]")
    for name, info in snap.items():
        excl = f"，排除 {info['excluded']}" if info["excluded"] else ""
        print(f"  {name:12} {info['rows']:5} 筆（可訓練 {info['train_eligible']}{excl}）"
              f"  sha {info['sha256'][:12]}…")

    if missing:
        print("\n[未就位的檔案]")
        for rel, why, how in missing:
            print(f"  · {rel}\n      原因：{why}\n      取得：{how}")

    if warnings:
        print("\n[提醒]")
        for w in warnings:
            print(f"  ⚠ {w}")

    print()
    if blockers:
        print("[阻斷性問題]")
        for b in blockers:
            print(f"  ✗ {b}")
        print("\n不可往下訓練／評估。")
        return 1
    print("✓ 沒有阻斷性問題。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
