#!/usr/bin/env python3
"""衍生檔是不是比它的輸入舊——一支指令掃完整條資料鏈。

動機（2026-09-02）：這個專案的資料是一條長鏈——

    影片庫 lexicon.json ＋ 品質掃描 entries_final.csv
        → sign_inventory.jsonl（總表）
            → splits（切分，也吃三份來源語料）
                → splits_script_v*（訓練資料）
                    → results/*.jsonl（推論）
                        → results/*_scriptmetrics.json（指標）

鏈上任何一環更新，下游全部過期。而「過期」不會報錯，只會安靜地讓你拿舊
結論做新決定——2026-09-01 影片庫更新後，總表、切分、訓練資料一次全舊掉，
但沒有任何東西會告訴你。

判定方式是比對 mtime，粗但誠實：只回答「有沒有可能過期」，不回答「內容
是否真的不同」。要精確比對內容請看各自的 hash（data_preflight.py 管來源，
infer_script_model 的 *.manifest.json 管推論產物）。

用法：
    python3 scripts/staleness_audit.py          # 掃描並列出過期項
    python3 scripts/staleness_audit.py --json   # 機器可讀

退出碼：0＝全部最新或只有可接受的落後；1＝有衍生檔比輸入舊。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# (產物, [輸入...], 說明)。產物不存在時視為「未產生」，不算過期。
CHAIN: list[tuple[str, list[str], str]] = [
    ("data/signs/sign_inventory.jsonl",
     ["data/video/lexicon.json", "data/video/entries_final.csv",
      "scripts/build_sign_inventory.py"],
     "sign_id 總表（含影片品質分層）"),
    ("data/signs/gloss_to_sign.json",
     ["data/signs/sign_inventory.jsonl"],
     "gloss → sign_id 查詢索引"),
    ("data/splits/manifest.json",
     ["data/tslcorpus/parallel.jsonl", "data/twtsl/twtsl_sentences.jsonl",
      "data/synth/tsl_synth.jsonl", "scripts/split_data.py"],
     "train/dev/test 切分"),
    ("data/splits_script_v18/train.jsonl",
     ["data/splits/manifest.json", "data/signs/sign_inventory.jsonl",
      "scripts/build_script_dataset.py", "scripts/sign_candidates.py",
      "scripts/eval_video_coverage.py", "scripts/script_schema.py"],
     "v18 訓練資料"),
]

# 推論產物 → 它用的訓練資料。tag 前綴對應 splits_script 目錄。
INFER_TAGS = {"v18script": "data/splits_script_v18"}


def mtime(p: Path) -> float | None:
    return p.stat().st_mtime if p.exists() else None


def check_chain() -> list[dict]:
    out = []
    for target, inputs, desc in CHAIN:
        tp = BASE / target
        tm = mtime(tp)
        if tm is None:
            out.append({"target": target, "desc": desc, "state": "未產生",
                        "newer_inputs": []})
            continue
        newer = []
        for src in inputs:
            sm = mtime(BASE / src)
            if sm is None:
                newer.append({"input": src, "state": "缺檔"})
            elif sm > tm:
                newer.append({"input": src,
                              "ahead_sec": round(sm - tm)})
        out.append({"target": target, "desc": desc,
                    "state": "過期" if any("ahead_sec" in n for n in newer) else "最新",
                    "newer_inputs": newer})
    return out


def check_infer() -> list[dict]:
    """推論產物是不是比它的訓練資料舊，且指標檔是不是比推論產物舊。"""
    out = []
    results = BASE / "results"
    if not results.is_dir():
        return out
    for tag, data_dir in INFER_TAGS.items():
        dm = mtime(BASE / data_dir / "train.jsonl")
        for pred in sorted(results.glob(f"{tag}_*.jsonl")):
            if pred.name.endswith("_scriptmetrics.json"):
                continue
            pm = mtime(pred)
            row = {"target": str(pred.relative_to(BASE)), "desc": "推論產物",
                   "state": "最新", "newer_inputs": []}
            if dm and pm and dm > pm:
                row["state"] = "過期"
                row["newer_inputs"].append({"input": f"{data_dir}/train.jsonl",
                                            "ahead_sec": round(dm - pm)})
            metrics = pred.with_name(pred.stem + "_scriptmetrics.json")
            mm = mtime(metrics)
            if mm is None:
                out.append(row)
                out.append({"target": str(metrics.relative_to(BASE)),
                            "desc": "指標檔", "state": "未產生", "newer_inputs": []})
                continue
            out.append(row)
            out.append({"target": str(metrics.relative_to(BASE)), "desc": "指標檔",
                        "state": "過期" if pm and mm < pm else "最新",
                        "newer_inputs": ([{"input": pred.name,
                                           "ahead_sec": round(pm - mm)}]
                                         if pm and mm < pm else [])})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = check_chain() + check_infer()
    stale = [r for r in rows if r["state"] == "過期"]

    if args.json:
        print(json.dumps({"rows": rows, "stale": len(stale)},
                         ensure_ascii=False, indent=2))
        return 1 if stale else 0

    print("=" * 68)
    print("衍生檔新舊稽核（比對 mtime）")
    print("=" * 68)
    icon = {"最新": "✓", "過期": "✗", "未產生": "·"}
    for r in rows:
        print(f"  {icon.get(r['state'], '?')} {r['state']:5} {r['target']}")
        if r["desc"] not in ("推論產物", "指標檔"):
            print(f"           {r['desc']}")
        for n in r["newer_inputs"]:
            if "ahead_sec" in n:
                m = n["ahead_sec"] // 60
                ago = f"{m} 分鐘" if m < 120 else f"{m // 60} 小時"
                print(f"           ↑ {n['input']} 比它新 {ago}")
            else:
                print(f"           ⚠ {n['input']}：{n['state']}")
    print()
    if stale:
        print(f"✗ {len(stale)} 個衍生檔比輸入舊，需重建：")
        for r in stale:
            print(f"    {r['target']}")
        return 1
    print("✓ 沒有衍生檔比輸入舊")
    return 0


if __name__ == "__main__":
    sys.exit(main())
