#!/usr/bin/env python3
"""把手語老師人工審核結果套進訓練資料（可重跑）。

來源：outputs/019f87d9-.../reviewed_data_2026-07-23/tsl_synth_reviewed.jsonl
（commit abe20fe「手語老師人工審核」：針對 115 句『需修正或追加實證』逐句審，
 修正 108 句、7 句維持待母語者影片裁定）。

本腳本把老師的 `teacher_corrected_gloss` 套進 data/synth/tsl_synth.jsonl，並據實
更新 review_status：
  - 108 句已修正 → `teacher-reviewed-2026-07-24`，原 Gloss 存 `pre_review_gloss_text`。
  - 7 句待影片（SYN0710–0714、SYN0892、SYN0955）→ `reviewed-pending-video`。
  - 其餘未逐句審核者 → 維持 `pending`（不冒充已通過）。

**保守政策**：本腳本只改 Gloss 與審核狀態，不改變 split 的 rule-derived 排除政策；
合成資料整體仍未取得全面人工通過，不得作正式訓練報告依據（見審核報告）。

用法：python3 scripts/apply_teacher_review.py  → 就地更新 data/synth/tsl_synth.jsonl
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SYNTH = BASE / "data" / "synth" / "tsl_synth.jsonl"
REVIEWED = (BASE / "outputs" / "019f87d9-976d-7961-84aa-a05c910dcd5c" /
            "reviewed_data_2026-07-23" / "tsl_synth_reviewed.jsonl")
PENDING_VIDEO = {"SYN0710", "SYN0711", "SYN0712", "SYN0713", "SYN0714",
                 "SYN0892", "SYN0955"}


def main():
    reviewed = {}
    for line in REVIEWED.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        reviewed[r["id"]] = r
    rows = [json.loads(l) for l in SYNTH.read_text(encoding="utf-8").splitlines()]

    n_corr = n_pend = 0
    for e in rows:
        rid = e["id"]
        rv = reviewed.get(rid)
        corrected = rv.get("teacher_corrected_gloss") if rv else None
        if corrected and corrected != e["gloss_text"]:
            e["pre_review_gloss_text"] = e["gloss_text"]
            e["gloss_text"] = corrected
            e["gloss"] = [g for g in corrected.split("/") if g]
            e["teacher_corrected"] = True
            e["review_status"] = "teacher-reviewed-2026-07-24"
            n_corr += 1
        elif rid in PENDING_VIDEO:
            e["review_status"] = "reviewed-pending-video"
            n_pend += 1
        # 其餘維持原 review_status（pending）

    with SYNTH.open("w", encoding="utf-8") as f:
        for e in rows:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"套用老師修正: {n_corr} 句；標待影片: {n_pend} 句")
    print("review_status 分布:", dict(Counter(e["review_status"] for e in rows)))
    # 影響訓練者（attested-pattern 被修正的）
    aff = sum(1 for e in rows if e.get("teacher_corrected")
              and e.get("confidence") == "attested-pattern")
    print(f"其中影響訓練集的 attested-pattern 修正: {aff} 句")


if __name__ == "__main__":
    main()
