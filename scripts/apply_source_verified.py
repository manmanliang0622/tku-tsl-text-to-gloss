#!/usr/bin/env python3
"""政策：語料庫／辭典／學術可驗證的 gloss 層資料，直接開放訓練（2026-08-05）。

依使用者決策：凡 Text→Gloss 的 gloss 層可由文化部語料庫、中正辭典或學術文獻
**驗證**者，不再等待逐句人工教師裁定，直接允許進訓練；僅 NMS（疑問／否定表情
範圍、狀態變化）、手形、移動方向、地區變體屬**影片軌**，與 Text→Gloss 訓練標的
無關，故不阻擋 gloss 層訓練，仍留待母語聾人於影片端裁定。

本腳本落實此政策中「原本因待影片而排除訓練」的 7 句：它們的 gloss 層已於
2026-08-04（scripts/make_video_pending_sheet.py）以語料庫 5,272 句實證裁定。
套用該裁定、把 teacher_train_eligible 轉為 True，原 Gloss 保留。

語料實證來源（可查證，非虛構）：
  - SYN0710–0714（WH「哪裡」＋情態「要」）：語料庫「WH 後接『要』」= 0/5,272；
    WH 取句末、『要』多不打出 → 去句末「要」。
  - SYN0892（上學）：語料庫 讀書 61 vs 唸書 7 → 詞形 唸書→讀書（同辭典詞條）。
  - SYN0955（感冒了）：語料庫「感冒」皆單一 gloss、無對應「了」token → gloss 不變；
    「了」之狀態變化屬影片軌。

用法：python3 scripts/apply_source_verified.py  → 就地更新 data/synth/tsl_synth.jsonl
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SYNTH = BASE / "data" / "synth" / "tsl_synth.jsonl"

# 語料實證裁定（gloss_text；來源見 make_video_pending_sheet.py）
VERIFIED = {
    "SYN0710": "你/去/哪裡",
    "SYN0711": "今天/你/去/哪裡",
    "SYN0712": "今天/你/去/哪裡",
    "SYN0713": "明天/你/去/哪裡",
    "SYN0714": "明天/你/去/哪裡",
    "SYN0892": "你/讀書/哪裡",
    "SYN0955": "我/感冒",          # gloss 不變，僅開放訓練
}
# 這些句 NMS 仍待影片（不阻擋 gloss 訓練）
NMS_VIDEO_PENDING = set(VERIFIED)


def main():
    rows = [json.loads(l) for l in SYNTH.read_text(encoding="utf-8").splitlines()]
    n_gloss = n_open = 0
    for e in rows:
        v = VERIFIED.get(e["id"])
        if v is None:
            continue
        if v != e["gloss_text"]:
            e.setdefault("pre_review_gloss_text", e["gloss_text"])
            e["gloss_text"] = v
            e["gloss"] = [g for g in v.split("/") if g]
            e["corpus_verified_correction"] = True
            n_gloss += 1
        e["teacher_train_eligible"] = True
        e["review_status"] = "corpus-verified-2026-08-05"
        e["nms_video_pending"] = e["id"] in NMS_VIDEO_PENDING
        n_open += 1

    with SYNTH.open("w", encoding="utf-8") as f:
        for e in rows:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    from collections import Counter
    elig = sum(1 for e in rows if e.get("teacher_train_eligible"))
    print(f"語料實證修正 gloss: {n_gloss} 句；開放訓練: {n_open} 句")
    print(f"合成句 teacher_train_eligible=True 總數: {elig}/{len(rows)}")
    print("review_status 分布:", dict(Counter(e.get("review_status") for e in rows)))


if __name__ == "__main__":
    main()
