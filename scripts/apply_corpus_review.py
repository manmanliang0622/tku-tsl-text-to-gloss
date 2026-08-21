#!/usr/bin/env python3
"""把三份人工校訂表套回**來源語料**（可重跑）。

來源：`data/reviews/人工校訂_*.xlsx`（2026-08-21 交付，三份共 6,634 列、涵蓋 6,783 個 id）

  文化部臺灣手語語料庫   5,125 列 → data/tslcorpus/parallel.jsonl（5,272 筆）
  中正大學辭典例句         542 列 → data/twtsl/twtsl_sentences.jsonl（544 筆）
  規則模板合成句           967 列 → data/synth/tsl_synth.jsonl（967 筆）

**為什麼改的是來源檔而不是訓練集**：`data/splits/` 由 `split_data.py` 從這三份切出，
`splits_json`／`splits_script` 又由 splits 衍生。修在來源層，下游重跑就全部繼承；
修在訓練階段，換一次格式就得重做一次。

**與 `add_correction.py` 的差別**：那支是「前端測到一句翻錯、補一筆」用的，會把每筆
修正複製 weight 份加權混入訓練集。本表 561 筆校訂的性質完全不同——它們不是新句子，
是既有語料的錯標，要的是**就地取代**。照 `apply_teacher_review.py` 的模式：取代 Gloss、
保留原值供追溯、據實更新 `review_status`、未審者維持原狀。

**追溯欄位**：被更正者一律寫入 `review_baseline_gloss_text`（＝本輪評估者實際看到的
Gloss），它同時是重跑時的冪等錨點。另依既有慣例補 `pre_review_gloss_text`，但**只在
該欄尚未存在時**才寫——合成句 45 筆更正全數落在 2026-07-24 老師已改過的句子上，
該欄已被那一輪佔用（存的是 7 月的更前值），覆蓋會讓兩輪的來歷混在一起。

**去重展開**：表格依內容去重，重複列的其他 id 放在「重複來源ID」欄（68 組共 147 個
額外 id）。一列的判定會套用到該欄列出的**每一個** id；已驗證同組 id 的中文與 Gloss
在來源檔中完全相同。

**不讀公式欄**：T:X（最終判定／最終Gloss／可匯入訓練）是 Excel 公式。本腳本一律從
J:S 自行推導，再與快取值對帳，不一致就中止——避免公式未重算時讀到空值。

**Gloss 正規化**：清掉全形標點 `「」『』，。！？、；：`、去除 token 前後空白與空 token。
依據：三份來源檔 6,783 筆 Gloss 中，全形標點出現 **0 次**（ASCII 的 `?` `!` `()` `+`
則是既有記號，予以保留）。有 21 筆校訂是依語料庫原始標註「補上引號」，正規化後與原值
相同——這類記為 annotation-only，內容不動，理由存 `review_note` 保留追溯。

用法：
  python3 scripts/apply_corpus_review.py --dry-run    # 只報告，不寫檔
  python3 scripts/apply_corpus_review.py              # 就地更新三份來源檔
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import openpyxl

BASE = Path(__file__).resolve().parent.parent
REVIEWS = BASE / "data" / "reviews"
REVIEW_TAG = "human-reviewed-2026-08-21"

# 表格欄位（A..X），只用 A:S；T:X 是公式欄，僅供對帳
COL = {c: i for i, c in enumerate(
    "A B C D E F G H I J K L M N O P Q R S T U V W X".split())}

SHEETS = [
    ("人工校訂_文化部臺灣手語語料庫_5125筆.xlsx",
     Path("data") / "tslcorpus" / "parallel.jsonl"),
    ("人工校訂_中正大學臺灣手語線上辭典例句_542筆.xlsx",
     Path("data") / "twtsl" / "twtsl_sentences.jsonl"),
    ("人工校訂_規則模板合成句_967筆.xlsx",
     Path("data") / "synth" / "tsl_synth.jsonl"),
]

# 來源 Gloss 從未使用的全形標點；校訂表引入的引號屬語料庫的引述標註，不是手語詞
FULLWIDTH_PUNCT = re.compile(r"[「」『』，。！？、；：]")


def cell(row, letter):
    v = row[COL[letter]] if COL[letter] < len(row) else None
    return "" if v is None else str(v).strip()


def norm_gloss(text):
    """正規化 Gloss：去全形標點、去 token 前後空白、丟掉空 token。"""
    stripped = FULLWIDTH_PUNCT.sub("", str(text))
    return "/".join(t for t in (x.strip() for x in stripped.split("/")) if t)


def final_verdict(main, second):
    """自行推導最終判定（等同表中 T 欄公式，但不依賴其快取值）。"""
    if main == "":
        return "未填"
    if main == "需複審":
        return "待複審" if second in ("", "仍需複審") else second
    return main


def read_sheet(path):
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    try:
        rows = [r for r in wb["人工校訂"].iter_rows(min_row=2, values_only=True)
                if any(c is not None and str(c).strip() for c in r)]
    finally:
        wb.close()
    return rows


def load_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for e in rows:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def apply_one(sheet_path, source_path, dry_run):
    rows = read_sheet(sheet_path)
    records = load_jsonl(source_path)
    by_id = {e["id"]: e for e in records}

    stats = Counter()
    problems = []
    annotation_only, excluded_ids = [], []

    for r in rows:
        rid = cell(r, "A")
        dup = cell(r, "B") or rid
        ids = [x.strip() for x in dup.split("|") if x.strip()]
        sheet_zh, sheet_gloss = cell(r, "E"), cell(r, "F")
        main, issue, fix_zh = cell(r, "J"), cell(r, "K"), cell(r, "L")
        fix_gloss, reason = cell(r, "M"), cell(r, "N")
        second, second_gloss = cell(r, "P"), cell(r, "R")

        verdict = final_verdict(main, second)
        if verdict != cell(r, "T"):
            problems.append(f"{rid}: 推導判定「{verdict}」與公式欄「{cell(r,'T')}」不符")
            continue
        if verdict == "需更正":
            corrected = second_gloss if main == "需複審" else fix_gloss
        else:
            corrected = ""

        for sid in ids:
            e = by_id.get(sid)
            if e is None:
                problems.append(f"{sid}: 來源檔查無此 id")
                continue
            # 漂移檢查：來源現值需等於表上的原始值（首次套用），
            # 或來源已存有 pre_review_gloss_text 等於表上原始值（重跑）。
            src_gloss = e["gloss_text"].strip()
            baseline = str(e.get("review_baseline_gloss_text", "")).strip()
            fresh = norm_gloss(src_gloss) == norm_gloss(sheet_gloss)
            reapplied = bool(baseline) and norm_gloss(baseline) == norm_gloss(sheet_gloss)
            if not fresh and not reapplied:
                problems.append(
                    f"{sid}: 來源 Gloss 與表上原始值不符（來源={src_gloss!r}／表={sheet_gloss!r}）")
                continue
            if e["chinese"].strip() != sheet_zh:
                problems.append(
                    f"{sid}: 來源中文與表上原始值不符（來源={e['chinese']!r}／表={sheet_zh!r}）")
                continue

            if verdict == "未填":
                stats["未填（維持原狀）"] += 1
                continue

            if issue:
                e["review_issue_type"] = issue
            if reason:
                e["review_note"] = reason

            if verdict == "排除":
                e["review_status"] = REVIEW_TAG + "-excluded"
                e["train_eligible"] = False
                excluded_ids.append(sid)
                stats["排除"] += 1
                continue

            e["train_eligible"] = True

            if verdict == "通過":
                e["review_status"] = REVIEW_TAG
                stats["通過"] += 1
                continue

            # verdict == 需更正
            new_gloss = norm_gloss(corrected)
            if not new_gloss:
                problems.append(f"{sid}: 判定需更正但更正 Gloss 為空")
                continue
            if new_gloss == norm_gloss(sheet_gloss):
                # 正規化後與原值相同：多為依語料庫原始標註補引號，對 Text→Gloss 無資訊
                e["review_status"] = REVIEW_TAG
                e["review_annotation_only"] = True
                annotation_only.append((sid, reason))
                stats["需更正→正規化後無實質改動"] += 1
                continue
            # 本輪基準：評估者看到的 Gloss。重跑時 src_gloss 已是修正後的值，
            # 故一律以表上原始值為準，避免把修正後的值誤存成「原值」。
            e["review_baseline_gloss_text"] = sheet_gloss
            if "pre_review_gloss_text" not in e:
                e["pre_review_gloss_text"] = sheet_gloss
            e["gloss_text"] = new_gloss
            e["gloss"] = [g for g in new_gloss.split("/") if g]
            e["review_status"] = REVIEW_TAG + "-corrected"
            stats["需更正→已取代 Gloss"] += 1
            if fix_zh and fix_zh != e["chinese"].strip():
                e["pre_review_chinese"] = e["chinese"]
                e["chinese"] = fix_zh
                stats["同時修正中文"] += 1

    if problems:
        return stats, problems, annotation_only, excluded_ids

    if not dry_run:
        write_jsonl(source_path, records)
    return stats, problems, annotation_only, excluded_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只報告，不寫檔")
    args = ap.parse_args()

    total_problems = 0
    for sheet_name, rel_src in SHEETS:
        sheet_path = REVIEWS / sheet_name
        source_path = BASE / rel_src
        print("=" * 72)
        print(f"{sheet_name}\n  → {rel_src}")
        stats, problems, annotation_only, excluded = apply_one(
            sheet_path, source_path, args.dry_run)
        for k, v in stats.most_common():
            print(f"    {k}: {v}")
        if annotation_only:
            print(f"    ── 正規化後無實質改動的 {len(annotation_only)} 筆："
                  f"{', '.join(i for i, _ in annotation_only)}")
        if excluded:
            print(f"    ── 排除的 {len(excluded)} 筆：{', '.join(excluded)}")
        if problems:
            total_problems += len(problems)
            print(f"    ⚠ {len(problems)} 個問題，本檔未寫入：")
            for p in problems[:20]:
                print(f"      {p}")
            if len(problems) > 20:
                print(f"      …另有 {len(problems)-20} 筆")

    print("=" * 72)
    if total_problems:
        raise SystemExit(f"共 {total_problems} 個問題，請先處理再重跑")
    print("乾跑完成，未寫檔。" if args.dry_run else "已寫回三份來源檔。")
    print("下一步：重跑 split_data.py → build_json_targets.py／build_script_dataset.py")


if __name__ == "__main__":
    main()
