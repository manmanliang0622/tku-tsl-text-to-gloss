#!/usr/bin/env python3
"""套用詞形校正到 data/synth/tsl_synth.jsonl（2026-08-04）。

背景：v_slots 機制原本規定「Gloss 一律用辭典正式名」，但以文化部語料庫
5,272 句真實聾人語料查核後發現三組詞形與真實用法相反：

    合成句採用      語料庫偏好      語料庫詞頻
    唸書       →   讀書           7 vs 61
    公共汽車    →   公車           0 vs 7
    疼        →   痛            0 vs 17

三組皆為同一辭典詞條的正式名／同義索引名，**下游檢索與可播放性不受影響**；
但保留舊詞形會讓 4,392 句語料庫與 22 句合成句對模型送出矛盾訊號，且在
語料庫 test 上會被判為錯誤。另註：T33／T35 原本引用的語料庫證據本身即使用
「公車」「讀書」「痛」，與其採用的正式名自相矛盾。

追溯：更新 gloss／gloss_text，並寫入 pre_vocab_fix_gloss_text、
vocab_fix_applied、vocab_fix_date、vocab_fix_basis；不動任何 teacher_* 欄位。

用法：python3 scripts/apply_vocab_fix.py  → 就地更新 tsl_synth.jsonl
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SYNTH = BASE / "data" / "synth" / "tsl_synth.jsonl"

FIXES = {
    "唸書": ("讀書", "語料庫詞頻 讀書 61 vs 唸書 7"),
    "公共汽車": ("公車", "語料庫詞頻 公車 7 vs 公共汽車 0；T33 所引證據本身即用公車"),
    "疼": ("痛", "語料庫詞頻 痛 17 vs 疼 0；T35 所引證據本身即用痛"),
}
DATE = "2026-08-04"


def main():
    rows = [json.loads(l) for l in SYNTH.read_text(encoding="utf-8").splitlines() if l.strip()]
    changed = []
    for e in rows:
        gloss = e.get("gloss") or []
        hit = [t for t in gloss if t in FIXES]
        if not hit:
            continue
        before = e.get("gloss_text")
        e["pre_vocab_fix_gloss_text"] = before
        e["gloss"] = [FIXES[t][0] if t in FIXES else t for t in gloss]
        e["gloss_text"] = "/".join(e["gloss"])
        e["vocab_fix_applied"] = True
        e["vocab_fix_date"] = DATE
        e["vocab_fix_basis"] = "；".join(sorted({FIXES[t][1] for t in hit}))
        changed.append((e["id"], before, e["gloss_text"]))

    with SYNTH.open("w", encoding="utf-8") as f:
        for e in rows:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"已套用詞形校正：{len(changed)} 句 / 合成總數 {len(rows)}")
    for cid, b, a in changed:
        print(f"  {cid}: {b}  →  {a}")

    # 驗證：舊詞形不應再出現於任何 gloss
    left = {t for e in rows for t in (e.get("gloss") or []) if t in FIXES}
    print(f"殘留舊詞形：{left or '無'}")


if __name__ == "__main__":
    main()
