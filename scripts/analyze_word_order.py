#!/usr/bin/env python3
"""統計文化部語料庫的語序重排程度，特別是「受詞前置（O-V）」的比例。

方法（無中文剖析器下的可重現啟發式，限制見文末）：
  1. **對齊**：Gloss token 若能在中文句字串中找到（子字串比對），取其字元位置，
     即可比較「該詞在中文的先後」與「在 Gloss 的先後」。
  2. **整體重排率**：對齊詞兩兩比較，計算逆序對比例（Kendall tau 距離的正規化版）。
     0 = 與中文完全同序，1 = 完全相反。
  3. **O-V 前置**：句中若含已知動詞 V，且某個對齊詞 X 滿足
       中文位置：X 在 V 之後　→　Gloss 位置：X 在 V 之前
     即判定為一次「跨動詞前置」。這正是 Tai & Su (2006) 描述的
     受詞/主題前置現象（例：蟑螂 我 討厭、小偷 警察 追）。

動詞清單：取自語料庫高頻詞中語意明確為動作/心理動詞者，
並納入三篇中正大學論文例句所用動詞。清單寫死於本檔，可增修後重跑。

用法：python3 scripts/analyze_word_order.py [--source tslcorpus|papers|all]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"

# 語意明確的及物動詞（會有受詞，才談得上 O-V）。可依需要增修。
VERBS = {
    "見", "看", "看到", "見到", "買", "賣", "吃", "喝", "討厭", "喜歡", "怕", "追",
    "咬", "打", "罵", "愛", "相信", "帶", "帶去", "學", "學習", "讀書", "唸書",
    "問", "告訴", "教", "幫", "幫忙", "找", "準備", "整理", "收集", "交給",
    "邀請", "認識", "知道", "懂", "聽", "拿", "放", "穿", "洗", "煮", "做",
    "寫", "畫", "用", "開", "關", "送", "給", "拍", "騎", "坐車", "搭",
}


def toks(gloss_text):
    return [t for t in str(gloss_text).replace("／", "/").split("/") if t.strip()]


def align(chinese, gloss_tokens):
    """回傳 [(gloss_index, chinese_char_pos, token)]，只含能在中文中定位者。"""
    out, used = [], 0
    for gi, t in enumerate(gloss_tokens):
        core = t.rstrip("+＋").split("+")[0]      # 去重複記號與複合標記
        if len(core) < 1:
            continue
        pos = chinese.find(core)
        if pos >= 0:
            out.append((gi, pos, core))
    return out


def inversion_rate(pairs):
    """對齊詞的逆序對比例：0=與中文同序，1=完全逆序。"""
    n = len(pairs)
    if n < 2:
        return None
    inv = tot = 0
    for a in range(n):
        for b in range(a + 1, n):
            gi_a, ci_a, _ = pairs[a]
            gi_b, ci_b, _ = pairs[b]
            if ci_a == ci_b:
                continue
            tot += 1
            # gloss 順序與中文順序相反即為逆序對
            if (gi_a < gi_b) != (ci_a < ci_b):
                inv += 1
    return inv / tot if tot else None


def analyze(rows, label):
    n_sent = len(rows)
    inv_rates, ov_sents, ov_examples = [], 0, []
    verb_sents = 0
    for r in rows:
        g = toks(r["gloss_text"])
        ch = r["chinese"]
        pairs = align(ch, g)
        ir = inversion_rate(pairs)
        if ir is not None:
            inv_rates.append(ir)
        # O-V 偵測
        vpos = [(gi, ci, t) for gi, ci, t in pairs if t in VERBS]
        if not vpos:
            continue
        verb_sents += 1
        found = None
        for vgi, vci, vt in vpos:
            for gi, ci, t in pairs:
                if t == vt:
                    continue
                if ci > vci and gi < vgi:      # 中文在動詞後、Gloss 在動詞前
                    found = (t, vt)
                    break
            if found:
                break
        if found:
            ov_sents += 1
            if len(ov_examples) < 8:
                ov_examples.append((ch, r["gloss_text"], found))
    return {
        "label": label, "sentences": n_sent,
        "aligned_sentences": len(inv_rates),
        "mean_inversion": sum(inv_rates) / len(inv_rates) if inv_rates else 0,
        "high_inversion_pct": (sum(1 for x in inv_rates if x >= 0.5) / len(inv_rates) * 100)
                              if inv_rates else 0,
        "verb_sentences": verb_sents,
        "ov_sentences": ov_sents,
        "ov_pct": ov_sents / verb_sents * 100 if verb_sents else 0,
        "examples": ov_examples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="all",
                    choices=["tslcorpus", "papers", "synth", "all"])
    args = ap.parse_args()

    sources = {}
    p = DATA / "tslcorpus" / "parallel.jsonl"
    if p.exists():
        sources["tslcorpus（文化部真實語料）"] = [json.loads(l) for l in
                                          p.read_text(encoding="utf-8").splitlines() if l.strip()]
    p = DATA / "papers" / "paper_examples.jsonl"
    if p.exists():
        sources["papers（中正論文例句）"] = [json.loads(l) for l in
                                      p.read_text(encoding="utf-8").splitlines()
                                      if l.strip() and not json.loads(l).get("has_notation")]
    p = DATA / "synth" / "tsl_synth.jsonl"
    if p.exists():
        sources["synth（本專案合成句）"] = [json.loads(l) for l in
                                     p.read_text(encoding="utf-8").splitlines() if l.strip()]

    results = []
    for label, rows in sources.items():
        if args.source != "all":
            key = {"tslcorpus": "tslcorpus", "papers": "papers", "synth": "synth"}[args.source]
            if not label.startswith(key):
                continue
        results.append(analyze(rows, label))

    print(f"{'來源':28s}{'句數':>7s}{'可對齊':>7s}{'平均逆序率':>11s}"
          f"{'高度重排%':>10s}{'含動詞句':>9s}{'O-V 前置':>9s}{'比例':>8s}")
    print("-" * 92)
    for r in results:
        print(f"{r['label']:28s}{r['sentences']:>7d}{r['aligned_sentences']:>7d}"
              f"{r['mean_inversion']*100:>10.1f}%{r['high_inversion_pct']:>9.1f}%"
              f"{r['verb_sentences']:>9d}{r['ov_sentences']:>9d}{r['ov_pct']:>7.1f}%")

    for r in results:
        if r["examples"]:
            print(f"\n=== {r['label']} 的 O-V 前置實例 ===")
            for ch, gl, (obj, vb) in r["examples"]:
                print(f"  中文: {ch[:36]}")
                print(f"  Gloss: {gl}")
                print(f"         →「{obj}」在中文位於「{vb}」之後，Gloss 中提前至其前")

    print("""
方法限制（據實說明）：
  - 子字串對齊會漏掉「中文與 Gloss 用字不同」的詞（如 公車→公共汽車、痛→疼），
    這類句子不計入對齊統計，故實際重排程度可能被低估。
  - 動詞清單為人工列舉，非完整詞性標註；未列入的動詞不會被偵測。
  - 「跨動詞前置」不等於語言學上嚴格的受詞前置（可能是時間/地點狀語移位），
    僅作為規模量級的估計，用於決定是否值得投入資料增強。""")


if __name__ == "__main__":
    main()
