#!/usr/bin/env python3
"""從中正大學臺灣手語研究論文抽取「Gloss ↔ 中文」例句（語言學家標註的黃金平行資料）。

來源（taiwansign.ccu.edu.tw）：
  - Tai & Su 2006《台灣手語的呼應方式》
  - 陳泱儒《台灣手語篇章中韻律標記與句法結構的對映》
  - 魏彥宜 2022《臺灣手語同義詞研究》

論文例句格式：
    (1) 姐姐 怕 蟑螂
        ‘姐姐怕蟑螂。’

抽取要點（皆為論文特有寫法，需轉成本專案格式）：
  1. 論文以**空格**分隔 Gloss；本專案用「/」。
  2. 論文的「/」代表**替代詞**（富/多、快1/快2），不是分隔符 → 取第一個。
  3. 下標數字為呼應/變體標記（幫1、椅子1）→ 去除。
  4. 括號註記（i→我）、（桌）、?標記 → 去除。
  5. 英文例句、單純引用詞（‘for’）→ 排除。

輸出：data/papers/paper_examples.jsonl
"""
import json
import re
import unicodedata
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "data" / "papers"

SOURCES = {
    "Tai_and_Su_2006": "戴浩一、蘇秀芬 2006《台灣手語的呼應方式》，"
                       "《百川匯海：李壬癸先生七秩壽慶論文集》341-363",
    "Chen,Yang-ru": "陳泱儒《台灣手語篇章中韻律標記與句法結構的對映》，"
                    "國立中正大學語言學研究所碩士論文",
    "Yen-YiWei2022": "魏彥宜 2022《臺灣手語同義詞研究》，"
                     "國立中正大學語言學研究所碩士論文",
}

# 例句編號行：(1) / （1） / (1) a.
NUM_RE = re.compile(r"^\s*[（(]\s*\d+\s*[）)]\s*(?:[a-z]\s*[.、]\s*)?(.*)$")
# 中文翻譯行：‘...’
TRANS_RE = re.compile(r"^\s*[‘'’]([^’']+)[’'‘]\s*$")


def has_cjk(s):
    return any("一" <= c <= "鿿" for c in s)


def clean_gloss_line(line):
    """把論文 Gloss 行轉成 token list。回傳 None 表示不可用。"""
    s = unicodedata.normalize("NFKC", line).strip()
    if not s or not has_cjk(s):
        return None
    # 去括號註記（含全形）
    s = re.sub(r"[（(][^（）()]*[）)]", "", s)
    # 去行末標點與問號標記
    s = s.strip(" 。，、？?!！；;：:")
    # 排除含拉丁字母為主的行（英文例句、IX-1 等注釋體）
    if re.search(r"[A-Za-z]{3,}", s):
        return None
    s = s.replace("，", " ").replace(",", " ")   # 子句逗號視為分隔
    toks = []
    for raw in s.split():
        t = raw.strip("，。、；：？?!！")
        if not t:
            continue
        # 「/」在論文是替代詞 → 取第一個
        t = t.split("/")[0]
        # 去下標數字（呼應/變體標記）
        t = re.sub(r"\d+$", "", t)
        t = t.strip("＋+")
        if t and has_cjk(t):
            if not toks or toks[-1] != t:      # 去相鄰重複（快1/快2 之類）
                toks.append(t)
    if len(toks) < 2:          # 至少兩個 Gloss 才算句
        return None
    return toks


# 呼應／分類詞／描述性註解等語言學標記：下游動作庫無法檢索，訓練與測試皆排除。
# 2026-08-08 收緊：原版只擋 代形詞/+/→/ijk，實測仍放行了
#   桌子-i、i-問-j（索引與呼應標記）、鉛筆的外形_由長變短（描述性註解）、
#   花分類詞、倚靠.垂直-3A（位置標記）、介系詞（後設標籤）等不可檢索寫法。
NOTATION_RE = re.compile(
    r"代形詞|分類詞|介系詞|[+＋]|[→←]|_"                 # 標記詞與描述性底線
    r"|[-－][ijk0-9]"                                    # -i / -j / -3A 索引
    r"|(?<=[一-鿿])[ijk](?![一-鿿])"                      # 詞尾 i/j/k
    r"|(?<=[一-鿿])\.(?=[一-鿿])"                        # 倚靠.垂直
    r"|的外形|的長度|的厚度|的高度"                        # 描述性 gloss
)


def has_notation(gloss_text):
    return bool(NOTATION_RE.search(gloss_text))


def extract(txt_path, source_key):
    lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    out, i = [], 0
    while i < len(lines):
        m = NUM_RE.match(lines[i])
        if not m:
            i += 1
            continue
        # 例句編號行之後，往下找「翻譯行」；中間允許 b. 等續行
        gloss_cand = m.group(1)
        j = i + 1
        while j < len(lines) and j - i <= 4:
            tm = TRANS_RE.match(lines[j])
            if tm:
                chinese = tm.group(1).strip(" 。")
                # 用最靠近翻譯行的 gloss 候選（可能是 b. 行）
                cand = gloss_cand
                for k in range(j - 1, i - 1, -1):
                    c = re.sub(r"^\s*[a-z]\s*[.、]\s*", "", lines[k]).strip()
                    if c and has_cjk(c) and not TRANS_RE.match(lines[k]):
                        cand = c
                        break
                toks = clean_gloss_line(cand)
                if toks and has_cjk(chinese) and len(chinese) >= 2 \
                        and not re.search(r"[A-Za-z]{3,}", chinese):
                    out.append({"chinese": chinese, "gloss": toks,
                                "gloss_text": "/".join(toks),
                                "raw_gloss_line": cand.strip()})
                break
            j += 1
        i = j + 1 if j > i else i + 1
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", default="/private/tmp/claude-501/-Users-leo-Documents-----"
                                         "/7105f925-4b9b-4613-a34d-48bc76579f86/scratchpad",
                    help="放 .txt（pdftotext 產出）的目錄")
    ap.add_argument("--scan-all", action="store_true",
                    help="掃描目錄下所有 .txt，而非只取 SOURCES 列出的三篇")
    ap.add_argument("--out", default="paper_examples.jsonl")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    src_dir = Path(args.src_dir)
    targets = ({p.stem: SOURCES.get(p.stem, f"中正大學手語研究中心文獻：{p.stem}")
                for p in sorted(src_dir.glob("*.txt"))}
               if args.scan_all else SOURCES)
    rows, n = [], 0
    seen = set()
    for key, citation in targets.items():
        p = src_dir / f"{key}.txt"
        if not p.exists():
            print(f"  ⚠ 找不到 {p}，略過")
            continue
        got = extract(p, key)
        for e in got:
            dedup = (e["chinese"], e["gloss_text"])
            if dedup in seen:
                continue
            seen.add(dedup)
            n += 1
            rows.append({
                "id": f"PAP{n:04d}",
                "type": "sentence",
                "chinese": e["chinese"],
                "gloss": e["gloss"],
                "gloss_text": e["gloss_text"],
                "nms": None,
                "source": "paper",
                "paper": key,
                "citation": citation,
                "raw_gloss_line": e["raw_gloss_line"],
                "batch": "中正大學手語論文例句",
                # 含呼應/分類詞標記者不進訓練（下游無法檢索），但保留供語法參考
                "has_notation": has_notation(e["gloss_text"]),
            })
        usable = sum(1 for e in got if not has_notation(e["gloss_text"]))
        print(f"  {key}: 抽出 {len(got)} 例（可訓練 {usable}）")
    path = OUT / args.out
    with path.open("w", encoding="utf-8") as f:
        for e in rows:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"OK → {path.relative_to(BASE)}（去重後 {len(rows)} 句）")


if __name__ == "__main__":
    main()
