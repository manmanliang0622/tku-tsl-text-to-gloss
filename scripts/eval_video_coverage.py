#!/usr/bin/env python3
"""影片庫覆蓋率評估：資料集的 gloss 有多少虛擬人真的演得出來。

判定條件照 `0813/composer.js` 的實際行為，不是照詞庫大小：

  build()     對每個 token 做 **exact key lookup**，`lexicon[token]` 不存在
              就 `continue` —— 該詞整個不動。
  tokenize()  整詞找不到時，對該詞做**逐字貪婪切分**，用較短的鍵拼出來。
              所以缺詞的後果分兩種:「不動」與「動錯」，後者更危險
              （畫面流暢但打的是別的意思）。

因此缺口要分級，不能只報一個覆蓋率：
  T0 原樣可播 / T1 去標註記號 / T2 變體後綴別名 / T3 行政區後綴別名 /
  T4 異體字 / T5 數字逐位打 / T6 大小寫折疊 / T7 真缺
  —— 只有 T7 需要新素材，T1–T6 都是查詢端或規則就能解決的。

用法（lexicon.json 在學校主機上，先抓下來）:

    scp tku-gpu:'~/0813/recordings/lexicon.json' data/video/
    python scripts/eval_video_coverage.py

輸出：console 報表 + data/video/video_gap.json（逐詞缺口，可直接排補片順序）
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DEFAULT_LEXICON = BASE / "data" / "video" / "lexicon.json"
DEFAULT_TWTSL = Path.home() / "Documents" / "手語影片" / "twtsl"
DEFAULT_MOE = BASE / "data" / "video" / "moe_dict_words.json"

_PUNCT = "，。？！?!,.、;；:：…「」『』（）"
_VAR = re.compile(r"_(?:[A-Z]|S|\d+)$")        # 影片庫的變體後綴 美國_A / 會_S
_ADMIN = re.compile(r"(縣|市|鄉|鎮|區|村|里)$")   # 南投 ↔ 南投縣
# 資料集用異體字、影片庫用正體
_VARIANT_CHARS = str.maketrans({"妳": "你", "臺": "台", "她": "他", "牠": "他", "祂": "他"})
_DIGIT_SIGN = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"]


def norm(gloss: str) -> str:
    """去掉語料庫的標註記號，還原成可查詢的詞形。

    ++ 是重複記號、+X 是複合、(X) 是註記 —— 都不是詞的一部分。
    `signavatar/lexicon.py` 建詞庫時就把帶記號的 gloss 排除在外，
    但資料集裡仍以帶記號的形式當訓練標的，所以查詢端要對齊。
    """
    inner = re.fullmatch(r"\((.+)\)", gloss.strip())
    if inner:                                   # (手勢) → 手勢，別整串刪掉
        gloss = inner.group(1)
    gloss = gloss.strip().strip(_PUNCT)
    gloss = re.sub(r"\([^)]*\)", "", gloss)     # 告訴(他) → 告訴
    gloss = re.sub(r"\+\+$", "", gloss)         # 買++ → 買
    for sep in ("+", "→", "/", "~"):            # 腳踝+這 → 腳踝
        gloss = gloss.split(sep)[0]
    return gloss.strip().strip(_PUNCT)


class Library:
    """影片庫 + 各層別名，回答「這個 gloss 演不演得出來」。"""

    def __init__(self, lexicon: dict):
        self.lex = lexicon
        self.keys = set(lexicon)
        self.alias: dict[str, list[str]] = collections.defaultdict(list)
        for key in self.keys:
            for base in (_VAR.sub("", key).rstrip("_"), _ADMIN.sub("", key)):
                if base and base != key:
                    self.alias[base].append(key)
        # 含 ASCII 字母的鍵:大小寫 + 括號註記折疊(LINE (通訊軟體) → line)
        self.folded: dict[str, str] = {}
        for key in sorted(self.keys):
            bare = re.sub(r"\s*[（(][^)）]*[)）]\s*", "", key).strip().lower()
            if bare and re.search(r"[a-z]", bare):
                self.folded.setdefault(bare, key)

    def tier(self, gloss: str) -> tuple[str, str | None]:
        if gloss in self.keys:
            return "T0", gloss
        n = norm(gloss)
        if n and n in self.keys:
            return "T1", n
        for cand in (gloss, n):
            if cand and self.alias.get(cand):
                hit = self.alias[cand][0]
                return ("T2" if _VAR.search(hit) else "T3"), hit
        v = n.translate(_VARIANT_CHARS)
        if v and v in self.keys:
            return "T4", v
        # 數字逐位打:2009 → 二/零/零/九。詞庫本來就有中文數字,純規則
        if n and re.fullmatch(r"[0-9]+", n):
            if all(_DIGIT_SIGN[int(d)] in self.keys for d in n):
                return "T5", "/".join(_DIGIT_SIGN[int(d)] for d in n)
        # 大小寫與括號註記折疊:語料庫寫 Line,詞庫是 LINE (通訊軟體)
        if n and re.search(r"[A-Za-z]", n):
            hit = self.folded.get(n.lower())
            if hit:
                return "T6", hit
        return "T7", None

    def greedy(self, word: str) -> tuple[list[str], int]:
        """composer.js tokenize 的貪婪切分：回傳 (切出的塊, 沒被覆蓋的字數)"""
        pieces, i, holes = [], 0, 0
        while i < len(word):
            match = None
            for ln in range(len(word) - i, 0, -1):
                if word[i:i + ln] in self.keys:
                    match = word[i:i + ln]
                    break
            if match:
                pieces.append(match)
                i += len(match)
            else:
                holes += 1
                i += 1
        return pieces, holes

    def grade(self, word: str) -> str:
        """缺這個詞的時候，虛擬人實際會演成什麼樣。"""
        pieces, holes = self.greedy(word)
        if not pieces:
            return "全無(整詞不動)"
        if holes:
            return "有破洞(一定打錯)"
        if len(pieces) >= 2 and all(len(p) >= 2 for p in pieces):
            return "乾淨複合(多半打得對)"
        return "含單字塊(可疑)"


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_datasets() -> dict[str, list[list[str]]]:
    """回傳 資料集名 → [gloss 序列]。splits 的 output 是 JSON 字串。"""
    data: dict[str, list[list[str]]] = {}
    for name in ["train", "dev", "test", "test_corpus", "test_papers"]:
        rows = load_jsonl(BASE / "data" / "splits_json" / f"{name}.jsonl")
        seqs = []
        for r in rows:
            try:
                gloss = json.loads(r["output"])["gloss"]
            except (KeyError, ValueError):
                continue
            seq = [t for t in re.split(r"\s+", gloss.strip()) if t]
            if seq:
                seqs.append(seq)
        if seqs:
            data[name] = seqs
    for name, rel in [("tslcorpus", "data/tslcorpus/parallel.jsonl"),
                      ("synth", "data/synth/tsl_synth.jsonl")]:
        seqs = [r["gloss"] for r in load_jsonl(BASE / rel) if r.get("gloss")]
        if seqs:
            data[name] = seqs
    return data


def load_fill_sources(twtsl_dir: Path, moe_path: Path) -> tuple[set[str], set[str]]:
    """(本機已下載的 twtsl 中文詞, 教育部辭典有影片的詞)"""
    local: set[str] = set()
    words_csv, videos = twtsl_dir / "dataset" / "words.csv", twtsl_dir / "videos"
    if words_csv.is_file() and videos.is_dir():
        have = {p.name for p in videos.glob("*.mp4")}
        with open(words_csv, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row["中文詞"].strip() and row["影片檔名"].strip() in have:
                    local.add(row["中文詞"].strip())
    moe = set(json.loads(moe_path.read_text(encoding="utf-8"))) if moe_path.is_file() else set()
    return local, moe


def source_of(word: str, local: set[str], moe: set[str]) -> str:
    if word in local:
        return "A 本機已有影片"
    if word in moe:
        return "B 教育部可抓"
    if re.fullmatch(r"[A-Za-z][A-Za-z\s]*", word):
        return "E 外語(待指拼)"
    return "C 需錄製"


TIER_LABEL = {"T0": "T0 原樣就能播", "T1": "T1 去標註記號後可播",
              "T2": "T2 變體後綴別名可播", "T3": "T3 行政區後綴別名可播",
              "T4": "T4 異體字對應後可播", "T5": "T5 數字逐位打(規則)",
              "T6": "T6 大小寫/括號折疊", "T7": "T7 真的缺影片"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    ap.add_argument("--twtsl", type=Path, default=DEFAULT_TWTSL)
    ap.add_argument("--moe", type=Path, default=DEFAULT_MOE)
    ap.add_argument("--out", type=Path, default=BASE / "data" / "video" / "video_gap.json")
    args = ap.parse_args()

    if not args.lexicon.is_file():
        print(f"找不到 {args.lexicon}\n"
              f"先抓下來：scp tku-gpu:"
              f"'~/0813/recordings/lexicon.json' {args.lexicon.parent}/")
        return 1

    lib = Library(json.loads(args.lexicon.read_text(encoding="utf-8")))
    local, moe = load_fill_sources(args.twtsl, args.moe)
    data = load_datasets()
    # 別名鍵會繼承本尊的 source，算進來每個來源都會灌水
    real = [v for v in lib.lex.values() if not v.get("alias_of")]
    src_mix = collections.Counter(str(v.get("source", "(none)")).split(":")[0] for v in real)
    print(f"影片庫：{len(lib.keys)} 個鍵（實體詞條 {len(real)}、別名 "
          f"{len(lib.keys) - len(real)}）—— "
          + "、".join(f"{k} {n}" for k, n in src_mix.most_common()))
    print(f"補片來源：本機 twtsl {len(local)} 詞、教育部辭典 {len(moe)} 詞\n")

    tier_tok: collections.Counter = collections.Counter()
    tier_of: dict[str, str] = {}
    gap: collections.Counter = collections.Counter()
    for seqs in data.values():
        for seq in seqs:
            for g in seq:
                tier, _ = lib.tier(g)
                tier_tok[tier] += 1
                tier_of[g] = tier
                if tier == "T7":
                    n = norm(g)
                    if n:
                        gap[n] += 1

    total = sum(tier_tok.values())
    print("=" * 78)
    print(f"Gloss token 分層（{len(data)} 個資料集、{total} 個 token）")
    print("=" * 78)
    cum = 0
    for t in ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7"]:
        cum += tier_tok[t]
        types = sum(1 for v in tier_of.values() if v == t)
        print(f"  {TIER_LABEL[t]:<24}{tier_tok[t]:>7} token ({tier_tok[t] / total:>5.1%})"
              f"{types:>6} 詞type   累計可播 {cum / total:>5.1%}")

    print("\n" + "=" * 78)
    print("缺詞 × 虛擬人現在會演成什麼 × 影片從哪來（數字＝出現次數）")
    print("=" * 78)
    grades = ["乾淨複合(多半打得對)", "含單字塊(可疑)", "有破洞(一定打錯)", "全無(整詞不動)"]
    srcs = ["A 本機已有影片", "B 教育部可抓", "C 需錄製", "E 外語(待指拼)"]
    grid: collections.Counter = collections.Counter()
    for w, c in gap.items():
        grid[(lib.grade(w), source_of(w, local, moe))] += c
    print(f"{'':<22}" + "".join(f"{s:>15}" for s in srcs) + f"{'小計':>9}")
    for gr in grades:
        row = [grid[(gr, s)] for s in srcs]
        print(f"{gr:<22}" + "".join(f"{v:>15}" for v in row) + f"{sum(row):>9}")
    print(f"{'小計':<22}" + "".join(f"{sum(grid[(g, s)] for g in grades):>15}" for s in srcs)
          + f"{sum(gap.values()):>9}")

    need_rec = [w for w, _ in gap.most_common() if source_of(w, local, moe) == "C 需錄製"]
    print("\n" + "=" * 78)
    print("全句可播率（句中每個 gloss 都演得出來才算；一個詞缺就毀一句）")
    print("=" * 78)

    def ok_exact(g):
        """詞庫鍵精準命中 —— 2026-08-17 之前 composer 的行為。"""
        return lib.tier(g)[0] == "T0"

    def ok_now(g):
        """現在的行為：composer.resolve 會去標註記號，別名鍵已寫進詞庫。"""
        return lib.tier(g)[0] != "T7"

    def ok_moe(g):
        return ok_now(g) or norm(g) in local or norm(g) in moe

    def ok_rec(n):
        top = set(need_rec[:n])
        return lambda g: ok_moe(g) or norm(g) in top

    stages = [("(舊)只認精準命中", ok_exact), ("現況(已部署)", ok_now),
              ("+現成來源補完", ok_moe), ("+錄 TOP200", ok_rec(200)),
              ("+錄 TOP500", ok_rec(500)),
              (f"+錄全部 {len(need_rec)}", ok_rec(len(need_rec)))]
    names = list(data)
    print(f"{'階段':<22}" + "".join(f"{n:>13}" for n in names) + f"{'加權':>10}")
    for label, fn in stages:
        cells, num, den = [], 0, 0
        for n in names:
            seqs = data[n]
            ok = sum(1 for s in seqs if all(fn(g) for g in s))
            cells.append(f"{ok / len(seqs):>12.1%}")
            num, den = num + ok, den + len(seqs)
        print(f"{label:<22}" + "".join(cells) + f"{num / den:>9.1%}")
    print(f"句數：" + "、".join(f"{n} {len(data[n])}" for n in names))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        [{"word": w, "n": c, "grade": lib.grade(w), "source": source_of(w, local, moe),
          "greedy": lib.greedy(w)[0]} for w, c in gap.most_common()],
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n逐詞缺口 → {args.out.relative_to(BASE)}（{len(gap)} 詞）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
