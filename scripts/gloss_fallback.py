#!/usr/bin/env python3
"""表外 Gloss 的退路策略（計畫 5 節 Stage D 的 fallback，先前只寫在計畫未實作）。

**要解決的實測問題**（2026-08-12 三層診斷）：模型遇到沒學過對應 Gloss 的詞時，
會直接把中文貼上去。未見層 45 個表外詞中 27 個屬這類自創／照抄，例如
「看久」「辦畫展」「聰學校」。這對下游是致命的——動作庫查不到就無法播放。

**判定基準用詞彙總表（7,002 詞）而非訓練詞彙**。拿訓練詞彙當基準會把
「模型產出了合法但訓練樣本內沒出現的手語詞」（如「畫家」「幼稚園」）誤判為
造詞——實測未見層 45 個表外詞中有 18 個（40%）屬此，會嚴重高估問題規模。

**設計原則：錯的手語比指拼更糟。**
指拼是臺灣手語本有的借用詞／字形詞機制（見計畫 5 節 Stage D、來源 7），
下游收到指拼標記知道怎麼處理；收到一個語意錯誤的 Gloss 則會播出錯誤內容，
而且錯誤被藏在看似正常的輸出裡難以追蹤。故所有修復規則都設保守門檻，
寧可標指拼也不猜。

**修復順序**（先命中者優先；門檻皆為避免實測到的誤修）：
  1. 總表內       → 原樣保留
  2. 正規化後命中 → 用正規化形式（臺→台、去 ++ 重複記號、去轉寫後綴）
  3. 唯一延伸詞   → 「中秋」→「中秋節」。限「唯一候選且只多 1 個字」：
                    放寬到 2 字會產生「社區」→「社區大學」、「扭動」→「扭動脖子」
                    這種語意跑掉的誤修（實測）。
  4. 可切分複合詞 → 「聰學校」→「學校」。限「每段至少 2 字」：
                    允許單字段會產生「選手」→「選 手」（選擇＋手）這種語意崩壞。
  5. 內含已知詞   → 「閃光+警報器」→「警報器」。限被包含詞至少 2 字且
                    長度佔原詞一半以上，否則等於用一個片段代表整個詞。
  6. 都不行       → 標記為指拼，交下游處理。

**刻意不做「刪除中文虛詞」**：初版曾用手工列的虛詞表（剛好、通常…），
但那份表是直接看未見層測試結果列出來的，等於拿測試集調參，在同一批
重測會虛高。改用訓練資料推導亦失敗——字元 n-gram 只能得到「要去」「天我」
這類無意義碎片，且目標詞（概念、樂意、安定）在訓練中文裡出現 0 次，
無從判定。故此規則移除，該類詞一律走指拼。
"""
import json
import re
from functools import lru_cache
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
MASTER = BASE / "data" / "vocab" / "gloss_master.jsonl"

FINGERSPELL_PREFIX = "指拼:"
MAX_EXTEND_CHARS = 1      # 延伸詞最多只能多這麼多字（見檔頭規則 3）
MIN_PART_CHARS = 2        # 切分後每段最少字數（見檔頭規則 4）
MIN_CONTAINED_RATIO = 0.5  # 被包含詞至少要佔原詞的比例（見檔頭規則 5）


def normalize(token):
    """與 metrics.normalize_gloss 一致的正規化。"""
    t = str(token).replace("臺", "台")
    t = re.sub(r"\+\+.*$", "", t)
    t = re.sub(r"_[A-Za-z]+$", "", t)
    t = re.sub(r"[（(][^）)]*[）)]", "", t)
    return t.strip(" ?？。，、!！") or str(token)


@lru_cache(maxsize=1)
def load_vocab(master_path=None):
    """回傳 (全部詞, 可播放詞)。可播放＝有辭典詞條，下游動作庫查得到。"""
    path = Path(master_path) if master_path else MASTER
    all_v, rend = set(), set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        all_v.add(r["surface"])
        if r.get("has_dict_entry") or r.get("norm_has_dict_entry"):
            rend.add(r["surface"])
    return frozenset(all_v), frozenset(rend)


@lru_cache(maxsize=4096)
def _unique_extension(token, vocab):
    """總表中唯一包含此 token、且只多 MAX_EXTEND_CHARS 字的詞。"""
    cands = [w for w in vocab
             if token in w and w != token and len(w) - len(token) <= MAX_EXTEND_CHARS]
    return cands[0] if len(cands) == 1 else None


@lru_cache(maxsize=4096)
def _best_contained(token, vocab):
    """總表中被此 token 包含、且長度佔比達門檻的最長詞。"""
    cands = [w for w in vocab
             if w in token and w != token and len(w) >= 2
             and len(w) / len(token) >= MIN_CONTAINED_RATIO]
    return max(cands, key=len) if cands else None


def _split(token, vocab, max_parts=3):
    """切成總表內的片段，每段至少 MIN_PART_CHARS 字。找不到回 None。"""
    n = len(token)
    if n < MIN_PART_CHARS * 2:
        return None
    dp = [None] * (n + 1)
    dp[0] = []
    for i in range(MIN_PART_CHARS, n + 1):
        for j in range(0, i - MIN_PART_CHARS + 1):
            if dp[j] is None or len(dp[j]) >= max_parts:
                continue
            piece = token[j:i]
            if piece in vocab and (dp[i] is None or len(dp[j]) + 1 < len(dp[i])):
                dp[i] = dp[j] + [piece]
    return dp[n]


def repair_token(token, vocab=None, renderable=None):
    """修復單一 Gloss 詞，回傳 (修復後詞串列, 採用的規則名稱)。"""
    if vocab is None:
        vocab, renderable = load_vocab()
    if token in vocab:
        return [token], "keep"
    norm = normalize(token)
    if norm in vocab:
        return [norm], "normalize"
    ext = _unique_extension(norm, vocab)
    if ext:
        return [ext], "extend"
    parts = _split(norm, vocab)
    if parts and len(parts) > 1:
        return parts, "split"
    cont = _best_contained(norm, vocab)
    if cont:
        return [cont], "contained"
    return [f"{FINGERSPELL_PREFIX}{norm}"], "fingerspell"


def repair_gloss(gloss_text, vocab=None, renderable=None):
    """修復整句 Gloss，回傳 (修復後字串, 每條規則的使用次數)。"""
    if vocab is None:
        vocab, renderable = load_vocab()
    toks = [t for t in str(gloss_text or "").replace("／", "/").replace("/", " ").split() if t]
    out, stats = [], {}
    for t in toks:
        fixed, rule = repair_token(t, vocab, renderable)
        out.extend(fixed)
        stats[rule] = stats.get(rule, 0) + 1
    return " ".join(out), stats
