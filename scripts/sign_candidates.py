#!/usr/bin/env python3
"""候選手語檢索：從中文句子撈出「可能用得到的 sign_id」清單。

為什麼是這支決定成敗（2026-08-20，教授指定的訓練資料格式）：
    新格式要模型「只能從候選清單挑 sign_id」。候選清單怎麼來，
    **訓練與推論必須用同一支程式**——訓練時若用「拿正解回頭湊candidates」
    的作弊法，上線時無正解可用、候選分布完全不同，模型學到的約束就失效。
    故本模組只吃中文句子，不吃正解，離線建資料與線上服務共用。

檢索方法（沿用 rag_retrieve.py 的原則：不裝額外依賴、共用機能跑）：
    1. 字面命中  句子裡直接出現的詞（「今天天氣很好」→ 今天／天氣／好）。
                 中文無空白分詞，改以動作庫的詞當詞典做最長匹配掃描，
                 等價於用 16,628 詞的詞典斷詞，精準度最高。
    2. 例句遷移  用既有 Retriever 找相似訓練句，把那些句子用到的手語一起放進來。
                 這能撈到字面撈不到的對應（「走走」→ 散步、「想去」→ 想）。
    3. 字元重疊  與句子共用字元的詞（今天→今年、天氣→氣）。純為湊出干擾項，
                 讓模型必須真的判斷而不是照抄唯一候選。

干擾項是刻意的：教授範例裡的 現在／熱／公園／游泳 都不是答案，
但少了它們，候選清單＝答案清單，模型不必學選擇，直接背誦即可。

用法：
    python3 scripts/sign_candidates.py "今天天氣很好，我想去海邊走走。"
    from sign_candidates import CandidateRetriever
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_video_coverage import fold, norm  # noqa: E402  複用既有 gloss 正規化

BASE = Path(__file__).resolve().parent.parent
INVENTORY = BASE / "data" / "signs" / "sign_inventory.jsonl"
INDEX = BASE / "data" / "signs" / "gloss_to_sign.json"
SYNONYMS = BASE / "data" / "signs" / "synonym_groups.json"

_PUNCT = "，。？！?!,.、;；:：…「」『』（）()《》〈〉 　\n\t"
_STOP = {"的", "了", "是", "在", "и"}   # 純虛詞，撈進候選只是雜訊


def strip_punct(text: str) -> str:
    return "".join(c for c in str(text) if c not in _PUNCT)


class CandidateRetriever:
    """從中文句子檢索候選手語。載入一次重複使用（線上服務常駐）。"""

    # 影片品質判定為「演不出動作」的 asset_class。判定來源見
    # build_sign_inventory.TIER_TO_CLASS（0813 quality_scan 的 tier）。
    UNUSABLE_CLASSES = {"unusable_quality"}

    def __init__(self, inventory: Path = INVENTORY, index: Path = INDEX,
                 use_examples: bool = True, synonyms: Path = SYNONYMS,
                 exclude_unusable: bool = True):
        all_rows = [json.loads(l) for l in
                    inventory.read_text(encoding="utf-8").splitlines() if l.strip()]
        # 已驗證的重複收錄（build_sign_inventory.DUPLICATE_OF）不進候選：
        # 同一支錄影出現兩次只會佔掉候選名額，還讓模型面對兩個等價選項。
        # ID 不刪除，故 by_id 仍收錄全部，舊資料引用得到才解析得出來。
        # superseded_by＝同詞有更準確的另一支影片（品質實測後選定），同樣不進候選
        self.rows = [r for r in all_rows
                     if "duplicate_of" not in r and "superseded_by" not in r]

        # 2026-08-31（教授審查意見 4.1）：品質判定為 severe／no_hands_raised 的
        # 影片不進候選。「總表有這個 ID」不等於「虛擬人演得出這個手語」——
        # 動作庫 17,085 支裡有 39.8% 是 severe（幾乎整段偵測不到舉手動作），
        # 讓它們留在候選等於允許模型輸出一個播出來沒有動作的腳本。
        #
        # 實測代價（train，2026-08-31）：參考 token 只有 2.35% 落在這類資產上，
        # 但 11.5% 的句子至少含一個——那些句子的 candidate_coverage_risk 會
        # 轉成 true。這不是變差，是本來就該標出來的缺口，之前被品質常數蓋掉了。
        # ⚠️ 開關預設為 True 會改變候選分布＝改變訓練資料，**必須重訓才生效**。
        # 要重建 v17 的候選請傳 exclude_unusable=False。
        self.exclude_unusable = exclude_unusable
        self.excluded_unusable = 0
        if exclude_unusable:
            before = len(self.rows)
            self.rows = [r for r in self.rows
                         if r.get("asset_class") not in self.UNUSABLE_CLASSES]
            self.excluded_unusable = before - len(self.rows)
        self.by_gloss = {r["gloss"]: r for r in self.rows}
        self.index: dict[str, str] = json.loads(index.read_text(encoding="utf-8"))
        self.by_id = {r["sign_id"]: r for r in all_rows}
        # 折疊後詞形 → sign_id：外語詞的大小寫與內部空白不是詞的一部分。
        # 語料寫 `BB call`／`QR code`，總表的鍵是 `BBCall`／`QR Code`，
        # 不折疊就會把庫裡明明有影片的詞判成 OOV，訓練標的被寫成
        # needs_review=true（實測 splits_script 有 8 句受害）。
        # 只折含 ASCII 字母的鍵；中文鍵去空白會併掉帶空白的重複收錄。
        self._folded: dict[str, str] = {}
        for gloss, sid in sorted(self.index.items()):
            f = fold(gloss)
            if f and re.search(r"[a-z]", f):
                self._folded.setdefault(f, sid)
        self.max_len = max(len(g) for g in self.by_gloss)
        # 字元 → 含該字元的 gloss，供干擾項檢索
        self.by_char: dict[str, list[str]] = {}
        for g in self.by_gloss:
            for ch in set(g):
                self.by_char.setdefault(ch, []).append(g)

        # 清理後詞形 → 動作庫原鍵：同義表用 gloss_clean，by_gloss 用原鍵，
        # 有 45 筆原鍵帶空白，不轉換這一層會白白漏掉
        self._clean_to_raw: dict[str, str] = {}
        for r in self.rows:
            self._clean_to_raw.setdefault(r["gloss_clean"], r["gloss"])

        self._syn: dict[str, list[str]] = {}
        if synonyms and Path(synonyms).exists():
            self._load_synonyms(Path(synonyms))

        self._ex_rows, self._ex_bg = [], []
        self._core: list[str] = []
        if use_examples:
            self._load_examples()

    def _load_synonyms(self, path: Path) -> None:
        """載入同義詞組（scripts/build_synonym_groups.py）。

        兩種來源的可信度不同，展開順序照可信度排：same_clip（同一支錄影同一
        區段，按定義就是同一個動作）優先於 dict_alias（辭典說可以用同一個手語
        打，但中文不見得全等，如 成→完成／成功）。k 截斷時先保住強的那些。
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        rank = {"same_clip": 0, "dict_alias": 1, "moe_alias": 2}
        by_id = {g["id"]: g for g in data.get("groups", [])}
        pairs: dict[str, list[tuple[int, str]]] = {}
        for gid, group in by_id.items():
            r = rank.get(group["source"], 9)
            members = group["members"]
            for m in members:
                for other in members:
                    if other != m:
                        pairs.setdefault(m, []).append((r, other))
        for m, lst in pairs.items():
            out, seen = [], set()
            for _, other in sorted(lst, key=lambda x: x[0]):
                raw = other if other in self.by_gloss else self._clean_to_raw.get(other)
                if raw and raw not in seen:
                    seen.add(raw)
                    out.append(raw)
            key = m if m in self.by_gloss else self._clean_to_raw.get(m, m)
            if out:
                self._syn[key] = out

    def _load_examples(self) -> None:
        """載入訓練句供「例句遷移」與詞對齊表；缺檔不致命，退化成純字面檢索。"""
        path = BASE / "data" / "splits" / "train.jsonl"
        if not path.exists():
            return
        self._all_ex = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
                        if l.strip()]
        self.refit_examples()

    # 候選生成的完整參數集。**訓練、離線評估、線上服務必須用同一組**，
    # 否則模型看到的候選分布與上線時不同（training-serving skew）。
    # 教授審查意見 2.1：v17 訓練用 n_sem=8，線上服務沒載向量模型、實際
    # 走 n_sem=0，兩邊每句約 8 個候選相異。
    CONFIG_KEYS = ("k", "n_examples", "distractor_ratio", "n_align", "n_core",
                   "n_syn", "n_sem", "pin_core", "exclude_unusable")

    def config(self, **overrides) -> dict:
        """回傳這個 retriever 實際會用的候選參數（含 candidates() 的預設值）。

        寫進資料集的 candidate_config.json，也供服務端與部署 checkpoint 對帳。
        """
        import inspect
        defaults = {k: v.default for k, v in
                    inspect.signature(self.candidates).parameters.items()
                    if v.default is not inspect.Parameter.empty}
        cfg = {k: defaults.get(k) for k in self.CONFIG_KEYS}
        cfg["exclude_unusable"] = self.exclude_unusable
        cfg["semantic_loaded"] = getattr(self, "semantic", None) is not None
        cfg.update({k: v for k, v in overrides.items() if k in self.CONFIG_KEYS})
        return cfg

    def has_examples(self) -> bool:
        """有沒有載到訓練句（沒有的話 cross-fitting 沒意義）。"""
        return bool(getattr(self, "_all_ex", None))

    def refit_examples(self, exclude_groups: set | None = None,
                       exclude_ids: set | None = None) -> int:
        """重建三張由訓練句衍生的表，可排除指定的資料。回傳實際採用的句數。

        2026-08-31（教授審查意見 2.2）。這三張表——例句遷移的 `_ex_rows`、
        詞對齊表 `_align`、高頻核心詞 `_core`——原本一律用**完整 train** 建，
        於是替 train 句產生候選時，表裡already含有那句自己的答案：

          - `_align` 數的是「中文片段 c 出現時 gloss g 也出現」的共現，
            該句自己的 (c, g) 對就在裡面
          - `_core` 是 train 全體 gloss 的前 30 高頻，該句也投了票

        `candidates(exclude_id=...)` 只擋掉例句遷移把同一句撈回來，擋不到
        這兩張統計表。教授抽 20 筆做 leave-group-out 實測：詞涵蓋率
        92.86%→88.10%、整句可拼出 80%→70%，7/20 筆候選集合改變——
        不是理論上的疑慮。

        **排除的單位是 group 不是 id**：長度平衡會把同一句複製 2–4 份，
        只排 id 會讓副本留在表裡；而且同一段對話／同一個詞條底下的句子
        高度相似，只排自己等於沒排。用 group 才是真正的 leave-group-out。
        """
        rows = getattr(self, "_all_ex", None)
        if rows is None:
            return 0
        self._ex_rows, self._ex_bg = [], []
        for r in rows:
            if not (r.get("chinese") and r.get("gloss_text")):
                continue
            if exclude_groups and r.get("group") in exclude_groups:
                continue
            if exclude_ids and r.get("id") in exclude_ids:
                continue
            self._ex_rows.append(r)
            self._ex_bg.append(self._bigrams(r["chinese"]))
        self._build_align()
        self._build_core()
        return len(self._ex_rows)

    def _build_core(self, size: int = 30) -> None:
        """高頻核心手語，每題都放進候選。

        為什麼不能只靠檢索（2026-08-20 實測）：自然手語的 這／那／他／我
        是**空間指示與代名詞**，屬語法機制不是中文詞的對譯。實例：

            往西邊走會到公園  →  西/那/走/那/公園/那
            他推我下去池子裡  →  池/這/池/他/我/他/我/他/推

        中文句面完全沒有「那」字，任何以中文為輸入的檢索都撈不到，
        但它們是 dev 集最常缺的詞（這 285、什麼 220、那 181 次）。
        手語者隨時都能比指示與代名詞，候選清單也就該隨時備著。
        """
        freq: collections.Counter = collections.Counter()
        for r in self._ex_rows:
            for t in str(r["gloss_text"]).split("/"):
                t = t.strip()
                if t in self.by_gloss:
                    freq[t] += 1
        self._core = [g for g, _ in freq.most_common(size)]

    def _build_align(self) -> None:
        """從平行語料挖「中文片段 → gloss」對齊表。

        為什麼非有不可：自然手語大量使用**中文句面上沒有的詞**。
        「我會注意政見發表的內容」的 gloss 是 政見/演講/什麼/我/注意/會——
        「什麼」是話題引入標記，中文裡根本沒這兩個字，字面掃描永遠撈不到。
        同類還有 這／那／他／有／是，實測是 dev 集最常見的漏撈詞。

        作法是最陽春的共現統計（等價於 IBM Model 1 的詞彙翻譯機率）：
        數「中文出現片段 c 時，gloss 出現 g」的次數，再除以 g 的總頻次抑制
        高頻詞洗版。不需外部套件，建表數秒。
        """
        co: dict[str, collections.Counter] = {}
        gfreq: collections.Counter = collections.Counter()
        for r in self._ex_rows:
            toks = {t.strip() for t in str(r["gloss_text"]).split("/") if t.strip()}
            toks = {t for t in toks if t in self.by_gloss}
            if not toks:
                continue
            gfreq.update(toks)
            t = strip_punct(r["chinese"])
            frags = set(t) | {t[i:i + 2] for i in range(len(t) - 1)}
            for c in frags:
                bucket = co.setdefault(c, collections.Counter())
                bucket.update(toks)
        # 轉成 條件機率 × 稀有度加權，只留每個片段最強的幾個對應
        self._align: dict[str, list[tuple[str, float]]] = {}
        for c, bucket in co.items():
            total = sum(bucket.values())
            if total < 2:
                continue
            scored = [(g, (n / total) * (1.0 / (1.0 + gfreq[g] ** 0.5)))
                      for g, n in bucket.items() if n >= 2]
            # 同分時**必須**用 gloss 當第二鍵：bucket 的鍵序來自上面 `for c in frags`
            # 迭代 set 的順序，而那個順序隨 PYTHONHASHSEED 每個行程都不同。只按分數
            # 排會讓同分者保持那個任意順序，_align 表因此每次跑都不一樣——2026-08-30
            # 實測同一條建資料指令連跑兩次，40 句裡有 2 句候選清單不同。
            scored.sort(key=lambda x: (-x[1], x[0]))
            if scored:
                self._align[c] = scored[:8]

    @staticmethod
    def _bigrams(text: str) -> set[str]:
        t = strip_punct(text)
        return {t[i:i + 2] for i in range(len(t) - 1)} or {t}

    def resolve(self, gloss: str) -> str | None:
        """gloss 寫法 → sign_id；解不到回 None（＝動作庫演不出來）。

        必須先過 norm()：語料庫的 gloss 帶標註記號（++ 重複、+X 複合、
        (X) 註記），`買++` 不是一個新手語而是「買」打兩次，動作庫只會有「買」。
        不正規化就查，會把一堆演得出來的詞誤判成缺片（實測 dev 集因此
        虛報 43 個缺口）。
        """
        g = str(gloss).strip()
        for form in (g, strip_punct(g), norm(g)):
            if form and form in self.index:
                return self.index[form]
        for form in (g, norm(g)):                 # 外語：折大小寫與內部空白
            if form and re.search(r"[A-Za-z]", form):
                sid = self._folded.get(fold(form))
                if sid:
                    return sid
        return None

    def _literal(self, text: str) -> list[str]:
        """撈出句子裡字面出現的詞——**所有**匹配，不是最長匹配切分。

        為什麼不能用最長匹配（2026-08-20 實測）：切分會把短詞吃掉，
        「這個東西」只切出 這個／東西，但語料的正解 gloss 是「這」；
        「有沒有」切成一塊，「有」就進不了候選。實測 train 集最常漏撈的
        這(285)／什麼(220)／那(181)／有(144) 全是被這樣吃掉的，
        它們在動作庫裡都有影片，純粹是查詢端的錯。

        候選清單本來就該是**選項**而非切分結果，長短詞並陳交給模型判斷才對。
        """
        t = strip_punct(text)
        hits: list[str] = []
        seen: set[str] = set()
        for i in range(len(t)):
            for ln in range(min(self.max_len, len(t) - i), 0, -1):
                chunk = t[i:i + ln]
                if chunk in self.by_gloss and chunk not in _STOP and chunk not in seen:
                    seen.add(chunk)
                    hits.append(chunk)
        # 長詞優先：較具體的對應排前面，k 截斷時先保住資訊量大的
        hits.sort(key=lambda g: (-len(g), t.find(g)))
        return hits

    def _from_examples(self, text: str, k: int, exclude_id=None) -> list[str]:
        """相似訓練句用到的手語。

        exclude_id 是**建訓練資料時必給**的：不排掉句子自己，檢索會把正解
        原封不動撈回候選，模型只要照抄就滿分，上線卻無此捷徑（實測 train
        涵蓋率因此虛高到 96.8%，dev 只有 73.4%）。
        """
        if not self._ex_rows:
            return []
        q = self._bigrams(text)
        scored = []
        for r, bg in zip(self._ex_rows, self._ex_bg):
            if exclude_id is not None and r.get("id") == exclude_id:
                continue
            inter = len(q & bg)
            if inter:
                scored.append((inter / len(q | bg), r))
        scored.sort(key=lambda x: -x[0])
        out = []
        for _, r in scored[:k]:
            for g in str(r["gloss_text"]).split("/"):
                g = g.strip()
                if g and g in self.by_gloss:
                    out.append(g)
        return out

    def _from_align(self, text: str, want: int) -> list[str]:
        """詞對齊表：撈出中文句面沒有、但語料顯示常一起出現的手語。"""
        if not getattr(self, "_align", None):
            return []
        t = strip_punct(text)
        frags = set(t) | {t[i:i + 2] for i in range(len(t) - 1)}
        agg: dict[str, float] = {}
        for c in frags:
            for g, s in self._align.get(c, ()):
                agg[g] = agg.get(g, 0.0) + s
        return sorted(agg, key=lambda g: (-agg[g], g))[:want]

    def _from_synonyms(self, seeds: list[str], want: int) -> list[str]:
        """把已命中的詞展開成同義詞。

        為什麼需要（2026-08-21 實測）：總表是以**中文詞面**編 ID 的，
        16,623 個詞面背後只有 15,354 個不同動作。句子寫「孩子」，
        動作庫裡那支叫「小孩」，字面檢索就撈不到——但它們是同一個手語。
        **實測是淨負的，故預設關閉**：train 92.6%→90.7%、dev 僅 +0.1pp。
        固定 k 之下候選是零和的，加同義詞就得擠掉別的；種子若放寬到例句與
        對齊通道的結果，再掉 0.7pp。故種子只取**多字的字面命中**——
        它們最可靠，其同義詞也最接近等價。

        **不會洩漏正解**：展開來源是動作庫與辭典，與該句的參考答案無關，
        上線時同樣拿得到（對比 _from_examples 必須用 exclude_id 留一）。
        """
        if not self._syn:
            return []
        out: list[str] = []
        for g in seeds:
            for other in self._syn.get(g, ()):
                if other not in out:
                    out.append(other)
                if len(out) >= want:
                    return out
        return out

    def _distractors(self, text: str, chosen: set[str], want: int) -> list[str]:
        """共用字元的詞，依重疊比排序。純為增加選擇難度。"""
        t = set(strip_punct(text))
        scored: dict[str, float] = {}
        for ch in t:
            for g in self.by_char.get(ch, ()):
                if g in chosen or g in _STOP:
                    continue
                scored[g] = max(scored.get(g, 0), len(set(g) & t) / len(set(g)))
        ranked = sorted(scored, key=lambda g: (-scored[g], len(g), g))
        return ranked[:want]

    # n_examples 預設 8 是實測結果，不是猜的：教授範例句「今天天氣很好…」
    # 的關鍵對應「天氣很好→晴朗」出現在相似度第 4 名的訓練句，top-3 撈不到。
    def candidates(self, text: str, k: int = 20, n_examples: int = 8,
                   distractor_ratio: float = 0.2, n_align: int = 12,
                   n_core: int = 30, exclude_id=None, n_syn: int = 0,
                   n_sem: int = 0, pin_core: int = 0) -> list[dict]:
        """回傳 [{sign_id, gloss}]，長度上限 k。順序穩定（可重現）。

        五個通道依序填：字面命中 → 例句遷移 → 詞對齊 → 高頻核心 →
        字元重疊干擾項。前四個負責把正解撈進來，最後一個負責讓題目不會只有答案。

        **n_syn 預設 0＝同義展開關閉**。通道本身已實作可用（設 n_syn>0 啟用，
        插在多字字面命中之後），但實測是淨負的：train 詞涵蓋率 92.6%→90.7%
        （−1.9pp），dev 只有 +0.1pp。原因是固定 k 之下候選是零和的——
        train 的例句遷移（留一法後仍很準）本來就撈得到正解，12 個同義名額
        把那些高價值候選擠掉了；dev 的例句遷移較不準，擠掉的損失才較小。
        留著是因為表與通道有其他用途（見 build_synonym_groups.py），
        且 k 若拉高到候選不再稀缺時值得重測。

        **n_sem 預設 0＝語義通道關閉**（2026-08-23）。啟用需先設
        `self.semantic = semantic_channel.SemanticRanker(self.rows)`，
        建資料端（Mac）才裝得起向量模型；線上服務不載入、行為不變。
        插在詞對齊之後、核心詞之前：它撈的是「學到了→學習」這類語義對應，
        價值高於核心詞的尾端（dev 實測核心第 21–30 名只命中 3.1% token）。
        **v17 的訓練資料（splits_script_k40sem）就是用這個通道建的**，
        拿掉它就無法重新產生 v17 的訓練集——這是它留著的主要理由。

        **pin_core 預設 0＝關閉，實測淨負，留著只為記錄這個否定結果。**
        動機：2026-08-27 的診斷發現高頻核心排在第五順位，k=40 時經常整批被
        `ordered[:k]` 截掉——而漏檢名單的頭部（這／什麼／有／是／那／再／他）
        全都在 core-30 裡，且這些詞在別句撈得到（「這」出現在 6,359 句的候選
        清單裡卻在 18 句漏掉）。看起來像純排序問題。

        但實測是**單調變差**的（k=40，參考詞落在候選內的總數）：

            pin_core      0     5    10    15    20
            dev         828   -1    -3    -6   -11
            corpus      848   -6   -12   -21   -37
            textbook    936   -2    -3    -9   -17

        原因與 n_syn 完全相同——**固定 k 之下候選是零和的**。當初的估算只算了
        「釘進來能回收多少 OOV」，沒算「被擠掉的例句遷移／詞對齊本來撈到多少
        別的參考詞」，而後者比前者大。這已經是同一個教訓的第三次（同義通道、
        語義通道、核心保底），三次都指向同一個結論：k=40 的候選組合已經在
        局部最佳，靠「多塞一個通道」沒有出路，只能走「撈寬再重排壓窄」。
        """
        ordered: list[str] = []
        seen: set[str] = set()

        def add(glosses):
            for g in glosses:
                if g not in seen and g in self.by_gloss:
                    seen.add(g)
                    ordered.append(g)

        # 字面命中含大量單字（這／個／片），把它們排在同義展開之前會用
        # 「片」這種碎片擠掉「小孩」。故先放多字命中，再放它們的同義詞，
        # 單字命中殿後——與 _literal 內部「長詞優先」是同一個道理。
        literal = self._literal(text)
        multi = [g for g in literal if len(g) > 1]
        add(multi)
        if n_syn:
            add(self._from_synonyms(multi, n_syn))
        if pin_core:
            # 保底名額：插在多字命中之後、其餘通道之前，確保不會被 [:k] 截掉。
            add(self._core[:pin_core])
        add([g for g in literal if len(g) == 1])
        add(self._from_examples(text, n_examples, exclude_id=exclude_id))
        add(self._from_align(text, n_align))
        if n_sem and getattr(self, "semantic", None) is not None:
            got = 0
            for g in self.semantic.rank(text):
                if got >= n_sem:
                    break
                if g not in seen and g in self.by_gloss:
                    seen.add(g)
                    ordered.append(g)
                    got += 1
        add(self._core[:n_core])
        room = max(0, k - len(ordered))
        if room:
            add(self._distractors(text, seen, min(room, max(1, int(k * distractor_ratio)))))
        return [{"sign_id": self.by_gloss[g]["sign_id"], "gloss": g}
                for g in ordered[:k]]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    r = CandidateRetriever()
    text = sys.argv[1]
    cands = r.candidates(text)
    print(f"輸入：{text}")
    print(f"候選 {len(cands)}：")
    for c in cands:
        print(f"  {c['sign_id']}  {c['gloss']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
