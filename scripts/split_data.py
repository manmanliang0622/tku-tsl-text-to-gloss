#!/usr/bin/env python3
"""第一階段 3.4：train/dev/test 切分。

實驗設計（本計畫決策，對應計畫 6.4 對照組）：
  - **test = 33 句真實已審核句**（tsl_sentences.jsonl 排除模板句 S24/S26），
    與 Stage A 提示法基線用的評測集完全相同 → 微調結果可與基線直接比較，
    且這批真實句永不進訓練集（無洩漏）。
  - **train/dev = 合成句（synth）＋ 中正辭典例句（twtsl）**。
    dev 由訓練池分層抽樣（依 source/confidence），供 early stopping。

洩漏防護（兩層）：
  1. train/dev 中若有句子的中文或 gloss_text 與 test 完全相同即剔除。
     （句型會與 test 重疊——這正是要測的泛化；只剔除「完全相同的句」。）
  2. train↔dev 去洩漏（2026-07-23）：dev 依 group 整組留存，同一段對話／詞條／
     模板不會同時落在 train 與 dev，避免同源近似句造成 dev 分數虛高、影響 early
     stopping。manifest 的 dev_group_leakage 應為 0。

資料品質標記：synth 與 twtsl 目前 review_status=pending（未經本團隊人工審核）。
本切分產出的 manifest.json 會記錄各來源筆數與審核狀態，供報告據實說明。

用法：
  python3 scripts/split_data.py --use-teacher-reviewed --corpus-test-ratio 0.12 --seed 42
                                                    # Stage B v4：教師審核資料＋584句真實test
  python3 scripts/split_data.py                       # 預設：排除 rule-derived，只用
                                                      #   attested/corpus 合成 + twtsl + 語料庫
  python3 scripts/split_data.py --include-rule-derived # 納回 rule-derived（僅供實驗，非正式訓練）
  python3 scripts/split_data.py --no-corpus            # 不加文化部語料庫（回到舊組成）
  python3 scripts/split_data.py --include-words 500    # 額外加 N 筆辭典詞→gloss 對

2026-07-23 更新：依全資料審核，rule-derived 合成句未經母語者逐句裁定，**預設排除**。
需納入時明確加 --include-rule-derived，且結果只能作管線驗證、不得作正式訓練報告依據。

2026-07-21 更新：train/dev 池加入文化部《臺灣手語語料庫》全爬平行語料
（data/tslcorpus/parallel.jsonl，5,272 句真實 Text↔Gloss）。這是最大宗真實資料，
預設納入；以 --min-gloss-len 過濾過短碎片（是／爺爺 等單詞句）。
"""
import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# ── 表面形式正規化（2026-08-31，教授審查意見 2.4）────────────────────────
# 原本的去洩漏與去重都比對**原始中文字串**，所以只差一個句號的兩句會被當成
# 不同句。實測後果：核心 33 句有 3 句（我住在台北。／我知道／我不知道）的
# 去標點形式出現在 train；dev 與 train 更有 6 句原字串就完全相同——因為去重
# 鍵是 (chinese, gloss_text)，同一句中文配不同 Gloss 就兩邊都留。dev 同時
# 決定 checkpoint 選擇與 needs_review 門檻，那個洩漏比 test 的還直接。
_PUNCT = "，。？！?!,.、;；:：…「」『』（）()《》〈〉“”\"' 　\t"
# 異體字與臺／台：同一個詞的兩種寫法，不是兩個詞。
_VARIANT = str.maketrans({"臺": "台", "妳": "你", "祂": "他", "牠": "他"})


def normalize_text(s: str) -> str:
    """比對用的中文正規形。全半形統一 → 異體字折疊 → 去標點與空白。

    只用於「這兩句算不算同一句」的判斷，**不改寫實際寫進切分檔的內容**——
    語料的原始寫法要保留，正規化只是比對的鏡片。
    """
    s = unicodedata.normalize("NFKC", str(s))
    s = s.translate(_VARIANT)
    return re.sub(f"[{re.escape(_PUNCT)}]", "", s).strip()


def normalize_gloss(s: str) -> str:
    """Gloss 的比對用正規形：逐 token 折疊異體字，去空白。"""
    toks = [t.strip().translate(_VARIANT) for t in str(s).split("/")]
    return "/".join(t for t in toks if t)

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT = DATA / "splits"
CORPUS_TEST_REVIEW = OUT / "test_corpus_teacher_review_2026-07-24.json"


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids_sha256(rows):
    payload = "".join(f"{e['id']}\n" for e in rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_corpus_test_review(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records", [])
    by_id = {}
    for row in records:
        rid = row.get("id")
        if not rid or rid in by_id:
            raise ValueError(f"test review sidecar ID 缺失或重複：{rid!r}")
        by_id[rid] = row
    declared = payload.get("counts", {})
    eligible = sum(bool(r.get("teacher_test_eligible")) for r in records)
    if declared != {
        "candidate": len(records),
        "eligible": eligible,
        "excluded": len(records) - eligible,
    }:
        raise ValueError("test review sidecar counts 與 records 不一致")
    return payload, by_id


def group_key(e, source):
    """去洩漏切分的分組鍵：同組資料只會整組落在 train 或 dev，不跨兩邊。

    語料庫按對話分組（seg_uuid 去掉段落號 P\\d+，讓同一段對話的各段落／句子
    同進退）；辭典例句按詞條；合成句按模板；其餘每筆自成一組。
    """
    if source == "tslcorpus":
        u = e.get("seg_uuid") or (e.get("corpus_id", "").split("/")[0]) or e["id"]
        return "corpus:" + re.sub(r"P\d+$", "", u)
    if source == "twtsl-sentence":
        return "twtsl:" + str(e.get("headword") or e["id"])
    if source == "synth":
        return "synth:" + str(e.get("template_id") or e["id"])
    if source == "paper":
        return "paper:" + str(e.get("paper") or e["id"])
    if source == "correction":
        return "correction:" + str(e["id"])
    return f"{source}:{e['id']}"


def human_excluded(e):
    """2026-08-21 人工校訂判定「排除」者（scripts/apply_corpus_review.py 寫入）。

    這是資料可用性判定（語助詞、只是表情、人名、無法由現有資訊修正），
    與 --use-all 那種「信度政策閘門」不同層次，故預設一律排除；
    要重現舊切分請加 --keep-excluded。
    """
    return e.get("train_eligible") is False


def norm_record(e, split_source):
    """統一訓練用欄位。"""
    return {
        "id": e["id"],
        "chinese": e["chinese"],
        "gloss_text": e["gloss_text"],
        "source": split_source,
        "group": group_key(e, split_source),
        "confidence": e.get("confidence"),
        "review_status": e.get("review_status", "n/a"),
        # 保留 NMS（非手部標記）：JSON 目標格式的 nonmanual 欄位需要，
        # 且下游虛擬人的表情同步也要用（計畫第 1 節）。
        "nms": e.get("nms"),
        # 子句切分（2026-08-31）：只有中正辭典例句有這個欄位，語料庫與合成句
        # 都沒有。**保留的是子句的 token 數，不是字串**——2026-08-21 的人工校訂
        # 只更新 gloss_text，clauses 與 gloss_raw 都沒跟著改，10 筆 -corrected
        # 的內容已經對不起來。下游 build_script_dataset.clause_breaks() 因此
        # 只採計數並對 gloss_text 驗總數，對不上就退回空陣列，不猜。
        "clauses": e.get("clauses") or None,
        # 上下文（SCOPE 路線）：同段落的前文中文句，由 attach_context() 填入。
        # 依據：留存測試集實測 22.3% 的參考 Gloss token 無法從中文字面推得，
        # 需前文才能還原（例：「聾人和重聽者能看到手語翻譯和字幕」的參考含
        # 「電視」，該詞來自前一句）。
        "context": e.get("context") or "",
    }


def attach_context(corpus_rows, n_prev=2):
    """為語料庫句子補上同段落的前 n_prev 句中文。

    語料庫的檔案原始順序即句序（已驗證 corpus_id 尾碼遞增，如 G2D1P1/a81→a82），
    故依出現順序在同一 seg_uuid 內取前文即可。非語料庫來源（合成句、辭典例句、
    論文例句）本就是獨立句，context 留空。
    """
    from collections import defaultdict
    by_seg = defaultdict(list)
    for e in corpus_rows:
        by_seg[e.get("seg_uuid") or e["id"]].append(e)
    for seg_rows in by_seg.values():
        for i, e in enumerate(seg_rows):
            prev = [seg_rows[j]["chinese"] for j in range(max(0, i - n_prev), i)]
            e["context"] = "".join(prev)
    return corpus_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-ratio", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-rule-derived", action="store_true",
                    help="納入 rule-derived 合成句（預設排除；依 2026-07-23 審核，"
                         "rule-derived 未經母語者逐句裁定，不得進正式訓練）")
    ap.add_argument("--use-teacher-reviewed", action="store_true",
                    help="依 2026-07-24 手語老師人工審核結果：synth 只納入 "
                         "teacher_train_eligible=True（gloss 層通過者，含已修正與 gloss-passed "
                         "rule-derived；排除 7 句待影片裁定者），並使用已修正 gloss。"
                         "詞彙層可用於 Text→Gloss；NMS 仍屬影片軌。")
    ap.add_argument("--include-words", type=int, default=0,
                    help="額外加入 N 筆辭典詞→gloss 對（0=不加）")
    ap.add_argument("--no-corpus", action="store_true",
                    help="不加入文化部語料庫平行語料")
    ap.add_argument("--min-gloss-len", type=int, default=2,
                    help="語料庫句最小 Gloss token 數（濾掉過短碎片，預設 2）")
    ap.add_argument("--use-all", action="store_true",
                    help="【2026-08-05 使用者決策】不做任何人工審核閘門：辭典、語料庫、"
                         "論文例句、合成句全部直接使用；語料庫全部 5,272 句納入訓練"
                         "（不留存 test_corpus）；不濾單詞句。test 仍保留自有 33 句真實"
                         "錄影句以維持與 Stage A/B 的可比性。授權已確認（標明出處即可）。")
    ap.add_argument("--length-balance", action="store_true",
                    help="長度平衡取樣：重複長句以矯正「訓練資料由短句主導、"
                         "模型偏好短輸出」的偏差（2026-08-07 實測：訓練集 54.9%% 為 "
                         "≤4 詞短句、≥8 詞僅 10.4%%；語料庫長句稽核顯示 70%% 的輸出"
                         "短於參考答案）。只作用於 train，dev 不重複以免評估失真。")
    ap.add_argument("--context", type=int, default=0,
                    help="上下文翻譯（SCOPE 路線）：為語料庫句補上同段落的前 N 句中文。"
                         "0=關閉。依據：留存測試集實測 22.3%% 的參考 Gloss token 無法從"
                         "中文字面推得，需前文才能還原。test_corpus 依對話群組整組留存，"
                         "故其前文也在 test 內，不會洩漏。")
    ap.add_argument("--papers-as-test", action="store_true",
                    help="論文例句改作**獨立測試集**（test_papers.jsonl）而非訓練資料。"
                         "這些是語言學家標註的黃金例句，且與語料庫來源不同，"
                         "留作 test 可衡量泛化（語料庫已全進訓練，缺乏獨立測試集）。")
    ap.add_argument("--corpus-test-min-len", type=int, default=0,
                    help="留存語料庫 test 時只取 Gloss 長度 >= 此值的句子"
                         "（0=不限）。長句才是目前的弱項，留長句當 test 更有鑑別度。")
    ap.add_argument("--keep-excluded", action="store_true",
                    help="保留 2026-08-21 人工校訂判定「排除」的句子（預設排除）。"
                         "僅用於重現該次校訂之前的切分。")
    ap.add_argument("--textbook-as-test", action="store_true",
                    help="產生 test_textbook.jsonl（台灣手語教材，423 句）。"
                         "2026-08-22 起取代論文例句作為第三個測試集。"
                         "需先跑 scripts/build_textbook_testset.py。")
    ap.add_argument("--no-papers", action="store_true",
                    help="不納入中正大學手語論文例句（預設在 --use-all 下納入）")
    ap.add_argument("--corpus-test-ratio", type=float, default=None,
                    help="從文化部語料庫依對話群組留存這比例的真實句作『擴大真實 test 集』"
                         "（test_corpus.jsonl）；整段對話移出訓練池以杜絕洩漏。0=不留存（預設，"
                         "沿用舊行為）。擴大 test 集以穩定 BLEU 時設 0.12 左右。")
    args = ap.parse_args()
    exclude_rule_derived = not args.include_rule_derived  # 預設 True（審核安全預設）
    explicit_corpus_test = args.corpus_test_ratio is not None
    if args.corpus_test_ratio is None:
        args.corpus_test_ratio = 0.0
    if args.use_all:
        # 使用者決策：不設任何審核閘門、不濾單詞句。
        # 語料庫預設全數進訓練，但**若明確指定 --corpus-test-ratio 就尊重它**
        # （2026-08-08：需留存測試集才能衡量泛化）。
        exclude_rule_derived = False
        if not explicit_corpus_test:
            args.corpus_test_ratio = 0.0
        args.min_gloss_len = 1
    rng = random.Random(args.seed)
    OUT.mkdir(exist_ok=True)

    # --- test：真實句（排除模板佔位符句） ---
    real = load_jsonl(DATA / "tsl_sentences.jsonl")
    test = [norm_record(e, "real") for e in real if not e.get("is_template")]
    # 正規形比對（見 normalize_text）：只差標點的句子必須算同一句。
    test_chinese = {normalize_text(e["chinese"]) for e in test}
    test_gloss = {normalize_gloss(e["gloss_text"]) for e in test}

    # --- train 池：synth + twtsl 例句 ---
    pool = []
    human_excluded_count = 0
    synth = load_jsonl(DATA / "synth" / "tsl_synth.jsonl")
    synth_source_note = "review_status=pending（管線驗證）"
    for e in synth:
        if not args.keep_excluded and human_excluded(e):
            human_excluded_count += 1
            continue
        if args.use_all:
            pass                      # 無審核閘門：全部納入
        elif args.use_teacher_reviewed:
            # 依手語老師 2026-07-24 審核：只納入 gloss 層通過者，gloss 已含教師修正
            if not e.get("teacher_train_eligible"):
                continue
        elif exclude_rule_derived and e.get("confidence") == "rule-derived":
            continue
        pool.append(norm_record(e, "synth"))
    if args.use_teacher_reviewed:
        synth_source_note = ("手語老師 2026-07-24 gloss 層審核通過（含108句修正；"
                             "排除7句待影片裁定；NMS 屬影片軌）")
    if args.use_all:
        synth_source_note = "無審核閘門，全數納入（2026-08-05 使用者決策）"
    twtsl_sents = load_jsonl(DATA / "twtsl" / "twtsl_sentences.jsonl")
    for e in twtsl_sents:
        if not args.keep_excluded and human_excluded(e):
            human_excluded_count += 1
            continue
        pool.append(norm_record(e, "twtsl-sentence"))

    # --- 人工修正回饋（scripts/add_correction.py 產出）---
    # 少數修正混在數千句裡幾乎沒有影響力，故依 weight 複製多份加權。
    # 這些是使用者實測後親自確認的正確答案，優先度最高，一律進 train（不抽到 dev）。
    corrections, corrections_rows = 0, []
    corr_path = DATA / "corrections" / "corrections.jsonl"
    if corr_path.exists():
        for e in load_jsonl(corr_path):
            w = max(1, int(e.get("weight", 1)))
            rec = norm_record(e, "correction")
            corrections_rows.extend([dict(rec) for _ in range(w)])
            corrections += 1

    # --- 中正大學手語論文例句（語言學家標註；含呼應/分類詞標記者排除） ---
    papers_added = 0
    test_papers = []
    papers_path = DATA / "papers" / "paper_examples_all.jsonl"
    if not papers_path.exists():
        papers_path = DATA / "papers" / "paper_examples.jsonl"
    if args.use_all and not args.no_papers and papers_path.exists():
        for e in load_jsonl(papers_path):
            if e.get("has_notation"):
                continue              # 代形詞／呼應下標／描述性註解，下游無法檢索
            rec = norm_record(e, "paper")
            if args.papers_as_test:
                test_papers.append(rec)
            else:
                pool.append(rec)
                papers_added += 1

    # --- 台灣手語教材（tslcopus.deaf.com.tw）：獨立測試集，不進訓練池 ---
    # 2026-08-22 取代論文例句成為第三個測試集。來源檔由
    # scripts/build_textbook_testset.py 產出，已在該處去重並排除與既有語料
    # 重複的句子（測試洩漏）。⚠️ 授權未查證，data/tsl_textbook/ 在 .gitignore 內。
    test_textbook = []
    tb_path = DATA / "tsl_textbook" / "testset.jsonl"
    if args.textbook_as_test:
        if not tb_path.exists():
            raise SystemExit(
                f"找不到 {tb_path.relative_to(BASE)}，"
                f"先跑 scripts/build_textbook_testset.py")
        test_textbook = [norm_record(e, "textbook") for e in load_jsonl(tb_path)]

    # --- 文化部語料庫：可先依對話群組留存一批當「擴大真實 test 集」，其餘進訓練池 ---
    corpus_dropped_short = 0
    test_corpus_candidates = []
    test_corpus = []
    test_corpus_review_payload = None
    test_corpus_rejected = []
    corpus_path = DATA / "tslcorpus" / "parallel.jsonl"
    if not args.no_corpus and corpus_path.exists():
        # 先在**原始列**上補前文（norm_record 不保留 seg_uuid），再正規化。
        # 注意：過濾短句要在補前文之後，否則會漏掉被濾掉的那句作為前文，
        # 但前文本身不必進訓練，故此處先補、後濾。
        raw_corpus = load_jsonl(corpus_path)
        if args.context > 0:
            attach_context(raw_corpus, n_prev=args.context)
        corpus_recs = []
        for e in raw_corpus:
            if not args.keep_excluded and human_excluded(e):
                human_excluded_count += 1
                continue
            if len(e.get("gloss", [])) < args.min_gloss_len:
                corpus_dropped_short += 1
                continue
            corpus_recs.append(norm_record(e, "tslcorpus"))

        holdout_groups = set()
        if args.corpus_test_ratio > 0:
            # 依對話群組（seg_uuid 去段落號）整組留存，確保 test 對話不出現在 train
            cgroups = defaultdict(list)
            for e in corpus_recs:
                cgroups[e["group"]].append(e)
            gkeys = list(cgroups.keys())
            rng.shuffle(gkeys)
            target = round(len(corpus_recs) * args.corpus_test_ratio)
            picked_seen, picked = set(), 0
            for gk in gkeys:
                if picked >= target:
                    break
                holdout_groups.add(gk)
                for e in cgroups[gk]:
                    if args.corpus_test_min_len and \
                            len(e["gloss_text"].split("/")) < args.corpus_test_min_len:
                        continue      # 短句不列入 test（長句才是弱項）
                    # test_corpus 自身去重，且不與核心 33 句 test 重複
                    k = (e["chinese"], e["gloss_text"])
                    if k in picked_seen \
                            or normalize_text(e["chinese"]) in test_chinese \
                            or normalize_gloss(e["gloss_text"]) in test_gloss:
                        continue
                    picked_seen.add(k)
                    test_corpus_candidates.append(e)
                picked += len(cgroups[gk])

        for e in corpus_recs:
            if e["group"] in holdout_groups:
                continue  # 留存作 test_corpus，不進訓練池
            pool.append(e)

    # 教師審核 sidecar 固定 seed=42、ratio=0.12 產生的 585 句候選。
    # 先選群組再套審核：被老師排除者雖不輸出到 test_corpus，原候選群組與 pair
    # 仍維持 holdout，避免 rejected duplicate 回流 train/dev 改變切分。
    test_corpus_blocklist = list(test_corpus_candidates)
    test_corpus = list(test_corpus_candidates)
    if args.use_teacher_reviewed and test_corpus_candidates:
        if not CORPUS_TEST_REVIEW.exists():
            raise FileNotFoundError(
                f"教師審核擴大 test sidecar 不存在：{CORPUS_TEST_REVIEW}")
        test_corpus_review_payload, review_by_id = load_corpus_test_review(
            CORPUS_TEST_REVIEW)
        candidate_ids = [e["id"] for e in test_corpus_candidates]
        review_ids = [e["id"] for e in test_corpus_review_payload["records"]]
        if candidate_ids != review_ids:
            raise ValueError(
                "目前 corpus test 候選 ID／順序與教師審核 sidecar 不一致；"
                "請使用 --seed 42 --corpus-test-ratio 0.12 與同一資料版本")
        test_corpus = []
        for e in test_corpus_candidates:
            rv = review_by_id[e["id"]]
            for field in ("group", "chinese", "gloss_text"):
                if rv.get(field) != e.get(field):
                    raise ValueError(
                        f"教師審核 sidecar 與候選內容不一致：{e['id']} / {field}")
            if rv.get("teacher_test_eligible"):
                reviewed = dict(e)
                reviewed["review_status"] = "teacher-reviewed-2026-07-24"
                reviewed["teacher_final"] = rv["teacher_final"]
                reviewed["teacher_note"] = rv["teacher_note"]
                test_corpus.append(reviewed)
            else:
                if not rv.get("canonical_id"):
                    raise ValueError(f"排除列缺 canonical_id：{e['id']}")
                test_corpus_rejected.append(rv)

    # 原始候選（含教師排除列）也納入洩漏 blocklist，確保訓練池不含其相同句。
    tc_chinese = {normalize_text(e["chinese"]) for e in test_corpus} \
        | {normalize_text(e["chinese"]) for e in test_papers}
    tc_gloss = {normalize_gloss(e["gloss_text"]) for e in test_corpus} \
        | {normalize_gloss(e["gloss_text"]) for e in test_papers}
    tc_block_chinese = {normalize_text(e["chinese"]) for e in test_corpus_blocklist}
    tc_block_gloss = {normalize_gloss(e["gloss_text"]) for e in test_corpus_blocklist}

    if args.include_words > 0:
        words = load_jsonl(DATA / "twtsl" / "twtsl_words.jsonl")
        rng.shuffle(words)
        added = 0
        for e in words:
            if added >= args.include_words:
                break
            pool.append(norm_record(e, "twtsl-word"))
            added += 1

    # --- 去重 + 洩漏防護 ---
    seen, dedup, leaked = set(), [], 0
    for e in pool:
        nc, ng = normalize_text(e["chinese"]), normalize_gloss(e["gloss_text"])
        if nc in test_chinese or ng in test_gloss \
                or nc in tc_block_chinese or ng in tc_block_gloss:
            leaked += 1
            continue
        # 去重鍵改用正規形的句對。**仍是句對不是單看中文**：同一句中文配不同
        # Gloss 可能是合法的變體標註，直接丟掉會少掉訓練資料；真正該防的是它們
        # 跨 train/dev，那由下面的 text cluster 處理。
        key = (nc, ng)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(e)

    # --- 群組合併：同一句中文（正規形）的所有列必須同進退 ---------------
    # 原本只靠 e["group"] 去洩漏，但同一句話會以不同 group 重複收錄——語料庫
    # 同一段話被切在兩個對話編號（corpus:G3D11 與 G3C26 都是「吃飯時不許看
    # 手機」），辭典例句則同時掛在多個詞條底下（「我會三種語言」同時是
    # twtsl:口語 與 twtsl:語言 的例句）。group 不同、內容相同，於是 6 句
    # 原字串完全相同的句子同時出現在 train 與 dev。
    #
    # 解法：用 union-find 把「共用同一個正規化中文」的 group 併成一個 cluster，
    # 之後整個 cluster 一起進 train 或 dev。
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for e in dedup:
        find((e["source"], e["group"]))
    by_text = defaultdict(list)
    for e in dedup:
        by_text[normalize_text(e["chinese"])].append((e["source"], e["group"]))
    merged_clusters = 0
    for gkeys in by_text.values():
        if len(set(gkeys)) > 1:
            merged_clusters += 1
            first = gkeys[0]
            for g in gkeys[1:]:
                union(first, g)

    # cluster 的 source 取其成員的多數（只有極少數 cluster 會跨來源——實測 1 個，
    # 「我是學生」同時是 synth 模板句與 twtsl 例句）。分層抽樣仍照 source 走，
    # 但 cluster 是不可分割的最小單位。
    cluster_rows = defaultdict(list)
    for e in dedup:
        cluster_rows[find((e["source"], e["group"]))].append(e)
    cluster_src = {c: Counter(e["source"] for e in rows).most_common(1)[0][0]
                   for c, rows in cluster_rows.items()}

    # --- 依 source 分層、依 cluster 去洩漏抽 dev（整組進 train 或 dev） ---

    by_src = defaultdict(list)
    for c, rows in cluster_rows.items():
        by_src[cluster_src[c]].extend(rows)
    train, dev = [], []
    for src, items in by_src.items():
        groups = defaultdict(list)
        for e in items:
            groups[find((e["source"], e["group"]))].append(e)
        gkeys = list(groups.keys())
        rng.shuffle(gkeys)
        target = round(len(items) * args.dev_ratio) if len(items) >= 10 else 0
        dev_groups, dev_count = set(), 0
        for gk in gkeys:
            if dev_count >= target:
                break
            dev_groups.add(gk)
            dev_count += len(groups[gk])
        for gk in gkeys:
            (dev if gk in dev_groups else train).extend(groups[gk])
    train.extend(corrections_rows)   # 修正資料一律進 train，不抽到 dev

    # --- 長度平衡：重複長句，矯正短句主導造成的「輸出過短」偏差 ---
    length_balance_stats = None
    if args.length_balance:
        def repeat_factor(n_tokens):
            # 分桶重複次數（透明可解釋）：短句不動、中句 ×2、長句 ×4
            if n_tokens <= 4:
                return 1
            if n_tokens <= 7:
                return 2
            return 4

        before = Counter(min(len(e["gloss_text"].split("/")), 12) for e in train)
        balanced = []
        for e in train:
            n = len(e["gloss_text"].split("/"))
            balanced.extend(dict(e) for _ in range(repeat_factor(n)))
        after = Counter(min(len(e["gloss_text"].split("/")), 12) for e in balanced)
        length_balance_stats = {
            "before_total": len(train), "after_total": len(balanced),
            "before_ge8_pct": round(sum(v for k, v in before.items() if k >= 8)
                                    / len(train) * 100, 1),
            "after_ge8_pct": round(sum(v for k, v in after.items() if k >= 8)
                                   / len(balanced) * 100, 1),
            "buckets": "≤4 ×1、5–7 ×2、≥8 ×4",
        }
        train = balanced
    rng.shuffle(train)
    rng.shuffle(dev)

    # 去洩漏驗證：同一 group 不得同時出現在 train 與 dev
    train_groups = {e["group"] for e in train}
    dev_groups_all = {e["group"] for e in dev}
    group_leak = len(train_groups & dev_groups_all)
    assert group_leak == 0, f"分組洩漏 {group_leak} 組同時在 train/dev"

    # 正規形去洩漏驗證（2026-08-31，教授審查意見 2.4）。group 相同不代表內容
    # 不同——同一句話會掛在不同對話編號／不同詞條底下，光靠 group 擋不住。
    # 這裡直接對「正規化後的中文」下斷言，是最終防線。
    train_norm = {normalize_text(e["chinese"]) for e in train}
    dev_norm = {normalize_text(e["chinese"]) for e in dev}
    norm_leak = sorted(train_norm & dev_norm)
    assert not norm_leak, (
        f"train/dev 正規化中文重疊 {len(norm_leak)} 句，前 5 句：{norm_leak[:5]}")

    # test.jsonl = 核心 33 句（向後相容，eval 預設讀此檔）；另出 test_corpus.jsonl
    out_splits = [("train", train), ("dev", dev), ("test", test)]
    if test_corpus:
        out_splits.append(("test_corpus", test_corpus))
    if test_papers:
        out_splits.append(("test_papers", test_papers))
    if test_textbook:
        out_splits.append(("test_textbook", test_textbook))
    for name, rows in out_splits:
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for e in rows:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # 洩漏驗證：test_corpus 的對話群組不得出現在 train/dev
    tc_groups = {e["group"] for e in test_corpus}
    tc_group_leak = len(tc_groups & (train_groups | dev_groups_all))
    assert tc_group_leak == 0, f"test_corpus 群組洩漏 {tc_group_leak} 組進 train/dev"

    train_dev = train + dev
    # 全部用正規形比對——tc_chinese／tc_gloss 在上面就已經是正規形了，
    # 這裡若還用原字串會變成兩種口徑相減，等於檢查失效。
    train_dev_chinese = {normalize_text(e["chinese"]) for e in train_dev}
    train_dev_gloss = {normalize_gloss(e["gloss_text"]) for e in train_dev}
    train_dev_pairs = {(normalize_text(e["chinese"]), normalize_gloss(e["gloss_text"]))
                       for e in train_dev}
    tc_pairs = {(normalize_text(e["chinese"]), normalize_gloss(e["gloss_text"]))
                for e in test_corpus}
    # 每個測試集的正規化中文都不得出現在 train/dev。這是 2026-08-31 之前
    # 漏掉的那一層：核心 33 句有 3 句（我住在台北。／我知道／我不知道）
    # 去標點後就在 train 裡。
    train_dev_norm = {normalize_text(e["chinese"]) for e in train_dev}
    for _name, _rows in (("test（核心 33）", test), ("test_corpus", test_corpus),
                         ("test_papers", test_papers), ("test_textbook", test_textbook)):
        _leak = sorted({normalize_text(e["chinese"]) for e in _rows} & train_dev_norm)
        assert not _leak, (
            f"{_name} 有 {len(_leak)} 句正規化中文出現在 train/dev：{_leak[:5]}")

    tc_chinese_leak = len(tc_chinese & train_dev_chinese)
    tc_gloss_leak = len(tc_gloss & train_dev_gloss)
    tc_pair_leak = len(tc_pairs & train_dev_pairs)
    assert tc_chinese_leak == 0, f"test_corpus 中文洩漏 {tc_chinese_leak}"
    assert tc_gloss_leak == 0, f"test_corpus Gloss 洩漏 {tc_gloss_leak}"
    assert tc_pair_leak == 0, f"test_corpus pair 洩漏 {tc_pair_leak}"

    # 台灣手語教材測試集的洩漏檢查，與 test_corpus 分開算。
    # **只對中文與句對下斷言**：那兩者才是真洩漏（同一個輸入、或同一組輸入輸出
    # 被模型看過）。Gloss 單獨相同不算——教材句的 Gloss 中位數只有 4 個詞、
    # 最短 1 個，單 token 的「美麗」這種必然會與訓練集碰撞，那是短序列的
    # 巧合而非記憶。test_corpus 沒這問題是因為它有 --corpus-test-min-len 6。
    tb_chinese = {normalize_text(e["chinese"]) for e in test_textbook}
    tb_pairs = {(normalize_text(e["chinese"]), normalize_gloss(e["gloss_text"]))
                for e in test_textbook}
    tb_chinese_leak = len(tb_chinese & train_dev_chinese)
    tb_gloss_leak = len({normalize_gloss(e["gloss_text"]) for e in test_textbook}
                        & train_dev_gloss)
    tb_pair_leak = len(tb_pairs & train_dev_pairs)
    assert tb_chinese_leak == 0, f"test_textbook 中文洩漏 {tb_chinese_leak}"
    assert tb_pair_leak == 0, f"test_textbook pair 洩漏 {tb_pair_leak}"

    def ngram4_count(rows):
        n = 0
        for e in rows:
            toks = e["gloss_text"].split("/")
            n += max(len(toks) - 3, 0)
        return n

    def compo(rows):

        return dict(Counter(e["source"] for e in rows))

    manifest = {
        "seed": args.seed,
        "dev_ratio": args.dev_ratio,
        # use_teacher_reviewed 模式下，synth 納入與否改由 teacher_train_eligible 決定
        # （gloss 層通過的 rule-derived 會納入），故此欄位以實際生效邏輯回報。
        "exclude_rule_derived": (False if args.use_teacher_reviewed
                                 else exclude_rule_derived),
        "synth_selection": ("teacher_train_eligible（2026-07-24 手語老師 gloss 層審核）"
                            if args.use_teacher_reviewed else "confidence-based"),
        "include_words": args.include_words,
        "counts": {"train": len(train), "dev": len(dev), "test": len(test),
                   "test_corpus": len(test_corpus), "test_papers": len(test_papers),
                   "test_textbook": len(test_textbook)},
        "train_composition": compo(train),
        "dev_composition": compo(dev),
        "test_composition": compo(test),
        "human_excluded_2026_08_21": human_excluded_count,
        "leaked_removed": leaked,
        "corpus_test_ratio": args.corpus_test_ratio,
        "corrections": {"unique": corrections, "rows_after_weighting": len(corrections_rows)},
        "length_balance": length_balance_stats,
        "context_sentences": args.context,
        "test_corpus_candidate_count": len(test_corpus_candidates),
        "test_corpus_reviewed_count": len(test_corpus),
        "test_corpus_review_excluded_count": len(test_corpus_rejected),
        "test_corpus_review_rejected_ids": [
            e["id"] for e in test_corpus_rejected],
        "test_corpus_review_file": (
            str(CORPUS_TEST_REVIEW.relative_to(BASE))
            if test_corpus_review_payload else None),
        "test_corpus_review_sha256": (
            sha256_file(CORPUS_TEST_REVIEW)
            if test_corpus_review_payload else None),
        "test_corpus_ids_sha256": ids_sha256(test_corpus) if test_corpus else None,
        "test_corpus_groups": len(tc_groups),
        "test_corpus_group_leakage": tc_group_leak,
        "test_textbook_chinese_leakage": tb_chinese_leak,
        "test_textbook_pair_leakage": tb_pair_leak,
        # 只報不斷言，理由見上（短 Gloss 必然碰撞）
        "test_textbook_gloss_collision": tb_gloss_leak,
        "test_corpus_chinese_leakage": tc_chinese_leak,
        "test_corpus_gloss_leakage": tc_gloss_leak,
        "test_corpus_pair_leakage": tc_pair_leak,
        "test_4gram_count": ngram4_count(test),
        "test_corpus_4gram_count": ngram4_count(test_corpus),
        "split_method": "group-holdout（語料庫按對話 seg_uuid、twtsl 按詞條、synth 按模板整組留存）",
        "dev_group_leakage": group_leak,
        # 2026-08-31 起的表面形式正規化（審查意見 2.4）
        "surface_normalization": {
            "applied": "NFKC + 標點空白移除 + 臺→台/妳→你/祂牠→他",
            "scope": "去洩漏比對與去重鍵；不改寫實際寫出的語料內容",
            "text_clusters_merged": merged_clusters,
            "train_dev_normalized_chinese_overlap": len(norm_leak),
        },
        "n_groups": {"train": len(train_groups), "dev": len(dev_groups_all)},
        "corpus_dropped_short": corpus_dropped_short,
        "min_gloss_len": args.min_gloss_len,
        "no_corpus": args.no_corpus,
        "use_teacher_reviewed": args.use_teacher_reviewed,
        "synth_source_note": synth_source_note,
        "note": (
            ("test=Stage A 相同的 33 句真實已審核句，永不進訓練；"
             "synth 依手語老師 2026-07-24 gloss 層審核（含修正、排除待影片句），"
             "tslcorpus／twtsl 為官方／辭典來源文字層可保留；"
             "重複列由 (中文,gloss) 去重、對話依 seg_uuid 群組化防洩漏。"
             "詞彙層可用於 Text→Gloss；NMS／手形／地區變體屬影片軌，未納入亦不輸出。"
             "散布仍須另補文化部語料＋中正辭典授權。")
            if args.use_teacher_reviewed else
            ("test=Stage A 相同的 33 句真實已審核句，永不進訓練；"
             "train/dev 來源 synth／twtsl／tslcorpus 已於 2026-08-21 逐句人工校訂"
             "（scripts/apply_corpus_review.py：561 筆取代 Gloss、23 筆判定排除不進切分），"
             "review_status 見各筆記錄；校訂只涵蓋中文語意與 Gloss 的詞彙與語序，"
             "NMS／手形／地區變體屬影片軌，未經母語者確認。"
             "tslcorpus＝文化部語料庫全爬真實平行語料（最大宗真實資料）。")),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
