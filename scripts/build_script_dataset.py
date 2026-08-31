#!/usr/bin/env python3
"""把切分資料轉成教授指定的「手語腳本」訓練格式（tsl-script-v2）。

教授 2026-08-20 指定的格式：模型不再自由生成 gloss，而是**從候選清單挑
sign_id**；候選缺必要手語時輸出旗標。等於把下游的可播放性約束前移到訓練
目標裡。

    {"messages":[{"role":"system",...},
                 {"role":"user","content":"{\\"text\\":...,\\"candidates\\":[...]}"},
                 {"role":"assistant","content":"{\\"schema_version\\":\\"tsl-script-v2\\",
                   \\"sign_ids\\":[...],\\"clause_breaks\\":[...],
                   \\"candidate_coverage_risk\\":false,\\"oov_items\\":[]}"}],
     "metadata":{...}}

⚠️ **v2 把 needs_review 正名為 candidate_coverage_risk**（2026-08-31，教授
審查意見 4.2）。要重建 v17（線上部署中）的訓練集請加
`--schema-version tsl-script-v1`。

三個設計決定（會影響結果解讀，寫在這裡免得日後誤讀）：

1. **候選只從中文生成，絕不看正解**（見 sign_candidates.py）。
   拿正解回頭湊候選會讓訓練分布與上線分布不一致，模型學到的約束上線即失效。

2. **旗標是自然產生的，不是人工抽掉候選製造的**。
   檢索撈不到正解裡的某個手語時，該句就真的無法用候選拼出來，
   標 candidate_coverage_risk=true＋oov_items 列出缺的詞。這既是真實的
   失敗樣態，也剛好提供教授 schema 需要的負例，不必造假。

3. **oov_items 記 gloss 文字不是 sign_id**。缺的東西本來就沒有可用 ID，
   記文字下游才能拿去排補片（對接 data/video/video_gap.json 的補片工作表）。

用法：
    python3 scripts/build_script_dataset.py                 # 轉全部切分
    python3 scripts/build_script_dataset.py --splits test   # 只轉指定
    python3 scripts/build_script_dataset.py --k 25          # 調候選數
    python3 scripts/build_script_dataset.py --dry-run       # 只看涵蓋率統計

輸出：data/splits_script/{split}.jsonl ＋ coverage_stats.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import script_schema  # noqa: E402
from eval_video_coverage import parse_token  # noqa: E402
from sign_candidates import CandidateRetriever  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
SPLITS = BASE / "data" / "splits"
OUT = BASE / "data" / "splits_script"

# schema 常數與旗標欄位名的單一定義處在 script_schema.py（讀取端也 import 同一份）。
SCHEMA_FIELD = script_schema.SCHEMA_FIELD
SYSTEM_BY_SCHEMA = script_schema.SYSTEM_BY_SCHEMA
SCHEMA_VERSION = script_schema.CURRENT
SYSTEM = SYSTEM_BY_SCHEMA[SCHEMA_VERSION]


def clause_breaks(clauses, gloss_tokens: list[str],
                  contrib: list[int]) -> tuple[list[int], str]:
    """子句邊界，表示成 **sign_ids 陣列裡的索引**。

    回傳 (breaks, reason)。reason 供統計用：ok／no_annotation／unaligned／collapsed。

    2026-08-31 重寫。舊版壞在三處，實測 8,915 列訓練資料的 clause_breaks
    **全部是空陣列**——欄位是死的，模型只學會固定輸出 []：

      1. 先用 `/` 切 gloss_text 再找 `//`，但 `//` 在切分那一步就消失了。
      2. 只認獨立的 `++` token，而語料庫的實際寫法是 `買++` 這種後綴。
      3. 把 `++` 與 `//` 當成同一件事。它們來自不相交的兩個來源、意思也不同：
         `++` 只出現在文化部語料庫（303 筆），是**重複**記號；
         `//` 只出現在中正辭典例句（40 筆），才是**子句邊界**。
         兩者從未共存於同一筆。混為一談是舊版最根本的錯。

    正確的來源是上游就有的 `clauses` 欄位（twtsl 544/544 筆都有），
    由 split_data 帶進切分記錄。但**不能直接信它的字串**：2026-08-21 的人工
    校訂只更新 gloss_text，clauses 與 gloss_raw 都沒跟著改，10 筆
    `-corrected` 已經對不起來（例：clauses 寫 `蟑螂媽媽`，gloss_text 是
    `蟑螂/媽媽`）。所以這裡只採**每個子句的 token 數**，再對現行 gloss_text
    的 token 總數驗證；總數對不上就退回空陣列並記 unaligned，不猜。
    唯一那筆同時多子句又 stale 的（TWS0086）改的是錯字 `這倆個→這兩個`，
    token 數沒變，計數對齊照樣正確。

    **索引空間是 sign_ids 不是 gloss token**：一個 gloss token 可能貢獻 0 個
    （OOV）或 2–3 個（複合詞 X+Y 攤平）sign_id，兩邊的索引對不起來。
    所以用 contrib[i]＝第 i 個 gloss token 產出幾個 sign_id 來換算：邊界落在
    gloss 索引 c，對應的 sign_ids 位置就是 sum(contrib[:c])。
    整個子句都被丟掉時兩個邊界會疊在一起，去重；落在頭尾的邊界沒有意義
    （切不出兩段），丟掉。
    """
    if not clauses or len(clauses) < 2:
        return [], "no_annotation"

    counts = [len([t for t in str(c).split("/") if t.strip()]) for c in clauses]
    if sum(counts) != len(gloss_tokens):
        # 校訂改動了 token 數（合併／拆開／刪詞），邊界位置不再可信
        return [], "unaligned"

    # gloss token 索引空間的邊界＝各子句的累積長度（不含最後一個）
    cuts, acc = [], 0
    for n in counts[:-1]:
        acc += n
        cuts.append(acc)

    out = []
    n_signs = sum(contrib)
    for c in cuts:
        # 該邊界之前總共產出了幾個 sign_id
        pos = sum(contrib[:c])
        if 0 < pos < n_signs and pos not in out:
            out.append(pos)
    if not out:
        return [], "collapsed"
    return out, "ok"


def _semantic_ids() -> bool:
    """總表是不是語義 ID 方案（build_sign_inventory.py --id-scheme）。"""
    stats = BASE / "data" / "signs" / "inventory_stats.json"
    if not stats.exists():
        return False
    return json.loads(stats.read_text(encoding="utf-8")).get("id_scheme") == "semantic"


SEMANTIC_IDS = _semantic_ids()


def convert_row(row: dict, retr: CandidateRetriever, k: int,
                split: str, compact: bool = False, n_syn: int = 0,
                n_sem: int = 0, n_core: int = 30,
                distractor_ratio: float = 0.2,
                pin_core: int = 0) -> tuple[dict, dict]:
    text = str(row.get("chinese", "")).strip()
    # 命名注意：這是 gloss_text（已攤平、無 // 與方位標記），不是來源檔的
    # gloss_raw 欄位。舊版變數叫 gloss_raw，害 clause_breaks 以為拿得到 //。
    gloss_text = str(row.get("gloss_text", "")).strip()
    tokens = [t.strip() for t in gloss_text.split("/") if t.strip()]

    # train 句必須排掉自己：例句遷移會把同一句撈回來，等於直接把正解塞進候選，
    # 訓練出「照抄檢索結果」的捷徑，上線無此捷徑即失效。
    cands = retr.candidates(text, k=k, exclude_id=row.get("id"), n_syn=n_syn,
                            n_sem=n_sem, n_core=n_core,
                            distractor_ratio=distractor_ratio,
                            pin_core=pin_core)
    cand_ids = {c["sign_id"] for c in cands}

    sign_ids, oov = [], []
    # 缺口分三類，各自對應不同的補救動作，混在一起看就分不出該做什麼：
    #   not_in_library  動作庫根本沒有這個詞      → 要補片
    #   unusable_asset  有 ID 但影片演不出動作    → 要換源重錄（2026-08-31 新增）
    #   retrieval_miss  庫裡有、影片也好，沒撈到  → 要改進檢索
    oov_reason = {}
    # 第 i 個 gloss token 產出幾個 sign_id。0＝整個掉了（OOV），
    # 2–3＝複合詞攤平。clause_breaks 靠它換算索引空間。
    contrib: list[int] = []
    compounds: list[list[int]] = []     # 同一個複合單位的 sign_ids 索引群組
    reduplicated: list[int] = []        # 帶重複貌的 sign_ids 索引
    for tok in tokens:
        # 2026-08-31（審查意見 3.1）：一個 token 可能是複合（X+Y）與／或帶
        # 重複貌（X++）。以前這裡直接 resolve(tok)，而 resolve 內部的 norm()
        # 只留第一段——`樹+見` 的「見」就這樣人間蒸發，train 共蒸發 664 次。
        segs, redup = parse_token(tok)
        if not segs:
            contrib.append(0)
            continue
        produced = []
        for seg in segs:
            sid = retr.resolve(seg)
            if sid is None:
                oov.append(seg)
                oov_reason[seg] = "not_in_library"
            elif sid not in cand_ids:
                oov.append(seg)
                cls = (retr.by_id.get(sid) or {}).get("asset_class")
                oov_reason[seg] = ("unusable_asset"
                                   if cls in retr.UNUSABLE_CLASSES else "retrieval_miss")
            else:
                produced.append(len(sign_ids))
                sign_ids.append(sid)
        contrib.append(len(produced))
        # 複合單位只在**兩段都留下來**時才記——只剩一段就不是複合了
        if len(produced) >= 2:
            compounds.append(produced)
        # 重複貌掛在該 token 產出的**最後一個** sign 上。
        # `樹+開花++` 這種複合＋重複並存的只有 8 個，語料沒說 ++ 作用在
        # 整個複合還是最後一段，取最後一段是保守解讀，已知有歧義。
        if redup and produced:
            reduplicated.append(produced[-1])

    # 旗標的定義：參考 Gloss 有沒有全部落進候選清單。這是候選覆蓋率風險，
    # 不是翻譯品質預警——欄位名在 v2 已據此正名（見 SCHEMA_FIELD）。
    coverage_risk = bool(oov)
    # compact：候選壓成字串陣列。語意與教授原格式相同，
    # 但 k=40 的 prompt 從 1,015 token 降到 654（實測 Gemma 4 tokenizer，
    # 省 36%）——原格式每個候選都要 {"sign_id":...,"gloss":...} 的引號與鍵名，
    # 40 個候選光是這些框架就吃掉三百多 token。序列長度直接吃顯存與訓練時間，
    # 同樣預算下壓縮後可以多放 15 個候選，涵蓋率換得更划算。
    # 語義 ID（TSL_今天）自己就是詞面，再附「=今天」等於把 gloss 寫兩次；
    # 實測 dev 中位數 472→377 token。流水號（TSL_01084）看不懂才需要附。
    # v17 的 splits_script_k40sem 就是走 SEMANTIC_IDS 這條（候選是純 ID），
    # 拿掉這個分支產出的資料會與 v17 訓練集不同。
    if compact and SEMANTIC_IDS:
        candidates = [c["sign_id"] for c in cands]
    elif compact:
        candidates = [f"{c['sign_id']}={c['gloss']}" for c in cands]
    else:
        candidates = cands
    breaks, breaks_reason = clause_breaks(row.get("clauses"), tokens, contrib)

    user = {"text": text, "candidates": candidates}
    assistant = {
        "schema_version": SCHEMA_VERSION,
        "sign_ids": sign_ids,
        "clause_breaks": breaks,
    }
    if SCHEMA_VERSION == script_schema.V3:
        # 索引陣列而非巢狀物件：約束解碼只管 sign_ids 陣列內的字串常值，
        # 陣列一關閉就整步放行，所以這兩個欄位完全不影響它。
        assistant["compounds"] = compounds
        assistant["reduplicated"] = reduplicated
    assistant[SCHEMA_FIELD[SCHEMA_VERSION]] = coverage_risk
    assistant["oov_items"] = oov
    record = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False)},
        ],
        "metadata": {
            "id": row.get("id"),
            "source_units": [row["group"]] if row.get("group") else [],
            "asset_class": "natural_playable" if not coverage_risk else "needs_asset",
            "review_status": row.get("review_status", "pending"),
            "split": split,
        },
    }
    stat = {
        "tokens": len(tokens),
        "covered": len(sign_ids),
        "oov": len(oov),
        "coverage_risk": coverage_risk,
        "clause_breaks": breaks,
        "clause_breaks_reason": breaks_reason,
        "compounds": compounds,
        "reduplicated": reduplicated,
        "oov_items": oov,
        "not_in_library": [t for t in oov if oov_reason.get(t) == "not_in_library"],
        "unusable_asset": [t for t in oov if oov_reason.get(t) == "unusable_asset"],
        "retrieval_miss": [t for t in oov if oov_reason.get(t) == "retrieval_miss"],
    }
    return record, stat


def main() -> int:
    global OUT, SCHEMA_VERSION, SYSTEM
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*",
                    default=["train", "dev", "test", "test_corpus", "test_papers"])
    # k=40：實測 dev 集涵蓋率 k=25→77.6%、k=40→79.9%、k=60→82.9%，
    # 但 token 成本 k=60（壓縮後約 950）會逼近 max_len。40 是性價比轉折點。
    ap.add_argument("--k", type=int, default=40, help="候選清單長度上限")
    ap.add_argument("--compact", action="store_true", default=True,
                    help="候選壓成 ID=gloss 字串（省 36%% token）")
    ap.add_argument("--no-compact", dest="compact", action="store_false",
                    help="用教授原始的候選物件寫法")
    ap.add_argument("--n-syn", type=int, default=0,
                    help="同義展開通道的名額上限。**預設 0＝關閉**：實測 train "
                         "涵蓋率 92.6%%→90.7%%、dev 僅 +0.1pp，是淨負的。"
                         "見 sign_candidates._from_synonyms")
    ap.add_argument("--n-sem", type=int, default=0,
                    help="語義向量通道名額（預設 0＝關閉）。>0 時載入 "
                         "semantic_channel.SemanticRanker，需 .venv-emb 環境。"
                         "v17 的 splits_script_k40sem 即用此通道建成")
    ap.add_argument("--n-core", type=int, default=30, help="高頻核心手語名額")
    ap.add_argument("--distractor-ratio", type=float, default=0.2,
                    help="字元重疊干擾項佔 k 的比例上限")
    ap.add_argument("--pin-core", type=int, default=0,
                    help="高頻核心保底名額；實測淨負，見 sign_candidates.candidates 的 docstring")
    ap.add_argument("--out", type=Path, default=None,
                    help="輸出目錄，預設 data/splits_script（會覆蓋！建新版務必指定）")
    ap.add_argument("--schema-version", choices=sorted(SCHEMA_FIELD),
                    default=SCHEMA_VERSION,
                    help="輸出 schema。v2 的旗標欄位是 candidate_coverage_risk，"
                         "v1 是 needs_review（重建 v17 訓練集時用）")
    ap.add_argument("--dry-run", action="store_true", help="只算涵蓋率不寫檔")
    ap.add_argument("--limit", type=int, default=0, help="每個切分只處理前 N 句（除錯用）")
    args = ap.parse_args()

    if args.out is not None:
        OUT = args.out
    SCHEMA_VERSION = args.schema_version
    SYSTEM = SYSTEM_BY_SCHEMA[SCHEMA_VERSION]
    print(f"schema={SCHEMA_VERSION}（旗標欄位 {SCHEMA_FIELD[SCHEMA_VERSION]}）", flush=True)
    print("載入 sign 總表與檢索器…", flush=True)
    retr = CandidateRetriever()
    if args.n_sem > 0:
        from semantic_channel import SemanticRanker
        retr.semantic = SemanticRanker(retr.rows)
    print(f"動作庫 {len(retr.rows)} 個手語；訓練例句 {len(retr._ex_rows)} 句", flush=True)

    all_stats = {}
    for split in args.splits:
        path = SPLITS / f"{split}.jsonl"
        if not path.exists():
            print(f"跳過 {split}（無此檔）")
            continue
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
                if l.strip()]
        if args.limit:
            rows = rows[:args.limit]

        records, stats = [], []
        for i, row in enumerate(rows):
            rec, st = convert_row(row, retr, args.k, split, compact=args.compact,
                                  n_syn=args.n_syn, n_sem=args.n_sem,
                                  n_core=args.n_core,
                                  distractor_ratio=args.distractor_ratio,
                                  pin_core=args.pin_core)
            records.append(rec)
            stats.append(st)
            if (i + 1) % 500 == 0:
                print(f"  {split}: {i+1}/{len(rows)}", flush=True)

        tok_total = sum(s["tokens"] for s in stats)
        tok_cov = sum(s["covered"] for s in stats)
        n_ok = sum(1 for s in stats if not s["coverage_risk"])
        n_comp = sum(len(st["compounds"]) for st in stats)
        n_redup = sum(len(st["reduplicated"]) for st in stats)
        rows_comp = sum(1 for st in stats if st["compounds"])
        rows_redup = sum(1 for st in stats if st["reduplicated"])
        cb_reason = collections.Counter(st["clause_breaks_reason"] for st in stats)
        cb_rows = sum(1 for st in stats if st["clause_breaks"])
        miss = collections.Counter()
        miss_lib = collections.Counter()
        miss_asset = collections.Counter()
        for s in stats:
            miss.update(s["oov_items"])
            miss_lib.update(s["not_in_library"])
            miss_asset.update(s["unusable_asset"])

        summary = {
            "sentences": len(stats),
            "token_recall": round(tok_cov / tok_total, 4) if tok_total else None,
            "sentence_fully_covered": round(n_ok / len(stats), 4) if stats else None,
            "candidate_coverage_risk_rate": (round(1 - n_ok / len(stats), 4)
                                             if stats else None),
            "oov_tokens": tok_total - tok_cov,
            "oov_not_in_library": sum(miss_lib.values()),
            "oov_unusable_asset": sum(miss_asset.values()),
            "oov_retrieval_miss": (sum(miss.values()) - sum(miss_lib.values())
                                   - sum(miss_asset.values())),
            "top_missing_unusable_asset": miss_asset.most_common(15),
            # 子句邊界的產出情形（2026-08-31）。**覆蓋率極低是事實不是 bug**：
            # 只有中正辭典例句有 clauses 欄位，且 504/544 是單子句，
            # 全資料集真正有邊界的只有 40 筆。引用時務必連同這個數字一起講。
            "compound_units": n_comp,
            "compound_rows": rows_comp,
            "reduplicated_signs": n_redup,
            "reduplicated_rows": rows_redup,
            "clause_breaks": {
                "rows_with_breaks": cb_rows,
                "rows_with_breaks_pct": (round(100 * cb_rows / len(stats), 2)
                                         if stats else None),
                "reasons": dict(cb_reason),
            },
            "top_missing": miss.most_common(15),
            "top_missing_not_in_library": miss_lib.most_common(15),
        }
        all_stats[split] = summary
        print(f"\n[{split}] {len(stats)} 句")
        print(f"  詞涵蓋率 {summary['token_recall']:.1%}"
              f"／整句可拼出 {summary['sentence_fully_covered']:.1%}"
              f"／候選覆蓋風險 {summary['candidate_coverage_risk_rate']:.1%}")
        print(f"  缺口 {summary['oov_tokens']} token = "
              f"庫裡沒有 {summary['oov_not_in_library']}（要補片）"
              f" + 影片不堪用 {summary['oov_unusable_asset']}（要重錄）"
              f" + 檢索沒撈到 {summary['oov_retrieval_miss']}（要改檢索）")
        print(f"  最常缺：{summary['top_missing'][:8]}")
        print(f"  子句邊界 {cb_rows} 列有（{summary['clause_breaks']['rows_with_breaks_pct']}%）"
              f"／{dict(cb_reason)}")
        print(f"  複合單位 {n_comp} 個（{rows_comp} 列）"
              f"／重複貌 {n_redup} 個（{rows_redup} 列）")

        if not args.dry_run:
            OUT.mkdir(parents=True, exist_ok=True)
            with (OUT / f"{split}.jsonl").open("w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if getattr(retr, "semantic", None) is not None:
        retr.semantic.flush()
    if not args.dry_run:
        (OUT / "coverage_stats.json").write_text(
            json.dumps({"k": args.k, "n_syn": args.n_syn, "n_sem": args.n_sem,
                        "n_core": args.n_core,
                        "distractor_ratio": args.distractor_ratio,
                        "pin_core": args.pin_core,
                        "splits": all_stats}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n寫出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
