#!/usr/bin/env python3
"""建 sign_id 總表：把「動作庫實際演得出來的詞」編成穩定 ID。

為什麼需要這張表（2026-08-20，教授指定的訓練資料格式）：
    教授要求訓練資料改成「從候選清單挑 sign_id」的形式：

        {"sign_id":"TSL_TODAY","gloss":"今天"}

    但本專案至今是拿 **gloss 中文字串**當事實上的 ID，沒有穩定代號，
    也沒有一張「這個詞到底演不演得出來」的權威表。少了它，
    `asset_class: natural_playable` 只是一句空話。

錨定在哪裡：
    `data/video/lexicon.json` 是 0813 虛擬人動作庫的實際索引
    （gloss → 錄影檔＋起訖秒數），**它有的才播得出來**。
    所以 sign_id 一律以 lexicon 的鍵為準，不是以辭典詞條數為準
    —— 中正辭典有 3,500 詞不代表虛擬人演得出 3,500 詞。

ID 穩定性（重要）：
    ID 一旦發出去就會寫進訓練資料、模型權重與下游腳本，**永不可變**。
    故採 synthesize.py 的 SYN 編號同一原則：重跑時先讀回既有 ID
    原封不動，只有新出現的詞才接續編號。排序只影響首次建表。

用法：
    python scripts/build_sign_inventory.py            # 建表／增量更新
    python scripts/build_sign_inventory.py --check    # 只檢查不寫檔

gloss 清理與 name_key（2026-08-21）：
    動作庫的鍵有髒資料——2 筆帶控制字元（`'\x08訪談'`）、45 筆前後有空白、
    21 筆內含空格。`gloss` 欄位**維持原樣不動**：它是回查 lexicon.json 的
    join key，也是 sign_id 的錨點，改了 ID 穩定性就沒了。改為另出兩欄：

      gloss_clean  去控制字元＋strip 後的詞形，供比對與顯示
      name_key     語義 ID 的命名基礎（碰撞時加 _2、_3，依既有 sign_id 排序）

    為什麼需要 name_key：教授要語義代號（`TSL_TODAY`）。實測 Gemma 4 把流水號
    逐位切開（`TSL_01084` 要 8 token，`TSL_今天` 只要 4），語義代號其實更省，
    但前提是命名基礎唯一。清理前 16,628 筆 gloss 看似完全唯一，清理後就冒出
    **7 組碰撞**——那個「唯一」有一半是髒空白撐出來的假象。ID 一旦凍結不可改，
    碰撞規則必須現在就定死。

    這 7 組已於 2026-08-21 逐組比對動作資料查清：**4 組（訪談／許久／傾聽／抹黑）
    的 frames 陣列 sha256 完全相同**，是同一支錄影被匯入兩次，label 只差一個
    控制字元或尾空白；另 3 組（事情／共同／軟弱）確為不同錄影，兩支都保留。
    重複者標 duplicate_of 不刪 ID（見 DUPLICATE_OF），對外報告請引用
    distinct_signs（16,624）而非 signs（16,628）。

輸出：
    data/signs/sign_inventory.jsonl   逐筆：sign_id / gloss / gloss_clean / name_key / 來源 / 影片位置
    data/signs/gloss_to_sign.json     查詢索引：gloss → sign_id（含別名與清理後詞形）
    data/signs/inventory_stats.json   組成統計，供報告引用
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_video_coverage import Library, norm  # noqa: E402  複用既有分級解析器

BASE = Path(__file__).resolve().parent.parent
LEXICON = BASE / "data" / "video" / "lexicon.json"
OUT_DIR = BASE / "data" / "signs"
INVENTORY = OUT_DIR / "sign_inventory.jsonl"
INDEX = OUT_DIR / "gloss_to_sign.json"
STATS = OUT_DIR / "inventory_stats.json"

ID_PREFIX = "TSL_"
ID_WIDTH = 5

# 2026-08-21 逐筆比對動作資料後確認的重複收錄：同一支錄影被收兩次，
# label 只差一個控制字元或尾空白。判定依據不是時長相近，是 frames 陣列的
# sha256 **完全相同**（下列括號內為前 16 碼）；created_at 不同代表兩次匯入。
# 重驗方式：在 VM 上比對 ~/0813/recordings/<recording>.json 的 frames 欄位。
#
# ID 不刪除（發出去的 ID 永不可變，且下游可能已引用），改標 duplicate_of，
# 使「動作庫有幾個手語」這個數字據實扣除。實測這 4 筆目前引用數為 0，
# 檢索器也撈不到（髒鍵不可能被中文查詢命中），標記純為報告數字誠實。
# 鍵是**動作庫原鍵（gloss）**不是 sign_id：sign_id 的形式會隨命名方案改變
# （2026-08-21 教授裁示後由流水號改為語義 ID），gloss 才是跨方案穩定的錨點。
# 值是 (正本 gloss, 本筆當初驗的 recording, 正本當初驗的 recording)。
# **後兩項是防呆，不是註解**：動作庫會更新，同一個 gloss 可能被改指到別支錄影。
# 2026-08-21 實際發生過——「傾聽」原本兩支 frames sha256 相同、判為重複收錄，
# 動作庫更新後正本改指 moc_G2C27_R，兩支變成完全不同的錄影，舊結論當場作廢。
# 若不比對就沿用，會把一個真正不同的手語當成重複刪掉。對不上一律中止，重驗後再更新。
DUPLICATE_OF = {
    "\x08訪談": ("訪談", "moe_12_1218.json", "moe_02_0925.json"),  # sha 707b2b53e2e3ddbe
    "\x08許久": ("許久", "moe_12_1219.json", "moe_01_0339.json"),  # sha 51deed856f7e9edf
    "抹黑 ":    ("抹黑", "moe_12_0711.json", "moe_13_0773.json"),  # sha c3b4450341292dbc
    # 「傾聽」原在此表，2026-08-21 動作庫更新後兩支已非同一錄影，故移除。
    # 重驗結果：兩支都是 severe（moc_G2C27_R act_eff 0.2、moe_05_0022 act_eff 0.0），
    # 選哪支都演不出來，屬素材缺口而非命名問題，見 BOTH_UNUSABLE。
}
# 同名但**不同錄影**時，若已比對品質並選定用哪一支，記在這裡。被取代者不進候選，
# ID 同樣不刪。判定依據是 ~/0813/quality_scan/entries_final.csv 的實測欄位，
# 不是主觀觀感；重驗方式見該檔與 [[0813-video-quality-audit]]。
SUPERSEDED_BY = {
    # 事情：moe_04_0090 的 6.83 秒裡只有 97/205（左手）、136/205（右手）幀是舉手狀態，
    # 一半以上是手放下的空檔，當單詞片段播出等於有 3 秒多沒動作；
    # 手腕可見度 0.585/0.663（moc 為 1.0/1.0）——缺 landmark 是 retarget 端補不出來的。
    # 抖動 0.0191 vs 0.0081、畫質 480x360@30 vs 1920x1080@60。moe 唯一勝出的是
    # flicker 0.107 vs 0.154。線上實際使用的也是 moc 那支（usage 71 vs 0）。
    " 事情": ("事情", "moe_04_0090.json", "moc_G2D7P1.json"),
    # 共同：moe_12_0041 是 severe（act_eff 0.0698，幾乎整段偵測不到手），
    # 2248_一同 是 ok（0.7069），手腕可見度 0.99/0.86 vs 0.64/0.66。差一個數量級。
    "共同 ": ("共同", "moe_12_0041.json", "2248_一同.json"),
    # 軟弱：moe_13_1548 是 severe（act_eff 0.0351），2417_弱 是 ok（0.6447）。
    # 後者線上已在用（usage 3 vs 0）。
    "軟弱 ": ("軟弱", "moe_13_1548.json", "2417_弱.json"),
    # 求救：2026-08-21 動作庫更新後新出現的碰撞。slw_求救 是 ok（act_eff 0.7955，
    # 88/88 幀舉手、手腕 1.0/1.0）；moe_04_0780 是 severe（act_eff 0.0）。
    "求救 ": ("求救", "moe_04_0780.json", "slw_求救.json"),
}

# 同名兩支**都不堪用**者。不選任何一支（選了也演不出來），列在這裡是為了
# 讓它進補片待辦，而不是被當成「已處理」。
BOTH_UNUSABLE = {
    # 傾聽：moc_G2C27_R 0.889 秒裡只有 2–5 幀舉手（act_eff 0.2）；
    # moe_05_0022 act_eff 0.0。兩支皆 severe，需換源重錄。
    "傾聽": "兩支皆 severe（moc_G2C27_R act_eff 0.2／moe_05_0022 act_eff 0.0）",
}

# 另 3 組清理後同名者已比對確認為**不同**錄影，且都已比過品質並選定：
#   事情 TSL_00883（語料庫長片切出的 0.65 秒片段）勝 TSL_00004（辭典單詞 6.8 秒）
#   共同 TSL_02059（錄影 label「一同」）勝 TSL_02060
#   軟弱 TSL_14281（錄影 label「弱」）  勝 TSL_14282
# 三組的敗方都是 moe 教育部辭典，與既有影片品質盤點的結論一致（該來源 92.7% 不佳）。

# recording 檔名前綴 → 素材來源。來源會影響授權標註，必須逐筆帶著走。
SOURCE_PREFIX = {
    "moe": "moe_signdict",      # 教育部常用手語辭典（可訓練，資料不得公開）
    "moc": "moc_corpus",        # 文化部臺灣手語語料庫（訓練＋散布皆可，須標出處）
    "pn": "tsl_placenames",     # 台灣手語地名網
    "fn": "tsl_familynames",    # 台灣手語姓氏網
    "tasli": "tasli",
}


def clean_gloss(gloss: str) -> str:
    """去控制字元與前後空白。不動內容，只去掉不該存在的字元。"""
    return "".join(c for c in gloss if unicodedata.category(c) != "Cc").strip()


def source_of(recording: str) -> str:
    prefix = recording.split("_", 1)[0].split(".", 1)[0]
    return SOURCE_PREFIX.get(prefix, "other")


def load_existing() -> dict[str, str]:
    """讀回上一版的 gloss → sign_id，存成 legacy_sign_id 供追溯與換方案時對照。"""
    if not INVENTORY.exists():
        return {}
    mapping = {}
    with INVENTORY.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                mapping[row["gloss"]] = row["sign_id"]
    return mapping


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", type=Path, default=LEXICON)
    ap.add_argument("--id-scheme", choices=["semantic", "serial"], default="semantic",
                    help="sign_id 形式。semantic=TSL_<name_key>（2026-08-21 教授裁示，"
                         "預設）；serial=沿用舊流水號（僅供重現舊資料）")
    ap.add_argument("--check", action="store_true", help="只報告，不寫檔")
    args = ap.parse_args()

    if not args.lexicon.exists():
        print(f"找不到 {args.lexicon}\n"
              f"先抓下來：scp tku-gpu:"
              f"'~/0813/recordings/lexicon.json' data/video/", file=sys.stderr)
        return 1

    lexicon: dict[str, dict] = json.loads(args.lexicon.read_text(encoding="utf-8"))
    legacy = load_existing()          # gloss → 前一版 sign_id（含 2026-08-21 前的流水號）
    print(f"動作庫 {len(lexicon)} 筆；既有 ID {len(legacy)} 筆；命名方案 {args.id_scheme}")

    rows = []
    for gloss in sorted(lexicon):                       # 排序只影響首次建表
        entry = lexicon[gloss]
        recording = entry.get("recording", "")
        start, end = entry.get("start"), entry.get("end")
        rows.append({
            "gloss": gloss,
            # gloss 維持 lexicon 原鍵（join key＋命名錨點，不可動）；清理另存一欄
            "gloss_clean": clean_gloss(gloss),
            "source": source_of(recording),
            "recording": recording,
            "start": start,
            "end": end,
            "duration": (round(end - start, 3)
                         if isinstance(start, (int, float)) and isinstance(end, (int, float))
                         else None),
            # 動作庫有這筆＝虛擬人演得出來。這是 asset_class 的唯一事實依據。
            "asset_class": "natural_playable",
            "legacy_sign_id": legacy.get(gloss),
        })

    # ---- name_key：ID 的命名基礎，碰撞時加序號後綴 ----
    # 必須在指派 sign_id **之前**算完：語義 ID 就是由 name_key 生成的。
    by_clean: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_clean[r["gloss_clean"]].append(r)
    collisions = []
    for key, group in by_clean.items():
        # 排序決定誰不加後綴：原鍵本來就乾淨的優先，其次比舊流水號（沒有就比 gloss）。
        # 用舊流水號當次序是為了**與換方案前的結果完全一致**，避免這次遷移順手
        # 改掉誰是正名。必須與下面查詢索引的解析順序一致——索引用原鍵精準命中，
        # 若這裡讓髒的那筆搶到乾淨名字，兩張表會指向不同影片。
        group.sort(key=lambda r: (r["gloss"] != r["gloss_clean"],
                                  r["legacy_sign_id"] or r["gloss"]))
        for i, r in enumerate(group):
            r["name_key"] = key if i == 0 else f"{key}_{i + 1}"
        if len(group) > 1:
            collisions.append({
                "gloss_clean": key,
                "members": [{"gloss": r["gloss"], "name_key": r["name_key"],
                             "source": r["source"], "recording": r["recording"],
                             "duration": r["duration"]} for r in group],
                # 2026-08-21 已逐組比對 frames：4 組完全相同（重複收錄），
                # 3 組不同（真變體，已逐組比品質選定）。新碰撞需照同樣方式驗。
                "verified_duplicate": any(r["gloss"] in DUPLICATE_OF for r in group),
            })

    # ---- 指派 sign_id ----
    # 2026-08-21 教授裁示：改用中文語義 ID。依據是 Gemma 4 把數字逐位切開，
    # TSL_01084 要 8 token 而 TSL_今天 只要 4，且流水號看不懂、候選一定得補
    # 「=今天」才有語意（10 token）；assistant 輸出的 sign_ids 用的是同一批 ID，
    # 輸入輸出兩邊都在付這個代價。實測整段序列 631→377 token、
    # 每輪訓練 4h10m→2h36m（見 交接說明_2026-08-21.md）。
    for r in rows:
        r["sign_id"] = (ID_PREFIX + r["name_key"] if args.id_scheme == "semantic"
                        else r["legacy_sign_id"])
    missing = [r["gloss"] for r in rows if not r["sign_id"]]
    if missing:
        raise SystemExit(f"--id-scheme serial 但有 {len(missing)} 筆查無舊 ID：{missing[:5]}")
    # 判定表以 gloss 為鍵（跨命名方案穩定），值也是 gloss，需轉成當前方案的 sign_id
    id_by_gloss = {r["gloss"]: r["sign_id"] for r in rows}
    rec_by_gloss = {r["gloss"]: r["recording"] for r in rows}
    stale = []
    for r in rows:
        for field, table in (("duplicate_of", DUPLICATE_OF),
                             ("superseded_by", SUPERSEDED_BY)):
            entry = table.get(r["gloss"])
            if entry is None:
                continue
            target, rec_self, rec_target = entry
            if target not in id_by_gloss:
                raise SystemExit(f"{field} 指向的 gloss 不在動作庫：{target!r}")
            # 防呆：判定當時驗的是哪支錄影，現在還是不是同一支
            now_self, now_target = rec_by_gloss[r["gloss"]], rec_by_gloss[target]
            if now_self != rec_self or now_target != rec_target:
                stale.append(
                    f"  {field} {r['gloss']!r}→{target!r}："
                    f"當初驗 {rec_self}／{rec_target}，現在是 {now_self}／{now_target}")
                continue
            r[field] = id_by_gloss[target]
    if stale:
        raise SystemExit(
            "動作庫已把下列 gloss 改指到別支錄影，原判定的證據不再成立：\n"
            + "\n".join(stale)
            + "\n請重新比對 frames／品質後更新 DUPLICATE_OF／SUPERSEDED_BY，"
              "不要直接沿用——2026-08-21「傾聽」就是這樣從『重複收錄』變成兩支不同錄影的。")

    ids = [r["sign_id"] for r in rows]
    if len(set(ids)) != len(ids):
        dup = [k for k, c in collections.Counter(ids).items() if c > 1]
        raise SystemExit(f"sign_id 不唯一：{dup[:10]}")

    name_keys = [r["name_key"] for r in rows]
    if len(set(name_keys)) != len(name_keys):
        dup = [k for k, c in collections.Counter(name_keys).items() if c > 1]
        raise SystemExit(f"name_key 不唯一，語義 ID 無法凍結：{dup[:10]}")

    dirty = {
        "control_chars": sum(1 for r in rows
                             if any(unicodedata.category(c) == "Cc" for c in r["gloss"])),
        "outer_space": sum(1 for r in rows if r["gloss"] != r["gloss"].strip()),
        "inner_space": sum(1 for r in rows
                           if " " in r["gloss_clean"] or "　" in r["gloss_clean"]),
        "slash": sum(1 for r in rows if "/" in r["gloss_clean"]),
    }
    if collisions:
        n_dup = sum(c["verified_duplicate"] for c in collisions)
        print(f"\n清理後 gloss 碰撞 {len(collisions)} 組"
              f"（已驗證 {n_dup} 組為重複收錄、{len(collisions) - n_dup} 組為不同錄影）：")
        for c in collisions:
            ms = "／".join(f"{m['name_key']}({m['recording']} {m['duration']}s)"
                           for m in c["members"])
            if c["verified_duplicate"]:
                tag = "重複收錄"
            elif c["gloss_clean"] in BOTH_UNUSABLE:
                tag = "真變體·兩支皆不堪用"
            elif any(m["gloss"] in SUPERSEDED_BY for m in c["members"]):
                tag = "真變體·已選定"
            else:
                tag = "真變體·未比品質"
            print(f"    [{tag}] {c['gloss_clean']}: {ms}")
        print("  已用 name_key 後綴區分。"
              "⚠ 標「未比品質」者尚未驗證，需比對 frames 與品質後更新判定表。")
        for g, why in BOTH_UNUSABLE.items():
            if any(c["gloss_clean"] == g for c in collisions):
                print(f"  ⚠ 「{g}」{why}——屬素材缺口，需換源重錄，不是選哪支的問題。")
    if any(dirty.values()):
        print(f"\n動作庫 gloss 髒資料（gloss 欄保持原狀，清理值在 gloss_clean）："
              f"控制字元 {dirty['control_chars']}／前後空白 {dirty['outer_space']}"
              f"／內含空格 {dirty['inner_space']}／含斜線 {dirty['slash']}")

    # ---- 查詢索引：除了本名，也把 T1–T6 能解到同一支影片的寫法一併收錄 ----
    # 訓練與推論都要用這份索引把 gloss 轉成 sign_id，寫法不一致才不會漏。
    lib = Library(lexicon)
    by_gloss = {r["gloss"]: r["sign_id"] for r in rows}
    index: dict[str, str] = dict(by_gloss)
    alias_added = clean_added = 0
    # 清理後詞形也要能查到：語料裡寫的是「訪談」，動作庫的鍵卻是 '\x08訪談'，
    # 不補這層就會把演得出來的詞誤判成缺片。已存在的鍵一律不覆蓋（先到先得，
    # 與碰撞報告一致），避免同一個詞被指到後編號的那支影片。
    for r in rows:
        c = r["gloss_clean"]
        if c and c not in index:
            index[c] = r["sign_id"]
            clean_added += 1
    for gloss in list(by_gloss):
        n = norm(gloss)
        if n and n not in index:
            index[n] = by_gloss[gloss]
            alias_added += 1

    # 指向重複收錄的鍵改導向正本：查詢應該拿到正規那支，兩者動作資料本就相同
    repointed = 0
    replaced_ids = {r["sign_id"]: r[f]
                    for r in rows
                    for f in ("duplicate_of", "superseded_by") if f in r}
    for key, sid in list(index.items()):
        if sid in replaced_ids:
            index[key] = replaced_ids[sid]
            repointed += 1
    if repointed:
        print(f"索引改導向：{repointed} 個鍵原指向重複收錄或已被取代的 ID")

    # name_key 與索引必須指向同一支影片，否則下游兩條路徑會播出不同的手語
    for r in rows:
        if r["gloss"] in SUPERSEDED_BY:
            continue                      # 已刻意改導向較準確的那支
        if r["name_key"] == r["gloss_clean"] and index.get(r["gloss_clean"]) != r["sign_id"]:
            raise SystemExit(
                f"name_key 與索引不一致：{r['gloss_clean']} → "
                f"name_key {r['sign_id']} / 索引 {index.get(r['gloss_clean'])}")

    dup_rows = [r for r in rows if "duplicate_of" in r]
    sup_rows = [r for r in rows if "superseded_by" in r]
    stats = {
        "lexicon_entries": len(lexicon),
        "signs": len(rows),
        # 對外報告用這個數字：扣掉已驗證的重複收錄
        "distinct_signs": len(rows) - len(dup_rows),
        "duplicates": {r["sign_id"]: r["duplicate_of"] for r in dup_rows},
        # 取代不影響 distinct_signs：詞還在，只是換一支影片演
        "superseded": {r["sign_id"]: r["superseded_by"] for r in sup_rows},
        "id_scheme": args.id_scheme,
        "ids_changed_vs_previous": sum(
            1 for r in rows if r["legacy_sign_id"] and r["legacy_sign_id"] != r["sign_id"]),
        "new_signs_this_run": sum(1 for r in rows if not r["legacy_sign_id"]),
        "index_keys": len(index),
        "alias_keys_added": alias_added,
        "clean_keys_added": clean_added,
        "index_keys_repointed_from_duplicate": repointed,
        "gloss_dirty": dirty,
        "name_key_collisions": collisions,
        "by_source": dict(collections.Counter(r["source"] for r in rows).most_common()),
        "by_gloss_len": dict(collections.Counter(
            min(len(r["gloss"]), 6) for r in rows).most_common()),
        "missing_timing": sum(1 for r in rows if r["duration"] is None),
    }
    print(json.dumps({k: v for k, v in stats.items()
                      if k != "name_key_collisions"}, ensure_ascii=False, indent=2))

    if args.check:
        print("--check：未寫檔")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with INVENTORY.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"寫出 {INVENTORY}（{len(rows)}）／{INDEX}（{len(index)}）／{STATS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
