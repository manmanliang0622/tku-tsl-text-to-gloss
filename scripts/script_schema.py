#!/usr/bin/env python3
"""手語腳本 schema 的單一定義處（tsl-script-v1 / v2）。

**v2 把 needs_review 正名為 candidate_coverage_risk**（2026-08-31，教授審查
意見 4.2）。這個旗標的正解是 `bool(oov)`——「參考 Gloss 有沒有全部落進候選
清單」，是**檢索覆蓋率**訊號，不是翻譯品質預警。它偵測不到：

  - 候選完整但模型選錯詞
  - 詞都選到但語序錯
  - 重複／複合動作遺失
  - 整句語意不自然
  - NMS 或影片品質問題

叫 needs_review 會讓讀報表的人以為模型有品質預警能力，實際上沒有。

兩版並存的理由：v17（線上部署中）是 v1 訓練的，重建那份訓練集才能重現它的
數字。**寫入端**依 schema 版本擇一輸出，**讀取端**一律用 `read_flag()`
兩種都收——推論結果檔的新舊混用是常態（results/ 裡 v14～v17cd 全是 v1）。

這個模組刻意零相依，因為 serve_model.py 會 import 它，而 serve_model 要能在
0821_bundle 那個精簡環境裡跑。部署時必須一起帶（見 serve_model 的相依清單）。
"""

V1 = "tsl-script-v1"
V2 = "tsl-script-v2"

# schema 版本 → 旗標欄位名
SCHEMA_FIELD = {
    V1: "needs_review",
    V2: "candidate_coverage_risk",
}
CURRENT = V2

# 讀取時兩種都認。順序無所謂——同一份輸出不會兩個都有。
FLAG_KEYS = tuple(SCHEMA_FIELD[v] for v in (V2, V1))

SYSTEM_BY_SCHEMA = {
    V1: ("將繁體中文轉成可執行的臺灣手語腳本。只能使用候選清單中的 sign_id，"
         "不得創造新 ID。缺少必要手語時，必須輸出 needs_review=true。只輸出 JSON。"),
    # v2 的措辭同步收緊：講明是「候選清單裡缺」，而不是含糊的「缺少必要手語」。
    V2: ("將繁體中文轉成可執行的臺灣手語腳本。只能使用候選清單中的 sign_id，"
         "不得創造新 ID。候選清單缺少必要手語時，必須輸出 "
         "candidate_coverage_risk=true。只輸出 JSON。"),
}


def read_flag(obj, default=False):
    """從模型輸出或參考答案裡取旗標，v1／v2 兩種欄位名都收。"""
    if not isinstance(obj, dict):
        return default
    for key in FLAG_KEYS:
        if key in obj:
            return bool(obj[key])
    return default


def flag_field(schema_version=CURRENT):
    """該 schema 版本的寫入欄位名。"""
    return SCHEMA_FIELD[schema_version]
