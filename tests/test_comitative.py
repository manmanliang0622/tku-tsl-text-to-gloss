#!/usr/bin/env python3
"""伴隨句雙數收攏規則（scripts/comitative.py）。

守的是兩件事：該補的補了，**不該補的一個都不能補**。後者才是重點——
語料庫 122 句含「跟」的句子裡有 53 句是非伴隨用法（跟X說／跟著／腳跟），
無差別補寫會把那些句子弄壞。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import comitative as cm

# (中文, 模型輸出 gloss, 期望輸出)
SHOULD_ADD = [
    ("我跟媽媽去吃飯",         ["我", "媽媽", "吃飯"],        ["我", "媽媽", "我們兩個", "吃飯"]),
    ("我跟朋友去吃飯",         ["我", "朋友", "吃飯"],        ["我", "朋友", "我們兩個", "吃飯"]),
    ("我和姐姐一起去公園散步",  ["我", "姐姐", "公園", "散步"], ["我", "姐姐", "我們兩個", "公園", "散步"]),
    ("我與太太去看電影",       ["我", "太太", "電影", "看"],   ["我", "太太", "我們兩個", "電影", "看"]),
    # 中間夾修飾語：TC00408 我和我的高中同學…
    ("我和我的高中同學出去玩",  ["我", "同學", "玩"],          ["我", "同學", "我們兩個", "玩"]),
]

SHOULD_NOT = [
    # 單向言談：語料用有方向性的「告訴」（TC01942 我跟你說 → 我/告訴）
    ("我跟你說",              ["我", "告訴"]),
    ("我跟醫生說我要考慮",     ["我", "醫生", "告訴", "考慮"]),
    # 跟著／跟隨（TC00720 我也跟著回頭看 → 我/一樣/回頭看）
    ("我也跟著吃素",           ["我", "一樣", "素", "吃"]),
    ("我跟團去埃及",           ["我", "旅行團", "去", "埃及"]),
    # 腳跟（TC00154）
    ("腳跟底部腫起來",         ["腳跟底部", "腫"]),
    # 比較句（TC00329 味道跟媽媽煮的一樣 → 一模一樣）
    ("這個味道跟媽媽煮的一樣",  ["這", "味道", "媽媽", "煮", "一模一樣"]),
    # 取得類單向動作（TC00545 我只好跟同學借筆記）
    ("我只好跟同學借筆記",      ["我", "同學", "筆記", "借"]),
    # 三人以上不是「兩個」（TC01302 我和朋友三個人視訊聊天 → 我們三人）
    ("我和朋友三個人視訊聊天",  ["我", "朋友", "視訊", "聊天"]),
    ("午餐時間我和同事們去餐廳", ["我", "同事", "餐廳", "去"]),
    # 家人／親戚人數不定（TC03600 我和家人一起去旅行 → 我/家人/一起/旅行）
    ("我和家人一起去旅行",      ["我", "家人", "一起", "旅行"]),
    # 伴隨對象不是人（TC00441 讀書跟打工之間）
    ("生活穿梭在讀書跟打工之間", ["來回", "繼續"]),
    # 模型已經自己收攏了，不重複補
    ("我跟媽媽去吃飯",         ["我", "媽媽", "我們兩個", "吃飯"]),
    ("我和朋友去博物館",       ["我", "朋友", "兩人去", "博物館"]),
    # 模型沒把兩個主語都打出來 → 沒有東西可收攏
    ("我跟媽媽去吃飯",         ["媽媽", "吃飯"]),
]


def main():
    bad = 0
    for zh, got, want in SHOULD_ADD:
        out, changed = cm.apply(zh, got, max_signs=18)
        if out != want or not changed:
            print(f"✗ 該補沒補對 {zh}: {'/'.join(out)}  期望 {'/'.join(want)}")
            bad += 1
    for zh, got in SHOULD_NOT:
        out, changed = cm.apply(zh, got, max_signs=18)
        if changed or out != got:
            print(f"✗ 不該補卻補了 {zh}: {'/'.join(got)} → {'/'.join(out)}")
            bad += 1

    # MAX_SIGNS 上限：滿了就不補，否則補出來的元素會被下游截掉
    full = ["我", "朋友"] + [f"詞{i}" for i in range(16)]
    out, changed = cm.apply("我跟朋友去吃飯", full, max_signs=18)
    if changed:
        print("✗ 已達 MAX_SIGNS 仍補寫")
        bad += 1

    # apply_ids：sign_ids 與 gloss 必須同步，且查無 ID 時不動
    idx = {"我": "TSL_我", "朋友": "TSL_朋友", "吃飯": "TSL_吃飯", "我們兩個": "TSL_我們兩個"}
    ids, gl, ch = cm.apply_ids("我跟朋友去吃飯", ["TSL_我", "TSL_朋友", "TSL_吃飯"],
                               ["我", "朋友", "吃飯"], idx, max_signs=18)
    if not ch or ids != ["TSL_我", "TSL_朋友", "TSL_我們兩個", "TSL_吃飯"] \
            or [idx.get(g, g) for g in gl] != ids:
        print(f"✗ apply_ids 兩份串列不同步: {ids} / {gl}")
        bad += 1
    ids2, gl2, ch2 = cm.apply_ids("我跟朋友去吃飯", ["TSL_我", "TSL_朋友", "TSL_吃飯"],
                                  ["我", "朋友", "吃飯"], {}, max_signs=18)
    if ch2 or ids2 != ["TSL_我", "TSL_朋友", "TSL_吃飯"]:
        print("✗ 總表查無「我們兩個」時仍硬塞 ID")
        bad += 1

    total = len(SHOULD_ADD) + len(SHOULD_NOT) + 3
    if bad:
        print(f"\n{bad}/{total} 項失敗")
        return 1
    print(f"✓ {total} 項全通過（該補 {len(SHOULD_ADD)}、不該補 {len(SHOULD_NOT)}、邊界 3）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
