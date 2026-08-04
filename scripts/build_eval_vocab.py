#!/usr/bin/env python3
"""建立「評估專用」的兩層詞彙表分母（計畫 6.1 節）。

為什麼不直接用 build_vocab.py 的 gloss_master？
  gloss_master 由**完整**語料庫（含後來被留存為 test_corpus 的句子）建成，
  拿它當評估分母等於用測試答案定義評分標準（循環論證／洩漏）。
  本腳本只用【辭典】＋【train/dev 切分】，絕不碰 test 與 test_corpus。

兩個分母意義不同，必須分開報告（2026-08-04 診斷結論）：

  renderable（可播放率）＝ 自有 85 ＋ 中正辭典詞條/別名
      每詞都有單詞示範影片 → 下游動作庫真的檢索得到、播得出來。
      這個數字低＝**下游素材缺口**。

  legit（合法詞率）＝ renderable ＋ train 出現過的真實語料 Gloss
      文化部語料庫是真實聾人手語產出（film_url 為整段對話影片、非逐詞），
      這些詞是合法 TSL 用詞但下游無單詞素材。
      這個數字低＝**模型可能亂造詞**。

輸出：data/vocab/eval_vocab.json
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
OUT = DATA / "vocab"


def toks(gloss_text):
    return [t for t in str(gloss_text).replace("／", "/").split("/") if t.strip()]


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    own = set(json.load((DATA / "tsl_gloss_vocab.json").open(encoding="utf-8"))["glosses"])

    dict_words = set()
    for line in (DATA / "twtsl" / "twtsl_words.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        w = json.loads(line)
        for key in ("gloss_text", "name", "chinese"):
            if w.get(key):
                dict_words.add(w[key])
        dict_words.update(w.get("aliases") or [])
    dict_words.discard(None)

    renderable = own | dict_words

    # train/dev 的 Gloss（真實語料用詞）；明確排除 test 與 test_corpus
    train_gloss = set()
    used = []
    for name in ("train.jsonl", "dev.jsonl"):
        p = DATA / "splits" / name
        if not p.exists():
            continue
        used.append(name)
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                train_gloss.update(toks(json.loads(line)["gloss_text"]))

    legit = renderable | train_gloss

    out = {
        "built_from": {
            "own_gloss": sorted(own).__len__(),
            "twtsl_dictionary": len(dict_words),
            "splits_used": used,
            "train_dev_gloss_types": len(train_gloss),
        },
        "excluded_on_purpose": ["test.jsonl", "test_corpus.jsonl",
                                "data/vocab/gloss_master.jsonl（由完整語料庫建成，含 test）"],
        "counts": {"renderable": len(renderable), "legit": len(legit)},
        "note": ("renderable=可直接檢索單詞影片（下游可播放）；"
                 "legit=再加上 train 出現過的真實語料 Gloss（合法 TSL 用詞但無單詞素材）。"
                 "評估時兩者都要與參考答案天花板 InVocabRef% 並列。"),
        "renderable": sorted(renderable),
        "legit": sorted(legit),
    }
    path = OUT / "eval_vocab.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK → {path.relative_to(BASE)}")
    print(f"  renderable {len(renderable)}（自有 {len(own)} + 辭典 {len(dict_words)}）")
    print(f"  legit      {len(legit)}（+ train/dev 真實語料 {len(train_gloss)} types，來源 {used}）")


if __name__ == "__main__":
    main()
