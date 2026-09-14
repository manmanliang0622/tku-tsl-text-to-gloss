# `data/splits_v17/`：v17／v17cd 當時的切分（凍結）

2026-08-31 依教授審查意見 2.4 修正 `split_data.py`——去洩漏與去重改用**表面
形式正規化**（NFKC＋去標點＋臺/台等異體字折疊），並把「同一句中文的所有列」
合併成不可分割的 cluster。修正後 `data/splits/` 的 train／dev 內容改變：

| split | 修正前 | 修正後 |
|---|---:|---:|
| train | 8,915 | 9,064 |
| dev | 663 | 548 |
| test（核心 33） | 33 | 33（**位元完全相同**） |
| test_corpus | 166 | 166（**位元完全相同**） |
| test_textbook | 423 | 423（**位元完全相同**） |

三個測試集完全沒動，所以 v14–v17cd 在 test／test_corpus／test_textbook 上的
歷史數字仍可直接重算對帳。**只有 dev 變了**，而 v17 的 checkpoint 選擇與
needs_review 門檻 0.095349 都是在舊 dev 上選的——要重驗那兩個決定，就必須用
這份凍結的舊 dev，因此保留在這裡。

修正抓到的實際洩漏（都在舊切分裡）：

- 核心 33 句有 3 句去標點後出現在 train：`我住在台北。`／`我知道`／`我不知道`
- dev 與 train 有 6 句**原字串完全相同**（去重鍵是 `(chinese, gloss_text)`，
  同一句中文配不同 Gloss 就兩邊都留），另有 5 句只差標點
- 合計 42 個 text cluster 被合併

新切分的驗證見 `data/splits/manifest.json` 的 `surface_normalization` 欄位，
`train_dev_normalized_chinese_overlap` 必須是 0（`split_data.py` 有硬斷言）。

**這份目錄不要再更新**，它是歷史對帳的錨點。
