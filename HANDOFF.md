# 專案交接（HANDOFF）

淡江大學專題：中文 → 臺灣手語 Gloss 翻譯模型（Gemma 4 微調）。
本檔給接手／並行的 session 快速掌握現況。最後更新：2026-07-23。

## 1. 分支結構（工作流分流）

| 分支 | 用途 |
|---|---|
| `main` | 整合線。兩邊工作完成後併回這裡 |
| `model` | 模型／訓練工作（train_qlora、eval_model、metrics、results） |
| `data` | 資料／爬蟲／審核工作（data/、scripts/scrape_*、synthesize、build_vocab、split_data、審核報告） |

- 模型工作在 `model` 提交、資料工作在 `data` 提交，各自完成再併回 `main`；避免多 session 同時改 `main` 衝突。
- 三分支目前對齊同一起點。併回 `main` 用 fast-forward。

## 2. 學校 VM 與工作流

- 連線：`ssh -p 2288 b310ai@163.13.202.125`（Ubuntu 22.04、RTX 4060 Ti 16GB）。**密碼由團隊保管，不入庫**。
- VM 已固定主機金鑰（勿用 `StrictHostKeyChecking=no`）。
- **VM 沒有儲存 git 認證**：push 需手動帶 token，或走「VM 跑生成／訓練 → 檔案帶回有 gh 認證的機器 → 由該機器 push」的橋接方式。
- VM 家目錄 `~/tku-tsl-text-to-gloss` 為 clone；跑訓練前 `git pull` 對應分支即可。

## 3. 目前資料與切分狀態

- 訓練切分（`scripts/split_data.py`，輸出 `data/splits/`，jsonl 不入庫、manifest 入庫）：
  - `exclude_rule_derived = True`（預設）：排除 609 句未經母語者裁定的 rule-derived 合成句。
  - `dev_group_leakage = 0`：dev 依對話（seg_uuid）／詞條／模板整組留存，無同源近似句跨 train/dev。
  - 組成：train 5,129／dev 553／test 33。test = Stage A 相同 33 句真實已審核句，永不進訓練。
  - 訓練集約 94% 為真實資料（文化部語料庫 4,392 ＋ 中正辭典例句 497）。
- 重跑切分：VM 上 `python3 scripts/split_data.py`（需納回 rule-derived 才加 `--include-rule-derived`，僅供實驗）。

## 4. 審核狀態與報告界線（務必遵守）

- 已完成逐筆資料審核（見 `臺灣手語全資料審核報告_2026-07-23.md`、規則實證 `臺灣手語規則實證與審核複核_2026-07-23.md`、可查證來源 `資料來源.md`）。
- 規則語序已用 5,272 句真實語料實證（`scripts/verify_rules_corpus.py`）：否定句尾 76%、時間句首 82%、情態句尾 68%、WH 句末 63%。
- **目前所有 train/dev 來源（synth 剩餘、tslcorpus、twtsl）仍為 `review_status = pending`。** 因此：

| 想宣稱的事 | 是否已可宣稱 |
|---|---|
| 「在固定 33 句真實 test 上 BLEU/EM = X」 | ✅ 有效（test Gloss 為真實標注；Text→Gloss 不輸出 NMS，不受 NMS 待審影響） |
| 「模型輸出語法正確的臺灣手語」 | ❌ 需人工評估（計畫 6.2，5 分制） |
| 「正式／最終成果」 | ❌ 需手語老師人工最終判定＋授權 |
| 對外發表／散布模型 | ❌ 需文化部語料＋中正辭典的訓練／散布授權 |

- 現階段官方定位＝**管線驗證**。跑訓練、量 BLEU 可行；宣稱「正確／正式／散布」須先補齊審核與授權。

## 5. 主要腳本

| 腳本 | 作用 |
|---|---|
| `scripts/build_jsonl.py` | 標記表 → JSONL（自有 35 句 / 38 詞 / 85 Gloss） |
| `scripts/synthesize.py` | 規則模板合成（T1–T44，967 句，增量輸出＋審核表） |
| `scripts/scrape_twtsl.py` | 中正辭典爬蟲（3,500 詞 / 544 例句） |
| `scripts/scrape_tslcorpus_full.py` | 文化部語料庫全爬（5,272 句真實平行語料） |
| `scripts/build_vocab.py` | 統一 Gloss 主詞彙表（7,002 詞）＋覆蓋率 |
| `scripts/split_data.py` | train/dev/test 切分（排除 rule-derived＋去洩漏） |
| `scripts/verify_rules_corpus.py` | 用真實語料實證規則語序（可重現） |

## 6. 階段進度

- ✅ 資料建置、爬取、合成、審核、去洩漏切分
- ✅ Stage A 提示法基線（BLEU-4 44.95 / EM 36.4%）
- ✅ Stage B QLoRA 首輪（BLEU-4 72.73 / EM 54.5%，管線驗證定位）
- ⬜ Stage B 正式版（待審核／授權補齊後）、Stage C 多任務混訓、Stage D RAG
- ⬜ 計畫 6.2 人工評估（5 分制）
