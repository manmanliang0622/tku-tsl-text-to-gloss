# 專案交接（HANDOFF）

淡江大學專題：中文 → 臺灣手語 Gloss 翻譯模型（Gemma 4 微調）。
本檔給接手／並行的 session 快速掌握現況。最後更新：2026-07-25。

> **最新（2026-07-25）**：完成教師審核資料＋擴大真實 test 的 Stage B v4 資料管線。
> 585 句候選經既有老師判定保留 584 句，並加入可重現 sidecar、SHA-256 與 leakage 檢查。
> VM 已同步至 GitHub `00d78a8`；訓練等待管理者修復 NVIDIA driver/library mismatch。

## 1. 分支結構（工作流分流）

| 分支 | 用途 |
|---|---|
| `main` | 整合線。兩邊工作完成後併回這裡 |
| `model` | 模型／訓練工作（train_qlora、eval_model、metrics、results） |
| `data` | 資料／爬蟲／審核工作（data/、scripts/scrape_*、synthesize、build_vocab、split_data、審核報告） |

- 模型工作在 `model` 提交、資料工作在 `data` 提交，各自完成再併回 `main`；避免多 session 同時改 `main` 衝突。
- 2026-07-25 VM 三分支已先 fast-forward 對齊 GitHub `00d78a8`；v4 資料變更仍依 data → main → model 流程整合。
- 開工前務必 `git pull`；在 VM 上跑訓練前 `git checkout model && git merge --ff-only main` 對齊。

## 2. 學校 VM 與工作流

- 訓練機：學校實驗室 VM（Ubuntu 22.04、RTX 4060 Ti 16GB）。**連線位址、帳號與密碼由團隊私下保管，不寫入公開 repo**（本 repo 為 public）。
- VM 已固定主機金鑰（連線勿用 `StrictHostKeyChecking=no`）。
- **VM 沒有儲存 git 認證**：push 需手動帶 token，或走「VM 跑生成／訓練 → 檔案帶回有 gh 認證的機器 → 由該機器 push」的橋接方式。
- VM 家目錄 `~/tku-tsl-text-to-gloss` 為 clone；跑訓練前 `git pull` 對應分支即可。

## 3. 目前資料與切分狀態

- **Stage B v4 正式重跑切分指令**：`python3 scripts/split_data.py --use-teacher-reviewed --corpus-test-ratio 0.12 --seed 42`
  - 輸出 `data/splits/`；`train/dev/test.jsonl` 不入庫（可重生），只 `manifest.json` 入庫。
  - `--use-teacher-reviewed`：synth 只納入 `teacher_train_eligible`（gloss 層通過者，含 108 句修正與已 gloss-pass 的 rule-derived；排除 7 句待影片裁定者），並使用已修正 gloss。
  - 組成：**train 5,038／dev 636／核心 test 33／擴大 test 584**。
  - 擴大 test 原候選 585 句／37 對話群組；老師排除重複列 `TC01419`（保留正本 `TC00378`），最終仍有 1,070 個 4-gram。
  - 審核 sidecar：`data/splits/test_corpus_teacher_review_2026-07-24.json`；manifest 記錄 sidecar 與 test ID SHA-256。
  - 去洩漏：train/dev、test_corpus 群組、中文、Gloss、`(中文,gloss)` 交集均為 0。
- 若不需要擴大 test，舊的教師審核切分仍可用 `--use-teacher-reviewed` 重生為 train 5,680／dev 605／test 33。
- 舊行為（管線驗證，不帶旗標）：`python3 scripts/split_data.py` 仍以 confidence 排除 rule-derived；`--include-rule-derived` 僅供實驗。

## 4. 審核狀態與報告界線（務必遵守）

- AI 逐筆預審＋規則實證：見 `臺灣手語全資料審核報告_2026-07-23.md`、`臺灣手語規則實證與審核複核_2026-07-23.md`、可查證來源 `資料來源.md`。
- 規則語序已用 5,272 句真實語料實證（`scripts/verify_rules_corpus.py`）：否定句尾 76%、時間句首 82%、情態句尾 68%、WH 句末 63%。
- **手語老師人工審核（2026-07-24，已完成）**：8 類全部逐筆給處置。報告 `臺灣手語老師審核修正_2026-07-24.md`；處置寫在 reviewed jsonl 的 `teacher_*` 欄位、Excel 工作簿各分頁與 `data/synth/tsl_synth.jsonl` 的 `teacher_train_eligible`。
  - 實際修正 108 句（坐→坐車 44、上班→工作 29、不→沒有 29、補程度詞「很」6）；7 句待影片維持原樣。
  - **審核只裁定「詞彙與語序（文字層）」；NMS／手形／地區變體／逐句影片對齊屬「影片軌」，一律未冒充母語者影片通過。**
  - 因 Text→Gloss 不輸出 NMS，「NMS 待影片」不阻擋 gloss 層作訓練標的。

| 想宣稱的事 | 是否已可宣稱 |
|---|---|
| 「在固定 33 句真實 test 上 BLEU/EM = X」 | ✅ 有效（test Gloss 為真實標注；Text→Gloss 不輸出 NMS） |
| 「訓練資料 gloss 已經手語老師詞彙／語序層審核」 | ✅ 有效（synth 已審；tslcorpus／twtsl 為官方／辭典來源文字層可保留） |
| 「模型輸出語法正確的臺灣手語」 | ❌ 需人工評估（計畫 6.2，5 分制） |
| 「NMS／非手部表達正確」 | ❌ 屬影片軌，需母語者看影片，未做 |
| 對外發表／散布模型 | ❌ 需文化部語料＋中正辭典的訓練／散布授權 |

- 定位：資料 gloss 層已具老師審核基礎，可跑 Stage B 正式訓練候選並量 BLEU；宣稱「NMS 正確／對外散布」仍須補影片軌與授權。

## 5. 主要腳本

| 腳本 | 作用 |
|---|---|
| `scripts/build_jsonl.py` | 標記表 → JSONL（自有 35 句 / 38 詞 / 85 Gloss） |
| `scripts/synthesize.py` | 規則模板合成（T1–T44，967 句，增量輸出＋審核表） |
| `scripts/scrape_twtsl.py` | 中正辭典爬蟲（3,500 詞 / 544 例句） |
| `scripts/scrape_tslcorpus_full.py` | 文化部語料庫全爬（5,272 句真實平行語料） |
| `scripts/build_vocab.py` | 統一 Gloss 主詞彙表（7,002 詞）＋覆蓋率 |
| `scripts/split_data.py` | train/dev/test 切分（`--use-teacher-reviewed` 用教師審核結果＋去洩漏） |
| `scripts/verify_rules_corpus.py` | 用真實語料實證規則語序（可重現） |

## 6. 階段進度

- ✅ 資料建置、爬取、合成、AI 預審、去洩漏切分
- ✅ **手語老師人工審核（8 類全數）＋依審核受控重切**（train 5,680／dev 605／test 33）
- ✅ Stage A 提示法基線（BLEU-4 44.95 / EM 36.4%）
- ✅ Stage B QLoRA 首輪（BLEU-4 72.73 / EM 54.5%，管線驗證定位）
- ✅ Stage B v3 核心 33 句（EM 63.6% / ROUGE-L 82.37）；舊 585 句評估保留 16 筆歷史結果，不續跑
- ✅ **Stage B v4 資料與穩定 BLEU 評估管線**（教師審核＋584 句真實 test＋group bootstrap CI）
- ⛔ **Stage B v4 訓練／評估**：等待管理者修復 VM NVIDIA 580.159.03／580.173.02 版本不一致
- ⬜ 影片軌：NMS／手形／地區變體由母語者看影片裁定（獨立於 Text→Gloss）
- ⬜ 授權：文化部語料＋中正辭典訓練／散布書面依據
- ⬜ Stage C 多任務混訓、Stage D RAG、計畫 6.2 人工評估（5 分制）

## 7. 接手第一步（TL;DR）

1. `git pull`；資料工作切 `data`、模型工作切 `model`，完成再 ff 併回 `main`。
2. 重生 v4 切分：`python3 scripts/split_data.py --use-teacher-reviewed --corpus-test-ratio 0.12 --seed 42`。
3. GPU preflight：`nvidia-smi` 正常、`torch.cuda.is_available()` 為 true、可用顯存至少 10 GiB；否則不啟動訓練。
4. 要看審核細節：`臺灣手語老師審核修正_2026-07-24.md`＋ `outputs/.../reviewed_data_2026-07-23/`（`teacher_*` 欄位）＋工作簿 `臺灣手語全資料逐筆審核_2026-07-23.xlsx`。
5. 守界線：可講 BLEU 與「gloss 詞彙／語序層已老師審核」；**不可**講「NMS 正確」或「可對外散布」。
