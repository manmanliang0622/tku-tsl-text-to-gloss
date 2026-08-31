---
title: Model Card — TSL Text→Script (Gemma 4 E4B QLoRA, v17script_k40sem)
updated: 2026-08-30
supersedes: v11 holdout (2026-08-13)
---

# Model Card：中文 → 臺灣手語腳本（Gemma 4 E4B QLoRA v17）

> **2026-08-30 改版**：本卡先前停在 **v11**，記錄的是舊的 Gloss 字串輸出格式與
> 該版指標。線上生成端自 2026-08-27 起是 **v17＋約束解碼**，輸出格式也已改為
> 候選 `sign_id` 的結構化 JSON（`tsl-script-v1`）。整卡依 v17 重寫；v11 的內容
> 不再適用，歷史數字見 git 紀錄與
> [results/stageB_v11_generalization_report.md](results/stageB_v11_generalization_report.md)。

淡江大學專題「譯手通 SignTranslate」。把中文句子翻成**臺灣手語詞彙 ID 序列**的
QLoRA adapter，輸出直接驅動 3D 虛擬人逐詞播放。本卡記錄**訓練配方、資料出處與
授權、評估結果與重現方式**。權重二進位不入 repo（`outputs/` 為 gitignore），
取得方式見文末。

## 定位與界線（先讀）

- **是什麼**：**候選挑選＋排序**模型。輸入一句中文與該句的 40 個候選 `sign_id`，
  輸出挑出並排好序的 `sign_ids`、子句切點與品質預警旗標。
- **不是什麼**：**不輸出、也不保證 NMS（非手部標記：表情／搖頭／揚眉）、手形、
  地區變體**。這些屬「影片軌」，需母語者看影片裁定，不在本模型範圍。
- **成熟度**：內部候選／管線驗證，**尚非最終成果**。語料庫留存長句的完全正確率
  僅 **0.60%**、教材集 **15.60%**；核心 33 句的 66.67% 是**問候語為主的短句**，
  不可拿來代表整體能力。
- **已知學到什麼**：語序與輸出格式（有效 JSON 100%）、從候選中挑詞。
  **沒學到的是詞彙**——錯誤仍以選詞為大宗。
- 自動指標高 ≠ 手語文法正確；正式品質須經計畫 6.2 手語老師 5 分制人工評估，
  **該評估尚未執行**。

## 基礎模型

- `google/gemma-4-E4B-it`（Gemma 4 E4B，instruction-tuned）。
- **授權（2026-08-22 已查證）**：Gemma 4 適用 **Apache License 2.0**，不適用
  Gemma Terms of Use（該條款明文僅涵蓋 Gemma 1–3n）。公開 adapter 在 Google
  授權側無阻擋，僅需標明衍生來源；查證依據見
  [Gemma條款查證_2026-08-22.md](Gemma條款查證_2026-08-22.md)。

## 資料與出處（散布須標明）

| 來源 | 用途 | 出處／授權 |
|---|---|---|
| 文化部臺灣手語語料庫（測試版） | 主要真實平行語料 | © 文化部臺灣手語語料庫。訓練＋散布授權已確認合法（2026-08-04），須標明出處 |
| 中正大學台灣手語線上辭典（第五版） | 詞彙查證、例句、論文例句 | 蔡素娟、戴浩一、劉世凱、陳怡君。2026。《台灣手語線上辭典（中文版第五版）》。嘉義：國立中正大學手語語言學台灣研究中心。訓練＋散布授權已確認合法（2026-08-04），須標明出處 |
| 自有標記表 | 核心 33 句真實錄影與自製詞彙標記 | 專案自製 |
| 規則模板合成句 | 資料擴充 | 合成；經手語老師 2026-07-24 gloss 層審核（108 句修正，7 句待影片者排除） |

> 授權詳見 [資料來源.md](資料來源.md) 檔首「授權更新（2026-08-04）」。
> **授權 ≠ 品質背書**：NMS／手形／地區變體正確性仍待母語者影片裁定。

## 訓練配方（可重現）

### 1) 切分

```bash
python3 scripts/split_data.py --use-all --length-balance --no-papers \
  --textbook-as-test --corpus-test-ratio 0.12 --corpus-test-min-len 6 --seed 42
```

| split | 列數 | 說明 |
|---|---:|---|
| `train` | 8,915 | 長度平衡過取樣後；**相異句對 5,321、相異中文 5,283** |
| `dev` | 663 | 選 checkpoint 與校準 needs_review 門檻 |
| `test`（核心 33） | 33 | 真實錄影、歷代可比 |
| `test_corpus` | 166 | 語料庫留存長句 |
| `test_textbook` | 423 | 教材集，最大也最接近展示情境 |

- **對外一律寫「5,321 句對」，不可寫「8,915 句」**——後者含刻意複製的過取樣列。
- 長度平衡用於矯正輸出過短的偏差。
- 去洩漏〔2026-08-30 複驗〕：三個測試集與 train 的中文、`(中文,Gloss)` 重疊
  **皆為 0**；dev 有 6 句中文與 train 相同但 Gloss 不同（標籤噪音，尚未處置）。

### 2) 建候選資料集（tsl-script-v1）

```bash
.venv-emb/bin/python3 scripts/build_script_dataset.py \
  --splits train dev test test_corpus test_textbook \
  --k 40 --n-sem 8 --compact --out data/splits_script_k40sem
```

每句附 **40 個候選 `sign_id`**，其中最多 8 個名額由語義向量檢索填入，其餘為字面／
例句遷移／詞對齊／核心詞／干擾項。`--n-sem` 需 `.venv-emb`（bge-small-zh-v1.5）。

⚠️ **`data/splits_script_k40sem/` 無法逐位元重現**。建資料當時的候選排序會受
Python set 迭代順序（隨 `PYTHONHASHSEED` 變動）影響，同一條指令連跑兩次會有少數
句子不同。此缺陷已於 2026-08-30 修正（commit 64ad7c5），**日後**建的資料集可重現，
但用修正後的程式重建 v17 資料只有約 8 成句子與原檔相同。v17 的資料集是已提交的
既有產物，不需重新產生。

### 3) 訓練

```bash
python3 scripts/train_script_qlora.py --verify-v17
```

| 項目 | 值 | 項目 | 值 |
|---|---|---|---|
| epochs | 2（**採 epoch 1**） | 學習率 | 2e-4 |
| 有效 batch | 16（2 × 8 累積） | 最大長度 | 768 tokens |
| LoRA r / α | 16 / 32 | LoRA dropout | 0.05 |
| 量化 | 4-bit nf4＋double quant | compute dtype | bf16 |
| 隨機種子 | 42 | 總步數 | 1,116 |
| 訓練時間 | 156.98 分鐘 | 峰值顯存 | 11.80 GB |

- 框架：**Unsloth ＋ TRL SFT**（PyTorch 2.7.1、CUDA 12.6），硬體 RTX 4060 Ti 16GB。
- **選模**：僅依 dev loss。epoch 1 = **0.19101**、epoch 2 = **0.22790**（上升＝過擬合），
  故採 **checkpoint-558**。訓練 loss 全程平均 0.1109（曲線由 1.036 降至 0.036）。
  **未用 test 選模。**
- target modules 見 `train_script_qlora.TARGET_MODULES`；可訓練參數 36,700,160。
  改動該常數會打到視覺／音訊塔，`--verify-v17` 會核對參數量擋下。
- 完整紀錄：`outputs/qlora_e4b_v17script_k40sem/unsloth_run.json`。

### 4) 評估

```bash
python3 scripts/eval_script_format.py --pred results/v17cd_test_textbook.jsonl \
    --threshold 0.039707
```

`tests/test_eval_script_format.py` 會拿 `results/v17cd_*` 重跑並與既有
`*_scriptmetrics.json` 逐欄比對（4 個資料集 × 43 欄全同），確保評分管線可重現。

## 評估結果（v17）

### 兩種參考口徑，引用時務必標明

- **Full_reference**：以 `data/splits/<split>.jsonl` 的完整 `gloss_text` 為參考，
  **含檢索撈不到的詞**——這才是系統整體水準。
- **候選內參考**：只拿有進候選的參考詞比對，回答「有沒有從候選裡挑對」。
  兩者差很大：`test_corpus` 有 **30.18%** 的參考詞從未進入候選。

**以下一律為 Full_reference。** 「v17」為模型本身，「v17＋約束」為線上實際部署組合。

| 測試集 | 版本 | BLEU-4 | ROUGE-L | EM% |
|---|---|---:|---:|---:|
| 核心 33（短句） | v17 ／ ＋約束 | 72.68 ／ 72.68 | 87.04 ／ 87.04 | 66.67 ／ 66.67 |
| `test_corpus` 166 | v17 ／ ＋約束 | 16.61 ／ 16.36 | 53.09 ／ 52.92 | 0.60 ／ 0.60 |
| `test_textbook` 423 | v17 ／ ＋約束 | 24.67 ／ 24.47 | 61.30 ／ 61.16 | 15.60 ／ 15.60 |
| `dev` 663 | v17 ／ ＋約束 | 27.98 ／ 27.87 | 62.31 ／ 62.19 | 28.05 ／ 27.90 |

### 輸出紀律：約束解碼把缺陷歸零

推論時把 `sign_ids` 鎖在該句候選清單上（`scripts/constrained_decode.py`）。
2026-08-31 起服務端與離線推論都 **import 同一份實作**，不再各留一份副本；
`tests/test_serve_parity.py` 改為守「不得再內嵌」。

| 指標 | v17 | v17＋約束 |
|---|---:|---:|
| 詞彙違規率（列）corpus／textbook | 5.42% ／ 5.67% | **0.0% ／ 0.0%** |
| 未知 sign_id（corpus／textbook） | 8 ／ 29 | **0 ／ 0** |
| ValidSignID%（corpus／textbook） | 99.18% ／ 98.37% | **100% ／ 100%** |
| 有效 JSON（全集合） | 100% | 100% |

品質代價為零（±0.2 BLEU 屬雜訊）。另加退化守衛：同一 `sign_id` 最多連續 6 次、
陣列最多 18 個元素（取自 10,200 句參考實測極值）。

> **2026-08-31 更正：上表的「可播放率 100%」是名不副實的，已改名為
> `ValidSignID%`。** 它只檢查「預測的 ID 存不存在於總表」，沒有檢查影片能不能
> 正常播、動作完不完整。實際比對 0813 的影片品質掃描後，動作庫 17,085 支裡
> 只有 43.7% 的品質判定為 ok，39.8% 是 severe（幾乎整段偵測不到舉手動作）。
> 指標已拆成四層（`scripts/eval_script_format.py`），v17cd 的真實數字是：
>
> | 資料集 | ValidSignID% | QualityPlayable% | ok／品質差／不堪用 |
> |---|---:|---:|---|
> | 核心 33 | 100.0 | **92.31** | 60／0／5 |
> | test_corpus | 100.0 | **96.93** | 913／34／30 |
> | test_textbook | 100.0 | **94.11** | 1621／73／106 |
> | dev | 100.0 | **96.35** | 2283／66／89 |
>
> `CompositionSuccess%`（多動作串接後能否正常播）尚未實作——那要在 0813
> 虛擬人端實際串起來播一次才量得到，不在本 repo 範圍。目前一律回報 `null`，
> **不可**解讀為「串接已驗證沒問題」。

### candidate_coverage_risk 校準（候選覆蓋風險）

> **2026-08-31 正名：原名 `needs_review`。** 這個旗標的正解是「參考 Gloss 有沒有
> 全部落進候選清單」，純粹是檢索覆蓋率訊號。它**偵測不到**候選完整但選錯詞、
> 語序錯、重複／複合遺失、整句語意不自然、NMS 或影片品質問題。舊名會讓人
> 以為模型具備翻譯品質預警能力，實際上沒有。schema v2 起欄位名為
> `candidate_coverage_risk`（`scripts/script_schema.py`）；線上 API 兩個鍵都回，
> `needs_review*` 僅為相容別名。

模型對每句輸出覆蓋風險機率，供前端提示「這句建議人工確認」。門檻只在 dev 上選，
不看測試集調。**線上採用 0.039707**（2026-08-27 改以最大化 F1 選定）：

| 門檻 | dev F1 | corpus F1 | textbook F1 | dev 漏放行 |
|---|---:|---:|---:|---:|
| 0.095349（舊規則：recall≥0.7 下最大 precision） | 0.702 | 0.827 | 0.653 | 93 |
| **0.039707（現行：最大化 F1）** | **0.741** | **0.905** | **0.702** | **23** |

偏 recall 是刻意的——漏放行會讓錯句直接送去給虛擬人比出來，誤攔只是多一次人看。
重選門檻用 `scripts/nr_threshold.py`。

### 與未微調基線的對照（核心 33 句）

| | zero-shot | rules | few-shot | **v17** |
|---|---:|---:|---:|---:|
| BLEU-4 | 39.50 | — | 44.95 | **72.68** |
| EM% | 27.3 | — | 36.4 | **66.67** |

來源 [results/stageA_report.md](results/stageA_report.md)（同一套推論堆疊，
唯一差別是有沒有掛 adapter）。**這是短句集，不可外推**：未微調在語料庫留存句上
的 BLEU-4 僅 6.74（見 v17 報告 §6.1），微調後也只到 16.61——長句仍是弱項。

## 部署到 0821_bundle 的必帶檔案

`serve_model.py` 依賴 `scripts/` 底下數個本地模組，少一個就起不來。清單由
`serve_model.BUNDLE_MODULES` 宣告，並由 `scripts/check_bundle_deps.py`
以 AST 算出遞移相依對帳（CI 會跑，加了 import 忘了更新清單就會紅）。

```bash
python3 scripts/check_bundle_deps.py --list          # 印出必帶檔案
python3 scripts/check_bundle_deps.py --deploy-check  # 印出部署端的驗證指令
```

> **2026-08-31**：`constrained_decode.py` 與 `script_schema.py` 是本日新增的
> 相依，**目前不在 0821_bundle 裡**。線上跑的仍是 517141b 那版（無此相依，
> 安全但比 branch 舊）。下次部署 serve_model 時這兩個檔必須一起帶，否則啟動
> 即死——`script_schema` 是模組層 import，會最先炸。

## 已知限制

- **切分曾有表面形式洩漏（2026-08-31 已修）**：`split_data.py` 原本只比對
  原始中文字串，核心 33 句有 3 句（`我住在台北。`／`我知道`／`我不知道`）
  去標點後就在 train，dev 與 train 更有 6 句原字串完全相同。**上表的 v14–v17
  數字都是在那份舊切分上量的。** 三個測試集（核心 33、test_corpus、
  test_textbook）修正後位元完全相同，所以測試集數字仍可比；但 dev 變了
  （663→548），依 dev 選出的 checkpoint 與門檻 0.095349／0.039707 都應在
  新 dev 上重驗。舊切分凍結於 `data/splits_v17/`。
- **測試集已被當成 validation 使用**：v14／v17／k40／k60／語義通道／pin_core
  等方案反覆參考 corpus、textbook 與核心 33 的結果，最後依教材集表現決定部署
  v17（見 `results/v17cd_deploy_note.md`）。這三個集合實際上已是開發集，
  **不應再宣稱為 unbiased final test**。真正的 final holdout 尚未建立。
- **重複、複合與方位資訊在建 `gloss_text` 時就流失（仍未修）**：twtsl 來源的
  `gloss_raw` 有 758 個方位標記（`_A`／`_B`／`_S`／`_N`），`gloss_text` 只剩 8 個。
  train 另有 478 列含重複記號 `++`、447 列含複合 `+`，正規化時第二段被直接
  刪除 664 次（多為 `X+手機`／`X+杯子` 這類分類詞結構）。需重新設計輸出 schema，
  並由手語老師裁定 `樹+見` 該算兩個動作還是一個複合詞。
- **`clause_breaks` 已修好，但訓練訊號極薄**（2026-08-31）：欄位原本在 8,915 列
  訓練資料中**全部是空陣列**——先用 `/` 切 gloss_text 再找 `//`，而 `//` 在切分
  那一步就沒了；且把語料庫的重複記號 `++` 誤當成子句邊界。現改為承接上游本來
  就有的 `clauses` 欄位（`split_data` 帶進切分、`build_script_dataset` 換算成
  `sign_ids` 索引）。
  **但覆蓋率是 train 1.1%／dev 0.73%**：只有中正辭典例句有 `clauses`，而其中
  504/544 是單子句，全資料集真正有邊界的僅 40 筆。模型固定輸出 `[]` 仍能對
  99% 的題目，**不可宣稱本模型具備子句切分能力**。要真的做這件事需要對語料庫
  補子句標註。
- **長句泛化是主要弱點**：`test_corpus` 完全正確率 0.60%，錯誤以選詞為大宗。
- **訓練與上線的候選清單組成不同**：訓練與評估的候選含語義向量名額，
  **線上服務未載入向量模型**（`0821_bundle` 的 `sign_candidates.py` 連 `n_sem`
  參數都沒有），改以純字面檢索補滿 40 個。2026-08-30 實測每句約 8 個候選相異
  （重疊率 corpus 90.6%、textbook 84.4%）；參考詞可及率由 100% 降為 99.0%／98.5%，
  約 5% 的句子少撈到至少一個參考詞。**上表數字是在訓練側候選分布下量得，
  線上實際表現可能略低。**
- **NMS 不在評估範圍**：本模型不輸出表情、搖頭、揚眉、手形與地區變體。
  原始計畫（`臺灣手語翻譯語言模型_微調訓練計畫.md` §1）要求 Gloss 與 NMS
  雙輸出，目前實作只做候選排序，等於偏離了原題目。**而且補不回來**：
  2026-08-31 清點三份訓練來源，NMS 標註只存在於合成句（967 筆中 589 筆，
  且只有 3 種模板字串），tslcorpus 5,272 筆與 twtsl 544 筆**完全沒有 NMS 欄位**。
  拿合成句的 NMS 訓練只會學回產生它的那三條規則。要真正做 NMS，必須對
  5,816 筆真實語料重新人工標註，需要具臺灣手語能力的標註者——這是資源決定，
  不是程式問題。
- **母語者人工評估無有效結果**：2026-08-22 曾以 v14 輸出回收一輪盲測（100 題），
  評分不符量表設計，**結果不採用、不引用**。
- **訓練資料未經母語者抽查**：抽查表已產出（`outputs/訓練資料抽查表_train.xlsx`）
  但尚未送出。合成句僅少數有審核紀錄——**「資料已經老師審核」這句話不可對外宣稱**。

## 如何取得 adapter 權重

- 訓練 VM：`outputs/qlora_e4b_v17script_k40sem/checkpoint-558`
  （`adapter_model.safetensors` 146,888,168 bytes）。
- 線上部署副本：`~/0821_bundle/model_service/checkpoint`（同一權重）。
- **重跑取得**：`python3 scripts/train_script_qlora.py --verify-v17`。已驗證忠實
  （2026-08-27）：eval_loss 與 wall time 差異全部落在 1.2% 以內，量級與 bf16
  跨行程非決定性相同，adapter 權重檔大小完全一致。
- 若日後要公開權重，建議發到 Hugging Face Hub 並附本卡出處
  （基礎模型 Apache 2.0）。

## 引用

- 文化部臺灣手語語料庫（測試版）。文化部。<https://tslcorpus.moc.gov.tw/>
- 蔡素娟、戴浩一、劉世凱、陳怡君。2026。《台灣手語線上辭典（中文版第五版）》。嘉義：國立中正大學手語語言學台灣研究中心。<https://twtsl.ccu.edu.tw/>
- 基礎模型：Google Gemma 4 E4B（Apache License 2.0）。
