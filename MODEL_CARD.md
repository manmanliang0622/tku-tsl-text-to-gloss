---
title: Model Card — TSL Text→Gloss (Gemma 4 E4B QLoRA, v11 holdout)
updated: 2026-08-13
supersedes: v4 teacher-reviewed (2026-08-04)
---

# Model Card：中文 → 臺灣手語 Gloss 翻譯（Gemma 4 E4B QLoRA v11）

> **2026-08-13 修訂**：本卡先前停在 v4，引用的是「擴大 584 句 BLEU 18.61」。
> 那批 584 句其後被證實**多數已在訓練集內**，該數字不代表泛化能力，已整段
> 汰換為 v11 在兩個留存測試集上的數字。詳見
> [results/stageB_v11_generalization_report.md](results/stageB_v11_generalization_report.md)。

淡江大學專題。將中文句子翻譯為**臺灣手語 Gloss 詞序**的 QLoRA adapter。
本卡記錄**訓練配方、資料出處與授權、評估結果與重現方式**。權重二進位（adapter）目前留在訓練 VM，
未放入本 repo；依下方配方可從本 repo 重跑得到同一個 adapter。

## 定位與界線（先讀）

- **是什麼**：Text→Gloss 的**詞彙／語序層**候選模型。輸入中文、輸出 Gloss token 序列（以 `/` 分隔）。
- **不是什麼**：**不輸出、也不保證 NMS（非手部標記：表情／搖頭／揚眉）、手形、地區變體**。這些屬「影片軌」，需母語者看影片裁定，不在本模型範圍。
- **成熟度**：內部候選 / 管線驗證，**尚非最終成果**，勿當成通用可用模型。
  未見過的語料庫長句 Exact Match 僅 **1.20%**、論文例句 **13.29%**；
  核心 33 句的 69.70% 是**問候語為主的短句**（參考答案平均 2.39 詞），
  不可拿來代表整體能力。
- **已知學到什麼**：語序（錯誤率僅 2–5%）、有效 JSON 99–100%、
  不照抄中文（未知詞 73%→34%，見基線對照）。
  **沒學到的是詞彙**——選詞錯誤＋未知詞佔全部錯誤的 60–81%。
- ⚠️ **疑問類型正確率不要單獨引用**（本卡先前列為「93–99%」，已撤回）。
  以修正後標籤重算為 94.61%／98.60%，但資料極不平衡——全部猜「none」就有
  91.02%／97.20%，**實際只贏過多數決基線 +3.59／+1.40**。否定正確率同理未複驗。
- 自動指標高 ≠ 手語文法正確；正式品質須經計畫 6.2 手語老師 5 分制人工評估，
  **該評估尚未執行**。

## 基礎模型

- `google/gemma-4-E4B-it`（Gemma 4 E4B，instruction-tuned）。
- **授權提醒**：本 adapter 為 Gemma 衍生物，散布 Gemma 衍生權重須遵守 **Google Gemma Terms of Use**（需隨附條款、含使用限制）。公開權重前請先確認 Gemma 條款；本卡不代為認定。

## 資料與出處（散布須標明）

| 來源 | 用途 | 出處／授權 |
|---|---|---|
| 文化部臺灣手語語料庫（測試版） | 主要真實平行語料 | © 文化部臺灣手語語料庫。訓練＋散布授權已確認合法（2026-08-04），須標明出處 |
| 中正大學台灣手語線上辭典（第五版） | 詞彙查證、例句 | 蔡素娟、戴浩一、劉世凱、陳怡君。2026。《台灣手語線上辭典（中文版第五版）》。嘉義：國立中正大學手語語言學台灣研究中心。訓練＋散布授權已確認合法（2026-08-04），須標明出處 |
| 自有標記表 | 自有 35 句／38 詞 | 專案自製 |
| 規則模板合成句 | 資料擴充 | 合成；經手語老師 2026-07-24 gloss 層審核（108 句修正，7 句待影片者排除） |

> 授權詳見 [資料來源.md](資料來源.md) 檔首「授權更新（2026-08-04）」。
> **授權 ≠ 品質背書**：NMS／手形／地區變體正確性仍待母語者影片裁定。

## 訓練配方（可重現）

### 1) 切分
```bash
python3 scripts/split_data.py --use-all --length-balance --papers-as-test \
    --corpus-test-ratio 0.12 --corpus-test-min-len 6
python3 scripts/build_json_targets.py --splits train dev test test_corpus test_papers
```
- 組成：train **5,347 個相異句對**（長度平衡過取樣後 8,992 列）／dev 666／
  核心 test 33／`test_corpus` 167／`test_papers` 143。
  **對外一律寫「5,347 句」，不可寫「8,992 句」**——後者含 3,645 列刻意複製。
- 長度平衡：≤4 詞 ×1、5–7 詞 ×2、≥8 詞 ×4，用於矯正輸出過短的偏差。
- synth 只納入 `teacher_train_eligible`（108 句教師修正生效；7 句待影片者不進訓練）。
- 去洩漏〔2026-08-13 複驗〕：三個測試集與 train 的中文、`(中文,Gloss)` 重疊
  **皆為 0**；dev 有 8 句中文與 train 相同但 Gloss 不同（標籤噪音，尚未處置）。
- Sidecar：`data/splits/test_corpus_teacher_review_2026-07-24.json`
  （SHA-256 `4f305cc44c37ed4c329b71c009f4418ce6c3c744ac1532e164cb7ea62f5a549a`）。

### 2) 訓練
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 scripts/train_qlora.py \
  --model google/gemma-4-E4B-it --output outputs/qlora_e4b_v11_holdout \
  --target json --epochs 2 --batch 2 --grad-accum 4 --max-len 192 --lr 2e-4 --seed 42
```
- QLoRA：LoRA **r=16、alpha=32、dropout=0.05**；4-bit **nf4** ＋ double quant、**bf16** compute。
  （r=32 試過，dev loss 反而略差 0.7268 vs 0.7061，見 v7 實驗。）
- target modules：`language_model` 的 `q/k/v/o/gate/up/down_proj`。
- **選模**：僅依最低 dev loss → `checkpoint-1124`（epoch 1，dev loss 0.2233）。
  `save_total_limit=None` 保留每個 epoch 的 checkpoint 供事後挑選，**未用 test 選模**。

### 3) 評估
```bash
python3 scripts/eval_json_model.py --adapter outputs/qlora_e4b_v11_holdout/checkpoint-1124 \
    --tag v11_test_corpus --test-file test_corpus.jsonl --ple gpu
```
⚠️ `--ple gpu` 必要：`device_map` 含任何 `"cpu"` 項目會讓 accelerate 掛 offload
hook，每 token 慢到約 35 秒（全放 GPU 是 0.06 秒/token）。

## 評估結果（v11）

**對外報告一律用 `test_corpus` / `test_papers`；核心 33 句僅作歷史對照。**

| 指標 | 核心 33（短句） | `test_corpus` 167 | `test_papers` 143 |
|---|---:|---:|---:|
| Exact Match | 69.70% | **1.20%** | **13.29%** |
| ROUGE-L | 82.71 | 50.08 | 57.08 |
| BLEU-4 | 62.96 | 14.83 | 17.71 |
| 可播放率 | 91.67% | 66.96% | 78.20% |
| 　（參考答案天花板） | 100% | 67.09% | 74.96% |
| 有效 JSON | 100% | 100% | 99.30% |
| 疑問類型正確 | 96.97% | 93.41% | 98.60% |
| 否定正確 | 100% | 95.81% | 97.20% |

`test_papers` 的 EM／ROUGE-L／BLEU 為 **2026-08-13 修正參考答案後**的重算值
（原為 11.19／55.23／16.51）——39 句參考答案曾把論文的替代詞串成 Gloss 序列。

### 分維度錯誤分析

| 錯誤型態 | `test_corpus` | `test_papers` |
|---|---:|---:|
| 選詞（Gloss替換） | 47.90% | 37.76% |
| 未知詞（OOV） | 33.53% | 22.38% |
| 漏詞 | 12.57% | 6.99% |
| **語序** | **2.40%** | **4.90%** |
| 亂加詞 | 0.60% | 10.49% |
| 完全錯誤 | 1.80% | 4.20% |
| 正確 | 1.20% | 13.29% |

**語序不是瓶頸，詞彙覆蓋才是。** 三層診斷（不同測試集、不同方法）獨立得到
同一結論，見 [results/three_tier_report.md](results/three_tier_report.md)。
OOV 判定基準為「詞彙總表 ∪ 訓練詞彙」（13,663 詞），非訓練詞彙——後者會把
合法但訓練未出現的手語詞誤判成造詞，實測高估 67%。

### 與未微調基線的對照（2026-08-13，教授回饋第 4 點）

同一套推論堆疊（同 4-bit 設定、同 chat template、greedy 無 beam），**唯一差別是有沒有掛 adapter**；310 句 × 3 策略 = 930 次生成。完整表見 [results/baseline_vs_finetuned_report.md](results/baseline_vs_finetuned_report.md)。

| | zero | rules | fewshot | **v11** |
|---|---:|---:|---:|---:|
| `test_corpus` ROUGE-L | 34.89 | 35.98 | 36.66 | **50.08** |
| `test_papers` ROUGE-L | 48.62 | 53.32 | 48.19 | **57.08** |
| `test_corpus` 未知詞句佔比 | 73.05% | 59.28% | 70.06% | **33.53%** |
| `test_papers` 未知詞句佔比 | 47.55% | 32.87% | 50.35% | **22.38%** |

沒有任何一格是基線勝出。但**幅度必須誠實講**：真實對話長句 ROUGE-L +13.42，
論文短句只有 +3.76、可播放率只差 1.51 個百分點——**論文型短句用規則提示就能
逼近微調**，微調的價值集中在規則罩不住的長篇對話。

微調真正改變的是「用不用手語詞」：基線的典型輸出是逐字照抄中文，未知詞句佔
47–73%，微調後降到 22–34%。

### 標籤品質（2026-08-13 稽核，教授回饋第 3 點）

JSON 目標的 `question_type`／`negation`／`nonmanual`／`topic`／`verb` **全部由
啟發式規則產生**（`build_json_targets.py`），不是人工標註。稽核結果：

| 欄位 | 狀況 |
|---|---|
| `question_type` | 曾把 gloss 裡當**話題標記**用的「什麼」判成疑問詞。train 標為疑問的 932 句中 203 句中文根本不是問句；test_corpus 24 個疑問標籤有 10 個可疑。**2026-08-13 已修**（改為「中文出現 WH 即可，只在 gloss 出現須位於句末」），可疑標籤降至 train 6%／test_corpus 0% |
| `nonmanual` | 隨 `question_type` 連動，故同樣有 203 句陳述句被配上「疑問表情」。已隨上項修正 |
| `verb` | 同一 bug 導致 111 句主要動詞被抽成「什麼」，已降至 2 句 |
| `verb_type` | **8,992 句全部是 `unknown`——死欄位，從未填值** |
| `agreement` | **8,992 句全部是 `none`——死欄位**。教授點名的「空間方向」屬此；論文的 `(j→i)` 標記未灌入。這是已知取捨，非遺漏 |

⚠️ 修正只改了標籤產生規則，**v11/v12 是用舊標籤訓練的**，模型仍帶有該偏差：
標籤被改掉的 12 句 test_corpus 中，模型有 5 句仍預測舊的錯標籤。要消除需重訓，
本期不做。

### 尚未具備的證據

- **母語者人工評估無有效結果**。2026-08-22 曾以 v14 輸出產表並回收一輪
  盲測（100 題），但評分不符量表設計，**結果不採用、不引用**
  （內部紀錄見 results/human_eval_v14script_report.md）。
  語意維度自動指標答不了，故本卡不宣稱語意正確性。
- **訓練資料未經母語者抽查**。表已產出（`outputs/訓練資料抽查表_train.xlsx`，
  150 題，`scripts/make_data_audit_sheet.py` 五層分層抽樣），同樣尚未送出。
  train 5,347 句中，合成句只有 85 句有審核紀錄，其餘皆為 `pending`
  ——**「資料已經老師審核」這句話不可對外宣稱**。
- **NMS 不在評估範圍**。本卡列的疑問／否定正確率是**文字層標記**，
  不等於表情、搖頭、揚眉等實際動作，後者需母語者看影片裁定。

- 詳細設定與錯誤分析：[results/stageB_v11_generalization_report.md](results/stageB_v11_generalization_report.md)、
  [教授回饋對帳與D1進度_2026-08-13.md](教授回饋對帳與D1進度_2026-08-13.md)。

## 如何取得 adapter 權重

adapter 目前只在訓練 VM：`outputs/qlora_e4b_v11_holdout/checkpoint-1124`（`outputs/` 為 gitignore，不入庫）。
另有 v12（上下文版）`outputs/qlora_e4b_v12_context/checkpoint-1124`——實測僅
+0.6pp EM／+1.04 BLEU，疑問類型反退 3.6pp，訓練成本翻倍，**不建議採用**。

兩種方式：
1. **重跑**：依上方三步（seed 42 固定）即可重現同一 adapter。
2. **取檔**：有 VM 存取權者
   `scp -r tku-gpu:.../qlora_e4b_v11_holdout/checkpoint-1124 ./`。
   若日後要公開權重，建議發到 Hugging Face Hub（內建 LFS）並附本卡出處與 Gemma 條款。

## 引用

- 文化部臺灣手語語料庫（測試版）。文化部。<https://tslcorpus.moc.gov.tw/>
- 蔡素娟、戴浩一、劉世凱、陳怡君。2026。《台灣手語線上辭典（中文版第五版）》。嘉義：國立中正大學手語語言學台灣研究中心。<https://twtsl.ccu.edu.tw/>
- 基礎模型：Google Gemma 4 E4B（依 Gemma Terms of Use）。
