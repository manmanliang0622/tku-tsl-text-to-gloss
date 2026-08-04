---
title: Model Card — TSL Text→Gloss (Gemma 4 E4B QLoRA, v4 teacher-reviewed)
updated: 2026-08-04
---

# Model Card：中文 → 臺灣手語 Gloss 翻譯（Gemma 4 E4B QLoRA v4）

淡江大學專題。將中文句子翻譯為**臺灣手語 Gloss 詞序**的 QLoRA adapter。
本卡記錄**訓練配方、資料出處與授權、評估結果與重現方式**。權重二進位（adapter）目前留在訓練 VM，
未放入本 repo；依下方配方可從本 repo 重跑得到同一個 adapter。

## 定位與界線（先讀）

- **是什麼**：Text→Gloss 的**詞彙／語序層**候選模型。輸入中文、輸出 Gloss token 序列（以 `/` 分隔）。
- **不是什麼**：**不輸出、也不保證 NMS（非手部標記：表情／搖頭／揚眉）、手形、地區變體**。這些屬「影片軌」，需母語者看影片裁定，不在本模型範圍。
- **成熟度**：內部候選 / 管線驗證。核心 33 句指標高，但涵蓋較廣真實對話的 584 句 BLEU 僅 18.61；**尚非最終成果**，勿當成通用可用模型。
- 自動指標高 ≠ 手語文法正確；正式品質須經計畫 6.2 手語老師 5 分制人工評估。

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
python3 scripts/split_data.py --use-teacher-reviewed --corpus-test-ratio 0.12 --seed 42
```
- 組成：train 5,038／dev 636／核心 test 33／擴大 test 584。
- synth 只納入 `teacher_train_eligible`（108 句教師修正生效；7 句待影片者不進訓練）。
- 去洩漏：train/dev/test 的群組、中文、Gloss、`(中文,Gloss)` 洩漏均為 0；擴大 test 排除重複列 `TC01419`、保留正本 `TC00378`。
- Sidecar：`data/splits/test_corpus_teacher_review_2026-07-24.json`
  （SHA-256 `4f305cc44c37ed4c329b71c009f4418ce6c3c744ac1532e164cb7ea62f5a549a`）。

### 2) 訓練
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 scripts/train_qlora.py \
  --model google/gemma-4-E4B-it \
  --output outputs/qlora_e4b_v4_teacher_holdout \
  --epochs 2 --batch 2 --grad-accum 4 --max-len 192 --lr 2e-4 --seed 42
```
- QLoRA：LoRA **r=16、alpha=32、dropout=0.05**；4-bit **nf4** ＋ double quant、**bf16** compute。
- target modules：`language_model` 的 `q/k/v/o/gate/up/down_proj`。
- 共 1,260 steps，約 43 分 31 秒，最終 train loss 3.459。
- **選模**：僅依最低 dev loss →
  `checkpoint-630`（epoch 1，dev loss 0.9821940660）＜ `checkpoint-1260`（epoch 2，1.0088934898）。**未用 test 選模**。

### 3) 評估
```bash
python3 scripts/eval_model.py \
  --adapter outputs/qlora_e4b_v4_teacher_holdout/checkpoint-630 \
  --test-file test_corpus.jsonl --tag finetuned_e4b_v4_teacher_ep1_corpus \
  --batch-size 8 --resume --bootstrap-samples 1000 --bootstrap-seed 42
```

## 評估結果

| 指標 | 核心 33 句 | 擴大 584 句 |
|---|---:|---:|
| BLEU-4 | 80.00 | **18.61** |
| BLEU-4 95% CI | 65.74–88.77 | **15.48–22.49** |
| ROUGE-L | 73.92 | 55.40 |
| Exact Match | 48.48% | 9.25%（54/584） |
| 聯集詞彙表內率 | 95.12% | 69.74% |
| bootstrap 群組數 | 33 | 37 |

- 兩套 test 句型／長度／難度不同，分數不可當同分布比較；泛化描述以 584 句為主。
- 預測 JSONL SHA-256 `d651612159a95ad4bd127470abd992e7fd05c501e347c277c9317028c5b290e3`；
  Summary SHA-256 `7305f819859c60c29d6f7151e051f932bffb3ad108b26019740f272a6d4dc8af`。
- 詳細設定與錯誤分析：[results/stageB_v4_report.md](results/stageB_v4_report.md)。

## 如何取得 adapter 權重

adapter 目前只在訓練 VM：`outputs/qlora_e4b_v4_teacher_holdout/checkpoint-630`（`outputs/` 為 gitignore，不入庫）。

兩種方式：
1. **重跑**：依上方三步（seed 42 固定）即可重現同一 adapter。
2. **取檔**：有 VM 存取權者
   `scp -r tku-gpu:.../qlora_e4b_v4_teacher_holdout/checkpoint-630 ./`。
   若日後要公開權重，建議發到 Hugging Face Hub（內建 LFS）並附本卡出處與 Gemma 條款。

## 引用

- 文化部臺灣手語語料庫（測試版）。文化部。<https://tslcorpus.moc.gov.tw/>
- 蔡素娟、戴浩一、劉世凱、陳怡君。2026。《台灣手語線上辭典（中文版第五版）》。嘉義：國立中正大學手語語言學台灣研究中心。<https://twtsl.ccu.edu.tw/>
- 基礎模型：Google Gemma 4 E4B（依 Gemma Terms of Use）。
