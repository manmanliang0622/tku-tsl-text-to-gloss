# Stage B：QLoRA 微調報告（Gemma 4 E4B）

日期：2026-07-21～22（學校 VM）
對應計畫：《臺灣手語翻譯語言模型_微調訓練計畫.md》第 5 節 Stage B、第 6 節評估、6.4 對照組。

## 1. 結論（先講重點）

QLoRA 微調在 33 句真實測試集上**大幅超越** Stage A 的最佳提示法，BLEU-4 從 44.95 提升到 **72.73**、Exact Match 從 36.4% 提升到 **54.5%**。在同一套推理管線的受控對照下（未微調基礎模型 vs 微調），BLEU-4 從 33.21 翻倍到 72.73。這驗證了計畫核心假設，也與 SignAlignLM（ACL 2025）「小模型微調勝過大模型純提示」的結論一致 [來源1]。

## 2. 結果對照

測試集 = 與 Stage A 完全相同的 33 句真實已審核句（P01–P05＋S01–S30，排除模板句），**全程未參與訓練**。

| 系統 | 推理管線 / 提示 | BLEU-4 | ROUGE-L | Exact Match | 詞彙表內率(聯集) |
|---|---|---|---|---|---|
| Stage A 基線（best） | Ollama，未微調，fewshot（規則＋9 示例） | 44.95 | 65.59 | 36.4% | 94.3% |
| 受控基礎模型 | HF，未微調，精簡 prompt | 33.21 | 68.15 | 39.4% | 89.3% |
| **Stage B 微調（ep2）** | **HF，QLoRA 微調，精簡 prompt** | **72.73** | **78.48** | **54.5%** | **97.6%** |

兩種對照都指向同一結論：微調帶來決定性提升。受控對照（同 HF 管線、同精簡 prompt）最能隔離「微調本身」的貢獻：BLEU-4 33.21 → 72.73（+39.5）。

## 3. 訓練設定

| 項目 | 設定 |
|---|---|
| 基礎模型 | `google/gemma-4-E4B-it`（非 gated，Apache 2.0；多模態 `Gemma4ForConditionalGeneration`） |
| 方法 | QLoRA（4-bit nf4 + double quant，bf16 compute），LoRA r=16 α=32 dropout=0.05 |
| LoRA 目標 | 僅語言模型的 q/k/v/o/gate/up/down_proj（regex 排除視覺/音訊塔的 Gemma4ClippableLinear） |
| 資料 | `data/splits`：train 5,788（tslcorpus 4,405＋synth 886＋twtsl 例句 497）、dev 503、test 33 |
| 超參數 | 3 epochs、batch 1（有效 batch=grad_accum 8）、lr 2e-4 cosine、warmup 0.03、max_len 192 |
| Loss | completion-only（prompt 段 label=-100，只在 Gloss 段算 loss） |
| 硬體 | 學校 VM，RTX 4060 Ti 16GB（**共用生產 GPU**，另有 root 服務佔 ~5.4GB） |
| 可訓練參數 | 34.9M（0.44%） |
| 訓練時間 | 3 epochs / 2,172 步 / 約 74 分鐘 |

## 4. 關鍵發現：3 epochs 過擬合，epoch 2 最佳

各 epoch 的 dev（eval）loss：

| epoch | eval_loss |
|---|---|
| 1 | 0.957 |
| **2** | **0.940（最佳）** |
| 3 | 1.258（回升，過擬合） |

因此採用 **checkpoint-1448（epoch 2）** 做最終評估。這說明對此資料規模，2 epochs 已足夠，3 epochs 開始過擬合——後續正式訓練建議設 2 epochs 或以 dev loss early stopping。

## 5. 錯誤分析（微調 ep2，18/33 完全正確）

15 句未完全匹配，分類：

- **可接受的同義變體（嚴格 EM 低估）**：S10 我/懂（ref 我/知道）、S21 現在/幾時（ref 現在/幾點）——語意等價，屬 TSL 合理變體。
- **短句多生成**：S02 早/安靜（ref 早安）、S03 晚安/睡覺、S04 謝謝/棒、S12 可以/可以——單詞句仍會多吐字。
- **語序**：S28 貴/太（ref 很/貴）、S18 我/身體/不/舒服（ref 我/舒服/不）。
- **詞形/切分**：S20 我/看/醫生/要（ref 我/看醫生/要，切分差異）、S13 這/不行（ref 不可以）。

相較 Stage A 主要錯在「詞彙選擇」（模型不知標準詞形），微調後**地名應用句（P01–P05）與多數常用句已正確**，殘留誤差集中在單詞句多生成與少數同義/語序差異。部分「錯誤」實為可接受變體，凸顯計畫 6.2 節人工評估的必要（自動 EM 會低估真實品質）。

## 6. 檔案

- `results/finetuned_e4b_ep2_test.jsonl`、`results/summary_finetuned_e4b_ep2.json`：微調模型逐句預測與指標
- `results/base_e4b_minimal_test.jsonl`、`results/summary_base_e4b_minimal.json`：受控基礎模型對照
- `scripts/train_qlora.py`、`scripts/eval_model.py`、`scripts/split_data.py`：可重跑
- 訓練 adapter：VM `outputs/qlora_e4b/checkpoint-1448`（epoch 2，最佳；139MB，未入庫）

## 7. 工程備註（Gemma 4 E4B 在共用 GPU 上的 QLoRA）

E4B 的 Per-Layer Embedding（`embed_tokens_per_layer`，5.6GB bf16）bnb 量化不到，4-bit 載入仍佔 9.3GB。解法：device_map 把 PLE 放 CPU、移除 accelerate 搬表 hook、改寫其 forward 讓查表在 CPU 進行只回傳 ~11MB 到 GPU，GPU 常駐降到 ~3.7GB。另外因 Gemma 262k 巨大詞彙的 logits 記憶體大，訓練 prompt 去掉 7 條規則（微調後不需規則拐杖），max_len 192、batch 1，峰值留 ~2.3GB free 給共用機的生產服務。細節見 `scripts/train_qlora.py` 註解。

## 8. 下一步

- Stage C：多任務混訓（混一般中文指令資料防災難性遺忘，計畫 5-C）。
- Stage D：推理端 RAG 詞彙約束（計畫 5-D）。
- 人工評估（計畫 6.2）：邀手語老師／聾人顧問對輸出 5 分制評分，並審核訓練用的合成/語料庫資料（目前 review_status=pending）。
- 以 2 epochs 重訓正式版（避免 epoch 3 過擬合）。
