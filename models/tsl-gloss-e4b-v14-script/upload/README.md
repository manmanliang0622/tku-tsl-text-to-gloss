---
license: gemma
base_model: google/gemma-4-E4B-it
library_name: peft
pipeline_tag: translation
language:
- zh
tags:
- peft
- lora
- qlora
- gemma
- sign-language
- taiwan-sign-language
- tsl
- gloss
- constrained-decoding
metrics:
- bleu
- rouge
---

# TSL Text→Sign Script（Gemma 4 E4B QLoRA, v14 / `tsl-script-v1`）

淡江大學專題。把中文句子轉成**可執行的臺灣手語腳本**——不是自由生成 gloss 字串，
而是**從候選清單裡挑 `sign_id`**，缺必要手語時輸出 `needs_review=true`。

## ⚠️ 這個 adapter 單獨下載沒有用

輸出的是 `sign_id`（例如 `TSL_今天`），要讓它有意義，執行端必須提供**同一套**：

1. **候選檢索器**（`scripts/sign_candidates.py`）——每題從中文生成候選清單，
   放進 user 訊息。**訓練與推論必須共用同一支**，否則模型學到的候選分布與
   線上不同，約束就失效。
2. **sign_id 總表**（17,078 筆，錨定在虛擬人動作庫實際有影片的詞）——
   把 `sign_id` 對回影片檔與起訖秒數。
3. **逐字相同的 system prompt**（見下）。prompt 與訓練不一致會讓輸出整組失效：
   本專案實測過一次，33 句測試的 Exact Match 與 ValidJSON 全部掛零。

總表與檢索器不在本 repo。沒有它們，這個 adapter 只會吐出你無法解析的 ID。

## 輸入輸出格式

System（**必須逐字相同**）：

```
將繁體中文轉成可執行的臺灣手語腳本。只能使用候選清單中的 sign_id，不得創造新 ID。缺少必要手語時，必須輸出 needs_review=true。只輸出 JSON。
```

User／Assistant：

```json
{"text": "現在網路很進步，", "candidates": ["TSL_現在", "TSL_網路", "TSL_進步", ...]}
{"schema_version": "tsl-script-v1", "sign_ids": ["TSL_現在", "TSL_網路", "TSL_進步"],
 "clause_breaks": [], "needs_review": false, "oov_items": []}
```

候選數 k=40。序列中位數 380 token（最長 426），`max_len` 512 即足夠。

## 評估

**對外一律引用「完整參考」那組。** 另一組「候選內參考」把檢索撈不到的詞排除在
參考答案外，回答的是「有沒有從候選裡挑對」而非「翻譯對不對」，數值天生偏高。

| 完整參考 | test 33 | test_corpus 166 | test_textbook 423 |
|---|---:|---:|---:|
| BLEU-4 | 77.80 | 16.20 | 22.99 |
| ROUGE-L | 88.38 | 52.33 | 60.88 |
| Exact Match | 69.7% | 2.41% | 14.89% |
| 有效 JSON | 100% | 100% | 100% |
| **可播放率** | **100%** | **99.5%** | **99.15%** |
| 參考被截斷比例 | 1.3% | 30.3% | 21.3% |

- `test_corpus`＝文化部語料庫依對話群組整組留存，**主指標**。
- `test` 33 句是問候語為主的短句（參考平均 2.39 詞），**不可代表整體能力**。
- 與同專案舊格式模型（v11，自由生成 gloss）在 `test_corpus` 上的對照：
  BLEU 14.83→16.20、ROUGE-L 50.08→52.33、**可播放率 66.96%→99.5%**。
  **真正的突破是可播放率，不是 BLEU。** 候選約束讓虛擬人演不出來的詞在架構上
  就進不了輸出。⚠️ 該對照的測試集不完全相同（167→166，人工校訂前後），
  且 v14 訓練資料含 561 筆校訂——部分增益可能來自資料品質而非格式。

### needs_review 需要機率校準

貪婪解碼直接讀 `needs_review` 欄位會嚴重漏報。建議改讀該位置 true/false 之間
正規化的機率並套門檻：

| 門檻 0.02 | 正例率 | precision | recall | F1 |
|---|---:|---:|---:|---:|
| dev（門檻在此選定） | 49.6% | 0.679 | 0.734 | 0.705 |
| test_corpus | 81.9% | 0.952 | 0.728 | 0.825 |
| test_textbook | 56.0% | 0.708 | 0.574 | 0.634 |

對照貪婪解碼：test_corpus 是 precision 0.941 / **recall 0.118**。
⚠️ **門檻會隨資料分布退化**（教材集 recall 只有 0.574），換領域請在自己的
開發集上重選，不要沿用 0.02。

## 訓練配方

- 基礎模型 `google/gemma-4-E4B-it`，QLoRA 4-bit nf4 ＋ double quant、bf16 compute
- LoRA r=16 / alpha=32 / dropout=0.05，掛在 language_model 的 q/k/v/o/gate/up/down_proj
- epochs 2、batch 2、grad_accum 8、max_len 768、lr 2e-4、seed 42
- 框架 Unsloth；2h35m47s、1,116 步、8.16 秒/步、峰值顯存 11.8GB（RTX 4060 Ti 16GB）
- **選模 `checkpoint-558`（epoch 1，dev loss 0.1957）**；epoch 2 升到 0.2325 已過擬合
- 訓練資料 5,321 個相異句對（長度平衡過取樣後 8,915 列）

⚠️ Gemma 4 的 chat template 標記是 `<|turn>model\n`，不是 Gemma 3 的
`<start_of_turn>model\n`。用錯 `train_on_responses_only` 不會報錯，只會靜默
退回全序列 loss。

## 資料出處（散布須標明）

| 來源 | 用途 | 出處 |
|---|---|---|
| 文化部臺灣手語語料庫（測試版） | 主要真實平行語料 | © 文化部。訓練＋散布授權已確認合法，須標明出處 |
| 中正大學台灣手語線上辭典（第五版） | 詞彙查證、例句 | 蔡素娟、戴浩一、劉世凱、陳怡君。2026。嘉義：國立中正大學手語語言學台灣研究中心 |
| 規則模板合成句 | 資料擴充 | 專案自製，經手語老師 gloss 層審核 |

訓練語料於 2026-08-21 逐句人工校訂（6,634 列：561 筆取代 Gloss、23 筆判定排除）。
校訂只涵蓋中文語意與 Gloss 的詞彙與語序。

**本 repo 不包含任何原始語料**，僅有 adapter 權重。

## 界線與已知限制

- **不輸出也不保證 NMS（表情／搖頭／揚眉）、手形、地區變體**。那些屬影片軌，
  需母語者看影片裁定，不在本模型範圍。
- **自動指標高 ≠ 手語文法正確。** 母語者人工評估**尚未執行**。
- **最大瓶頸不是模型是檢索**：`test_corpus` 上參考 token 有 30.3% 候選裡根本
  沒有；模型在「有得挑」時命中 73.1%。缺的是「我學會了→領悟」這類語義距離遠
  的對應，字面比對與共現統計都撈不到。
- **語義 ID 讓幻覺變容易**：ID 是中文，模型可以「造」一個看起來合理的。實測
  `test_corpus` 有 0.7% 的輸出 ID 落在候選外，其中含總表查無者。
  **執行端務必驗證「ID 是否在候選清單內」**——那是可播放率的最後保證。
- 成熟度：內部候選／管線驗證，**非最終成果**，勿當通用可用模型。

## 授權

- adapter 為 **Gemma 衍生物**，散布須遵守 [Google Gemma Terms of Use](https://ai.google.dev/gemma/terms)，
  含其使用限制。使用者自負確認責任。
- 資料出處如上，散布時須一併標明。
