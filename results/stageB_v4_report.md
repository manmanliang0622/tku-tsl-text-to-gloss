# Stage B v4：教師審核資料＋擴大真實 Test

日期：2026-07-26
定位：**內部正式候選（Text→Gloss 詞彙／語序層）**。NMS、手形、地區變體與模型散布授權不在本輪宣稱範圍。

## 1. 執行摘要

- ✅ 2026-07-24 老師工作簿已匯出為機器可讀 sidecar；585 筆候選中 584 筆可作 test，重複列 `TC01419` 排除並指向保留正本 `TC00378`。
- ✅ 教師審核切分可重現：train 5,038、dev 636、核心 test 33、擴大 test 584；所有群組、中文、Gloss 與 pair 洩漏檢查均為 0。
- ✅ 管理者重開 VM 後，NVIDIA 核心模組、驅動與 NVML 已一致為 `580.173.02`；CUDA 恢復正常。
- ✅ 2-step 冒煙測試及正式 2 epochs QLoRA 訓練完成，無 OOM、NaN 或 CUDA 錯誤。
- ✅ 只依 dev `eval_loss` 選出 epoch 1 的 `checkpoint-630`，核心 33 句評估已完成。
- ✅ 教師通過的 584 句評估已以 batch 8、獨立 v4 tag 與 `--resume` 完成；BLEU-4 18.61（37 群組 bootstrap 95% CI 15.48–22.49）。

VM 評估程式基準為 `model` commit `c362c33`，結果文件更新前基準為 `d5ed2ea`。舊 v3 的 16/585 筆 corpus 預測仍獨立保留，未混入 v4。

## 2. 資料與切分

正式重生指令：

```bash
python3 scripts/split_data.py \
  --use-teacher-reviewed \
  --corpus-test-ratio 0.12 \
  --seed 42
```

| split | 句數 | 說明 |
|---|---:|---|
| train | 5,038 | 教師通過 synth＋文化部語料＋中正辭典例句 |
| dev | 636 | 依對話／詞條／模板整組留存 |
| test | 33 | Stage A 起固定的核心自有真實句 |
| test_corpus | 584 | 文化部真實語料，37 個對話群組，老師文字／Gloss 層通過 |

擴大 test 原始候選為 585 句／37 群組。老師排除完全重複列 `TC01419` 後輸出 584 句；排除列及其 pair 仍留在 leakage blocklist，不回流 train/dev。

| 完整性檢查 | 結果 |
|---|---:|
| train/dev 群組洩漏 | 0 |
| test_corpus 對話群組洩漏 | 0 |
| test_corpus 中文洩漏 | 0 |
| test_corpus Gloss 洩漏 | 0 |
| test_corpus `(中文, Gloss)` 洩漏 | 0 |
| 核心 test reference 4-gram | 5 |
| 擴大 test reference 4-gram | 1,070 |

審核 sidecar SHA-256：
`4f305cc44c37ed4c329b71c009f4418ce6c3c744ac1532e164cb7ea62f5a549a`

最終 584 筆 test ID SHA-256：
`c10b42b59698c46374d33bc9b43a2de777e03eda8cda8869f650c326218c57c8`

教師修正的 108 句仍生效；7 句待影片裁定者不進 synth 訓練池。所有輸出中文與 Gloss 皆非空。

## 3. GPU 修復與訓練紀錄

2026-07-25 重開機後的 preflight：

- GPU：NVIDIA GeForce RTX 4060 Ti 16GB
- 核心模組／磁碟驅動／NVML：`580.173.02`
- PyTorch：`2.7.1+cu126`
- `torch.cuda.is_available() = True`
- 啟動前可用顯存：約 10,378–10,379 MiB

2-step 冒煙測試使用 batch 1、gradient accumulation 4、max length 192；2/2 steps 完成，adapter 與 tokenizer 正常儲存。

正式訓練指令：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 scripts/train_qlora.py \
  --output outputs/qlora_e4b_v4_teacher_holdout \
  --epochs 2 \
  --batch 2 \
  --grad-accum 4 \
  --max-len 192 \
  --lr 2e-4 \
  --seed 42
```

LoRA r=16、alpha=32、dropout=0.05；4-bit nf4＋double quant、bf16 compute。正式訓練共 1,260 steps，耗時約 2,611 秒（43 分 31 秒），最終 train loss 3.459。

| checkpoint | epoch | dev eval_loss | dev 評估耗時 | 是否入選 |
|---|---:|---:|---:|---|
| `checkpoint-630` | 1 | **0.9821940660** | 47.44 秒 | ✅ 最佳 |
| `checkpoint-1260` | 2 | 1.0088934898 | 47.49 秒 | — |

模型選擇只依 dev loss；核心 test 與擴大 test 均未參與 checkpoint 決策。

## 4. 評估結果與進度

核心 33 句以最佳 checkpoint 評估：

```bash
python3 scripts/eval_model.py \
  --adapter outputs/qlora_e4b_v4_teacher_holdout/checkpoint-630 \
  --test-file test.jsonl \
  --tag finetuned_e4b_v4_teacher_ep1 \
  --batch-size 8 \
  --bootstrap-samples 1000 \
  --bootstrap-seed 42
```

| 評測集 | BLEU-4 | 95% CI | ROUGE-L | Exact Match | 自有詞彙表內率 | 聯集詞彙表內率 |
|---|---:|---:|---:|---:|---:|---:|
| 核心 33 句 | **80.00** | 65.74–88.77 | 73.92 | 48.48% | 86.59% | 95.12% |
| 擴大 584 句 | **18.61** | **15.48–22.49** | 55.40 | 9.25% | 16.83% | 69.74% |

核心 test 只有 5 個 reference 4-gram，bootstrap 單位實際為 33 個 singleton 群組，因此 CI 寬 23.03 點。擴大 test 含 1,070 個 reference 4-gram，37 個真實對話群組的 CI 寬 7.01 點，提供更穩定的 corpus BLEU 估計。

同一 checkpoint 在兩套 test 上的差距反映資料難度與涵蓋範圍：核心集多為短句與常用句，擴大集包含較長、較多樣的真實敘事及對話。兩者不是同分布抽樣，不把 80.00→18.61 描述成模型在訓練後退步；對較廣真實語料的泛化能力應以 584 句結果為主。

擴大 test 使用獨立 tag，不讀取 v3 的 16 筆歷史結果：

```bash
python3 scripts/eval_model.py \
  --adapter outputs/qlora_e4b_v4_teacher_holdout/checkpoint-630 \
  --test-file test_corpus.jsonl \
  --tag finetuned_e4b_v4_teacher_ep1_corpus \
  --batch-size 8 \
  --resume \
  --bootstrap-samples 1000 \
  --bootstrap-seed 42
```

評估於 2026-07-26 00:39:48（Asia/Taipei）啟動，14:08:24 完成，wall-clock 為 13 小時 28 分 36 秒；程序正常退出，未曾中斷或續跑。

完成後驗證：

- 584 筆 JSONL 均可解析，ID 唯一、順序與固定 ID SHA-256 `c10b42b59698c46374d33bc9b43a2de777e03eda8cda8869f650c326218c57c8` 完全一致。
- `chinese`、`ref`、`group` 逐筆等於 `test_corpus.jsonl`；37 群組、零空預測。
- 以相同 `scripts/metrics.py`、雙層詞彙表、1,000 次 group bootstrap 與 seed 42 獨立重算，與 summary 全欄位一致。
- 預測／summary SHA-256 分別為 `d651612159a95ad4bd127470abd992e7fd05c501e347c277c9317028c5b290e3`、`7305f819859c60c29d6f7151e051f932bffb3ad108b26019740f272a6d4dc8af`。

### 批次評估的資源決策

Gemma 4 E4B 的 Per-Layer Embedding 需 CPU offload，自回歸生成速度遠慢於訓練。新增 `--batch-size` 後：

- batch 8 已用核心 33 句驗證，可在現有 30 GiB RAM 與共用 GPU 服務下完成。
- batch 32 診斷曾使 MKLDNN 嘗試配置約 40 GiB CPU 記憶體；雖 fallback 成功，但不符合共用 VM 安全界線，因此正式 corpus 評估固定使用 batch 8。
- 單筆、batch 8 與 batch 32 診斷結果保留在 `results/stageB_v4_diagnostics/`，不列入正式指標。

## 5. 錯誤分析

自動指標之外仍可觀察到下列問題：

- **多餘或重複詞**：如 `你好/我`、`謝謝/謝謝`、`對不起/我/對不起`。
- **詞彙或語意錯誤**：如「早安」產生 `早/安靜`、「我不舒服」產生 `我/不對`。
- **語序／片語邊界差異**：如 `慢/請`、`我/看醫生/要`，需依老師接受度判定。
- **可能可接受的近義變體**：如 `再見/拜拜`、`不行/禁止`、`現在/幾時`，不宜只用 Exact Match 判錯。

這些案例說明 BLEU、ROUGE-L 與 EM 只能作自動比較；「語法正確」與可接受變體仍需計畫 6.2 的手語老師人工評分。

擴大 584 句有 54 句完全匹配；平均 reference 4.70 tokens、hypothesis 4.41 tokens。模型輸出較 reference 短 218 句、等長 236 句、較長 130 句，沒有空預測，4 句有相鄰重複 token。聯集詞彙表內率 69.74% 顯示真實敘事中的低頻詞、複合詞及標記差異仍是主要限制；後續人工評估應優先區分可接受變體、Gloss 缺失／多增、語序錯誤與真正的表外詞。

## 6. 產物與宣稱界線

- VM adapter：`outputs/qlora_e4b_v4_teacher_holdout/checkpoint-630`
- 核心預測：`results/finetuned_e4b_v4_teacher_ep1_test.jsonl`
- 核心摘要：`results/summary_finetuned_e4b_v4_teacher_ep1.json`
- 擴大評估：`results/finetuned_e4b_v4_teacher_ep1_corpus_test.jsonl`
- 擴大摘要：`results/summary_finetuned_e4b_v4_teacher_ep1_corpus.json`

本輪可宣稱「在固定、老師文字／Gloss 層審核之真實 test 上的自動指標」。不可宣稱 NMS 正確、所有模型輸出皆為正確臺灣手語，亦不可在文化部語料與中正辭典授權未釐清前散布 adapter。
