# Stage B v4：教師審核資料＋擴大真實 Test

日期：2026-07-25
定位：**內部正式候選（Text→Gloss 詞彙／語序層）**。NMS、手形、地區變體與模型散布授權不在本輪宣稱範圍。

## 1. 目前狀態

- ✅ VM 三個 v3 未追蹤結果已備份並核對 SHA-256；16/585 的舊 corpus 評估另行保留。
- ✅ VM `model`／`main`／`data` 已 fast-forward 至 GitHub `00d78a8`。
- ✅ 2026-07-24 老師工作簿已匯出成機器可讀 sidecar：585 筆都有判定，584 筆可作 test，`TC01419` 為重複列排除並保留正本 `TC00378`。
- ✅ 教師審核＋擴大 test 的切分已可重現並通過洩漏驗證。
- ⛔ 訓練與全量評估等待 VM 管理者修復 NVIDIA driver/library mismatch；本專案不自行重開共用 VM，也不退回 CPU 長跑。

## 2. 資料切分

正式指令：

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

擴大 test 原始候選 585 句；老師排除完全重複列 `TC01419` 後為 584 句。其候選群組與 pair 仍維持 holdout，不回流 train/dev。

| 完整性檢查 | 結果 |
|---|---:|
| train/dev 群組洩漏 | 0 |
| test_corpus 對話群組洩漏 | 0 |
| test_corpus 中文洩漏 | 0 |
| test_corpus Gloss 洩漏 | 0 |
| test_corpus `(中文, Gloss)` 洩漏 | 0 |
| 核心 test 4-gram | 5 |
| 擴大 test 4-gram | 1,070 |

切分 manifest 另保存 sidecar SHA-256 與 test ID SHA-256，確保之後重跑使用同一評測集合。

## 3. 訓練設定

GPU 修復後先執行 2-step 冒煙測試，再跑正式訓練：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 scripts/train_qlora.py \
  --output outputs/qlora_e4b_v4_teacher_holdout_smoke \
  --epochs 2 --batch 1 --grad-accum 4 --max-len 192 \
  --lr 2e-4 --seed 42 --max-steps 2

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 scripts/train_qlora.py \
  --output outputs/qlora_e4b_v4_teacher_holdout \
  --epochs 2 --batch 2 --grad-accum 4 --max-len 192 \
  --lr 2e-4 --seed 42
```

LoRA r=16、alpha=32、dropout=0.05；4-bit nf4＋double quant、bf16 compute。只依 dev `eval_loss` 在 epoch 1／2 checkpoint 中選最低者，不使用 test 選模型。

## 4. 評估

最佳 checkpoint 確認後依序執行：

```bash
python3 scripts/eval_model.py \
  --adapter outputs/qlora_e4b_v4_teacher_holdout/checkpoint-<BEST> \
  --test-file test.jsonl \
  --tag finetuned_e4b_v4_teacher_ep<BEST_EPOCH>

python3 scripts/eval_model.py \
  --adapter outputs/qlora_e4b_v4_teacher_holdout/checkpoint-<BEST> \
  --test-file test_corpus.jsonl \
  --tag finetuned_e4b_v4_teacher_ep<BEST_EPOCH>_corpus \
  --resume
```

評估摘要同列 BLEU-4、ROUGE-L、Exact Match、詞彙表內率、reference/hypothesis 4-gram 數，以及以對話群組為抽樣單位、1,000 次重抽樣的 BLEU-4 95% bootstrap CI。

| 評測集 | BLEU-4 | 95% CI | ROUGE-L | Exact Match | 詞彙表內率 |
|---|---:|---:|---:|---:|---:|
| 核心 33 句 | 待 GPU 修復 | 待評估 | 待評估 | 待評估 | 待評估 |
| 擴大 584 句 | 待 GPU 修復 | 待評估 | 待評估 | 待評估 | 待評估 |

## 5. GPU 阻礙與安全界線

2026-07-25 VM 現況：

- 執行中核心模組：`580.159.03`
- 磁碟驅動／NVML：`580.173.02`
- `torch.cuda.is_available() = False`

需由管理者重開機或安全重載驅動。恢復後必須確認兩個版本一致、NVML 正常、CUDA 可用且至少 10 GiB 顯存空閒，才可啟動訓練。

本輪可宣稱「在固定、老師文字／Gloss 層審核之真實 test 上的自動指標」；不可宣稱 NMS 正確、所有模型輸出皆為正確臺灣手語，亦不可在文化部語料與中正辭典授權未釐清前散布 adapter。
