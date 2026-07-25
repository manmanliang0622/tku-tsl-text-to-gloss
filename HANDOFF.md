# 專案交接（HANDOFF）

淡江大學專題：中文 → 臺灣手語 Gloss 翻譯模型（Gemma 4 微調）。
本檔供接手／並行 session 快速掌握現況。最後更新：2026-07-26。

> **最新（2026-07-26）**：Stage B v4 教師審核資料、正式 2 epochs 訓練與核心 33 句評估已完成。
> 最佳模型為 epoch 1 `checkpoint-630`；核心 BLEU-4 80.00（95% CI 65.74–88.77）、ROUGE-L 73.92、EM 48.48%。
> 教師通過的 584 句／37 對話群組正在 VM 以 batch 8、獨立 v4 tag 與 `--resume` 評估，尚未產生完整 summary。

## 1. 分支與版本狀態

| 分支 | 用途 |
|---|---|
| `main` | 整合線；資料與模型工作完成後 fast-forward 回此分支 |
| `model` | 模型、訓練、評估、metrics 與 results |
| `data` | 資料、爬蟲、審核、詞彙表與 split |

- GitHub 同步基準：`00d78a8`。
- Stage B v4 資料 commit：`6a05cf8`。
- Stage B v4 初始模型／資料整合 commit：`68e84cd`。
- VM `model` 目前 HEAD：`c362c33`（加入安全的 batch generation 評估）。
- `model` 目前領先 `origin/model` 3 commits；尚未推送公開 repo。
- 資料修改依 data → main → model 整合；584 句完成前先不 fast-forward 本輪未完成結果回 `main`。

## 2. 學校 VM 與安全界線

- 訓練機：Ubuntu 22.04、NVIDIA GeForce RTX 4060 Ti 16GB；SSH alias 為 `tku-gpu`。
- 連線位址、帳號與憑證不寫入 public repo；SSH 已固定主機金鑰，不使用 `StrictHostKeyChecking=no`。
- VM repo：`/home/b310ai/tku-tsl-text-to-gloss`。
- 管理者重開機後，核心模組、磁碟驅動與 NVML 均為 `580.173.02`；PyTorch `2.7.1+cu126`，CUDA 可用。
- 訓練／評估啟動前門檻：NVML 正常、`torch.cuda.is_available() == True`、至少 10 GiB 可用顯存。
- 不停止共用服務、不自行重開 VM、不覆蓋既有 adapter。GPU 被其他工作占用或顯存不足時停止啟動新工作。
- VM 沒有 git push 認證；採「VM 執行 → 小型結果與 bundle 帶回本機 → 由有認證的機器推送」。

## 3. 教師審核資料與切分

正式切分指令：

```bash
python3 scripts/split_data.py \
  --use-teacher-reviewed \
  --corpus-test-ratio 0.12 \
  --seed 42
```

| split | 句數 |
|---|---:|
| train | 5,038 |
| dev | 636 |
| 核心 test | 33 |
| 擴大 test | 584 |

- Synth 只納入 `teacher_train_eligible`；108 句教師修正已生效，7 句待影片裁定者不進 synth 訓練池。
- 擴大 test 原候選為 585 句／37 群組。老師排除重複列 `TC01419`，保留正本 `TC00378`，最終為 584 句與 1,070 個 reference 4-gram。
- Sidecar：`data/splits/test_corpus_teacher_review_2026-07-24.json`。
- Sidecar SHA-256：`4f305cc44c37ed4c329b71c009f4418ce6c3c744ac1532e164cb7ea62f5a549a`。
- 584 筆 test ID SHA-256：`c10b42b59698c46374d33bc9b43a2de777e03eda8cda8869f650c326218c57c8`。
- Train/dev 與 test_corpus 的群組、中文、Gloss、`(中文, Gloss)` 洩漏均為 0；排除的重複 pair 仍留在 blocklist，不回流 train/dev。

老師審核只裁定文字層詞彙與語序。NMS、手形、地區變體及影片對齊仍屬獨立影片軌，不得描述為已通過母語者影片審核。

## 4. Stage B v4 訓練結果

2-step 冒煙測試先以 batch 1、gradient accumulation 4、max length 192 執行，2/2 steps 通過，無 OOM、NaN 或 CUDA 錯誤。

正式設定：

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

- Gemma 4 E4B QLoRA：LoRA r=16、alpha=32、dropout=0.05；4-bit nf4＋double quant、bf16 compute。
- 共 1,260 steps，約 2,611 秒（43 分 31 秒），最終 train loss 3.459。
- `checkpoint-630`：epoch 1 dev loss **0.9821940660**。
- `checkpoint-1260`：epoch 2 dev loss 1.0088934898。
- 最佳 checkpoint 僅由最低 dev loss 決定，因此固定使用 `outputs/qlora_e4b_v4_teacher_holdout/checkpoint-630`；未用 test 選模。

Adapter 只保留在 VM，不放入 git bundle、不推送或散布。

## 5. 評估結果與執行中工作

核心 33 句已完成：

| 指標 | 結果 |
|---|---:|
| BLEU-4 | 80.00 |
| BLEU-4 95% CI | 65.74–88.77 |
| ROUGE-L | 73.92 |
| Exact Match | 48.48% |
| 自有 85 詞詞彙表內率 | 86.59% |
| 聯集詞彙表內率 | 95.12% |
| reference／hypothesis 4-gram | 5／3 |

核心產物：

- `results/finetuned_e4b_v4_teacher_ep1_test.jsonl`
- `results/summary_finetuned_e4b_v4_teacher_ep1.json`
- 詳細設定與錯誤分析：`results/stageB_v4_report.md`

584 句長評估執行命令：

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

截至 2026-07-26 00:45（Asia/Taipei）：

- 背景程序 PID 記錄於 `results/finetuned_e4b_v4_teacher_ep1_corpus_eval.pid`。
- 預測輸出為 `results/finetuned_e4b_v4_teacher_ep1_corpus_test.jsonl`；每批 8 筆 flush。
- Summary 完成後會寫到 `results/summary_finetuned_e4b_v4_teacher_ep1_corpus.json`。
- 第一批尚未 flush，完整 584 句指標仍為 pending；不得拿部分 JSONL 計算並宣稱最終 BLEU。
- 若程序中斷，先確認無舊 PID存活，再用完全相同的 tag、checkpoint、batch size 與 `--resume` 續跑。

v3 的 `results/finetuned_e4b_v3_ep1_corpus_test.jsonl` 僅有 16 筆歷史結果，必須保留，但禁止改名或混入 v4。

## 6. 評估效能決策

`scripts/eval_model.py` 新增 `--batch-size`，預設仍為 1；本次 v4 明確使用 8。

- Batch 8 已完整跑完核心 33 句，並和保留的單筆 v4 前三句逐字一致。
- Batch 32 曾使 MKLDNN 嘗試配置約 40 GiB CPU RAM；VM 僅 30 GiB，雖 fallback 成功仍不安全，因此禁止用於正式 584 句。
- Gemma 4 E4B 的 Per-Layer Embedding 需 CPU offload，自回歸生成很慢；584 句預估約 24–36 小時。
- 單筆、batch 8 與 batch 32 的診斷檔集中在 `results/stageB_v4_diagnostics/`，不作正式指標。

## 7. 主要腳本

| 腳本 | 作用 |
|---|---|
| `scripts/split_data.py` | 教師審核切分、sidecar、manifest 與 leakage 驗證 |
| `scripts/train_qlora.py` | Gemma 4 E4B QLoRA 訓練、seed 與 smoke `max_steps` |
| `scripts/eval_model.py` | 核心／corpus greedy generation、batch 與 resume |
| `scripts/metrics.py` | BLEU-4、ROUGE-L、EM、詞彙表內率、group bootstrap CI |
| `scripts/verify_rules_corpus.py` | 以 5,272 句真實語料實證規則語序 |

## 8. 階段進度

- ✅ 資料建置、爬取、合成、AI 預審、去洩漏切分
- ✅ 手語老師文字層審核與 108 句修正
- ✅ Stage A 提示法基線（BLEU-4 44.95／EM 36.4%）
- ✅ Stage B QLoRA 首輪（BLEU-4 72.73／EM 54.5%，管線驗證定位）
- ✅ Stage B v3 核心 33 句；舊 585 句評估保留 16 筆歷史結果
- ✅ Stage B v4 教師審核資料、可重現切分、group bootstrap 管線
- ✅ Stage B v4 2 epochs 訓練與核心 33 句評估
- ⏳ Stage B v4 教師通過 584 句評估與穩定 BLEU CI
- ⬜ 計畫 6.2 手語老師人工評估（5 分制）
- ⬜ 影片軌：NMS／手形／地區變體由母語者看影片裁定
- ⬜ 文化部語料與中正辭典訓練／模型散布書面授權
- ⬜ Stage C 多任務混訓、Stage D RAG

## 9. 接手第一步（TL;DR）

1. 先確認長評估程序是否仍存活，以及 v4 corpus JSONL 已完成幾筆；不要同時啟動第二個相同 tag 的程序。
2. 程序中斷時以第 5 節命令及 `--resume` 續跑；不可混用 v3 的 16 筆結果。
3. 完成後驗證 JSONL 為 584 筆、ID 唯一且與 `test_corpus.jsonl` 完全相符，並由 JSONL 重算 summary。
4. 將 584 句 BLEU-4、37 群組 bootstrap 95% CI、ROUGE-L、EM、詞彙表內率及耗時補入 `stageB_v4_report.md`。
5. 只提交小型 JSONL、summary、report 與 handoff；adapter 留在 VM。重建 bundle 並複製回本機後，再依授權決定是否推送。
6. 可宣稱「老師文字／Gloss 層審核」與固定 test 自動指標；不可宣稱 NMS 正確、所有輸出皆為正確臺灣手語或可對外散布。
