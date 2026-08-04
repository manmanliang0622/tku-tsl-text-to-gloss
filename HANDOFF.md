# 專案交接（HANDOFF）

淡江大學專題：中文 → 臺灣手語 Gloss 翻譯模型（Gemma 4 微調）。
本檔供接手／並行 session 快速掌握現況。最後更新：2026-07-26。

> **最新（2026-07-26）**：Stage B v4 教師審核資料、正式 2 epochs 訓練及核心／擴大 test 評估均已完成。
> 最佳模型為 epoch 1 `checkpoint-630`；核心 33 句 BLEU-4 80.00（95% CI 65.74–88.77），教師通過的
> 584 句／37 群組 BLEU-4 **18.61**（95% CI **15.48–22.49**）、ROUGE-L 55.40、EM 9.25%。
> 584 句結果已逐筆驗證並由 JSONL 獨立重算 summary；完整評估耗時 13 小時 28 分 36 秒。

## 1. 分支與版本狀態

| 分支 | 用途 |
|---|---|
| `main` | 整合線；資料與模型工作完成後 fast-forward 回此分支 |
| `model` | 模型、訓練、評估、metrics 與 results |
| `data` | 資料、爬蟲、審核、詞彙表與 split |

- GitHub 同步基準：`00d78a8`。
- Stage B v4 資料 commit：`6a05cf8`。
- Stage B v4 初始模型／資料整合 commit：`68e84cd`。
- 本輪結果提交前的 VM `model` 基準：`d5ed2ea`（Stage B v4 訓練與評估進度文件）。
- 最終結果提交後 `model` 領先 `origin/model` 5 commits；公開遠端仍停在 `00d78a8`。
- 資料與文件依常規推送並 fast-forward `main`。訓練出的 adapter 目前留在 VM 為**團隊操作決定**（非授權限制——資料授權已允許訓練與散布，見 §8）。

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

Adapter 目前只保留在 VM、暫不推送（團隊操作決定；資料授權已允許散布，見 §8）。

## 5. 評估結果

| 指標 | 核心 33 句 | 擴大 584 句 |
|---|---:|---:|
| BLEU-4 | 80.00 | **18.61** |
| BLEU-4 95% CI | 65.74–88.77 | **15.48–22.49** |
| ROUGE-L | 73.92 | 55.40 |
| Exact Match | 48.48% | 9.25%（54/584） |
| 自有 85 詞詞彙表內率 | 86.59% | 16.83% |
| 聯集詞彙表內率 | 95.12% | 69.74% |
| reference／hypothesis 4-gram | 5／3 | 1,070／901 |
| bootstrap 群組數 | 33 | 37 |

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

評估於 2026-07-26 00:39:48 啟動、14:08:24 完成，wall-clock 13 小時 28 分 36 秒。
程序正常退出，無續跑或重啟；輸出每批 8 筆 flush。

完成驗證：

- JSONL 584 筆皆可解析、ID 唯一，順序及 ID SHA-256 均與固定 test 完全一致。
- `chinese`、`ref`、`group` 逐筆等於 `test_corpus.jsonl`，無空預測。
- 37 個群組、1,070 個 reference 4-gram；bootstrap 固定 1,000 次、seed 42、group 為抽樣單位。
- 以同一 `scripts/metrics.py` 與詞彙表從 JSONL 獨立重算，所有 summary 欄位完全一致。
- 預測 JSONL SHA-256：`d651612159a95ad4bd127470abd992e7fd05c501e347c277c9317028c5b290e3`。
- Summary SHA-256：`7305f819859c60c29d6f7151e051f932bffb3ad108b26019740f272a6d4dc8af`。

v3 的 `results/finetuned_e4b_v3_ep1_corpus_test.jsonl` 僅有 16 筆歷史結果，必須保留，但禁止改名或混入 v4。

核心 CI 寬 23.03 點，擴大 test CI 寬 7.01 點；584 句提供較穩定的 corpus BLEU
估計。兩套 test 的句型、長度與難度不同，分數不可當成同分布下的模型退步比較；
正式泛化描述應以涵蓋較多真實對話的 584 句結果為主。

## 6. 評估效能決策

`scripts/eval_model.py` 新增 `--batch-size`，預設仍為 1；本次 v4 明確使用 8。

- Batch 8 已完整跑完核心 33 句，並和保留的單筆 v4 前三句逐字一致。
- Batch 32 曾使 MKLDNN 嘗試配置約 40 GiB CPU RAM；VM 僅 30 GiB，雖 fallback 成功仍不安全，因此禁止用於正式 584 句。
- Gemma 4 E4B 的 Per-Layer Embedding 需 CPU offload，自回歸生成很慢；584 句實際耗時 13 小時 28 分 36 秒。
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
- ✅ Stage B v4 教師通過 584 句評估與穩定 BLEU CI（18.61，95% CI 15.48–22.49）
- ⬜ 計畫 6.2 手語老師人工評估（5 分制）
- ⬜ 影片軌：NMS／手形／地區變體由母語者看影片裁定
- ✅ 文化部語料與中正辭典訓練／散布授權：**已確認合法（2026-08-04，負責人確認），條件為標明出處**
- ⬜ Stage C 多任務混訓、Stage D RAG

## 9. 接手第一步（TL;DR）

1. Stage B v4 自動評估已完成並驗證；不再用固定 test 調參或選模。
2. 下一個模型品質步驟為計畫 6.2：將固定 v4 輸出交由手語老師／聾人評估者做 5 分制人工評估。
3. NMS、手形、地區變體及下游可播放性須走獨立影片軌，由母語者看片裁定。
4. 文化部語料與中正辭典的訓練／散布授權已確認合法（2026-08-04），**標明出處即可**（辭典：蔡素娟等 2026，中正大學；語料：文化部臺灣手語語料庫）。
5. Adapter 目前留在 VM、暫不推遠端（團隊操作決定，非授權限制）；本輪小型結果與 bundle 已帶回本機。
6. 可宣稱「老師文字／Gloss 層審核」與固定 test 自動指標；**授權允許對外散布（需標明出處）**。仍不可宣稱 NMS 正確或所有輸出皆為正確臺灣手語（品質界線，屬影片軌）。
