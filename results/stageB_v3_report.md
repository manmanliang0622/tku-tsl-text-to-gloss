# Stage B v3：擴大真實 test 集重訓報告

日期：2026-07-24～25（學校 VM）
定位：**管線驗證**（訓練資料為當時 review_status=pending 的版本，不得宣稱「正確／正式／可散布」）。
對應：《微調訓練計畫》第 5 節 Stage B、6.1 自動評估；HANDOFF.md 第 3–4 節。

## 1. 本輪目的

Stage B v2 發現 **BLEU-4 在 n=33 的核心 test 上極不穩定**（全集僅 5 個 4-gram，單句語序小差即讓分數腰斬）。v3 的目的是建立一個**夠大的真實 test 集**讓 BLEU 穩定下來。

做法（`scripts/split_data.py --corpus-test-ratio`）：從文化部語料庫**依對話群組（seg_uuid）整組留存** 585 句真實聾人 Gloss 作 `test_corpus`，整段對話移出訓練池。

| 檢查項 | 結果 |
|---|---|
| test_corpus 句數 / 對話數 | 585 句 / 37 段對話 |
| 4-gram 數（核心 test → test_corpus） | 5 → **1,070**（214 倍） |
| 對話群組洩漏 | **0** |
| 句子層級與 train/dev 重疊 | **0** |

因 585 句原本在訓練集內，留存後**必須重訓**，即本輪 v3。

## 2. v3 訓練設定與切分

**切分（本輪實際使用，對應 commit 771d8e5 的 manifest）**：

```
python3 scripts/split_data.py --corpus-test-ratio 0.12   # seed 42
→ train 4,659 / dev 412 / test 33 / test_corpus 585
```

> ⚠️ 版本註記：repo 現行 `data/splits/manifest.json` 已被後續的**教師審核受控重切**（commit 583cfd7，train 5,680／dev 605）取代。**v3 是在教師審核落實前的切分上訓練的**，兩者不可混淆；要重現 v3 請用上列指令與 seed。

訓練：3 epochs／1,749 步／約 60 分鐘；QLoRA 4-bit nf4、LoRA r=16 α=32、batch 2、grad_accum 4、lr 2e-4、max_len 192。

各 epoch dev loss：

| epoch | eval_loss |
|---|---|
| **1** | **1.460（最佳）** |
| 2 | 1.539 |
| 3 | 2.080 |

> dev 集因留存 585 句而改變，絕對值不與 v1/v2 可比。本輪內 **epoch 1 即最佳**，過擬合比 v1/v2（皆 epoch 2 最佳）更早，採 **checkpoint-583**。

## 3. 結果：核心 33 句（四版同一 metrics.py 重算）

| 系統 | BLEU-4 | ROUGE-L | Exact Match | 詞彙表內率(聯集) |
|---|---|---|---|---|
| 受控基礎模型（未微調） | 33.21 | 68.15 | 39.4% | 89.3% |
| v1（含 rule-derived, ep2） | 72.73 | 78.48 | 54.5% | 97.6% |
| v2（乾淨切分, ep2） | 42.17 | 79.75 | 57.6% | 96.2% |
| **v3（擴大留存, ep1）** | 66.69 | **82.37** | **63.6%** | 94.9% |

**v3 是目前最佳版本**：Exact Match 63.6%（21/33 完全正確，v1 18、v2 19）、ROUGE-L 82.37 皆為四版最高。

BLEU-4 再次印證 v2 的發現：v1 72.73 → v2 42.17 → v3 66.69 大幅跳動，而同期 EM 與 ROUGE-L 單調上升（54.5→57.6→63.6、78.48→79.75→82.37）。**在 n=33 上 BLEU-4 不可作為版本優劣依據**。

## 4. v3 錯誤分析（12 句未完全匹配）

| 類型 | 句數 |
|---|---|
| 完全正確 | 21 |
| 詞彙替換／混合 | 5 |
| Gloss 缺失 | 4 |
| 語序錯誤 | 2 |
| Gloss 多增 | 1 |

殘留誤差多為**輕微切分或詞形差異**，非語意錯誤：

- **切分差**：S02 早安→`早/安`、S06 再見→`再/見`（把單一詞切成兩個 Gloss）
- **尾詞缺失**：S07 `請/再/說`（漏「一次」）、S08 `請/慢`（漏「一點」）、S30 `我/要`（漏「這」）
- **語序**：S14 `水/我/要/喝`（ref `我/水/喝/要`）、S18 `我/不/舒服`（ref `我/舒服/不`，否定後置）
- **可接受變體**：S15 `我/上廁所/要`、S16 `我/肚子/餓`（多「肚子」但語意正確）

## 5. ⚠️ 585 句擴大評估未完成：VM GPU 驅動版本不一致

核心 33 句評估完成後，585 句評估在跑到第 16 句時被發現**極度緩慢（每句 500–830 秒）**。診斷結果：

```
執行中核心模組：NVRM 580.159.03   （開機時載入）
磁碟上已更新版本：modinfo 580.173.02
→ NVML 初始化失敗：Driver/library version mismatch
→ torch.cuda.is_available() = False
```

VM 的 NVIDIA 驅動被更新（推測為系統自動更新）但**未重新開機**，導致 CUDA 不可用，評估退回 CPU 執行。以此速度 585 句需約 **95 小時**，且佔用共用機 88% CPU／45% RAM，已將該程序停止（`--resume` 可續跑，16 筆結果保留於 VM）。

**此為機器層級問題，需管理者重開機或重新載入 nvidia 核心模組後才能修復**；本專案不在共用生產機上自行執行重開機。GPU 恢復後續跑指令：

```bash
python eval_model.py --adapter ../outputs/qlora_e4b_v3/checkpoint-583 \
  --test-file test_corpus.jsonl --tag finetuned_e4b_v3_ep1_corpus --resume
```

因此「以擴大 test 集取得穩定 BLEU」的目標**尚未達成**，工具與資料已就緒，僅待 GPU 修復。

## 6. 檔案

- `results/finetuned_e4b_v3_ep1_test.jsonl`、`results/summary_finetuned_e4b_v3_ep1.json`（核心 33 句，完整）
- VM adapter：`outputs/qlora_e4b_v3/checkpoint-583`（epoch 1，最佳；未入庫）
- VM 部分結果：`results/finetuned_e4b_v3_ep1_corpus_test.jsonl`（16/585，未入庫）

## 7. 下一步

1. **GPU 修復後**跑完 585 句評估 → 取得穩定 BLEU-4，補完本報告第 5 節。
2. **以教師審核後的新切分重訓**（commit 583cfd7，train 5,680，含 108 句教師修正）——v3 尚未納入這些修正。
3. 訓練 epoch 數：v3 顯示 epoch 1 即最佳，建議正式訓練用 **1–2 epochs 或 early stopping**，勿固定跑 3。
4. 評估規範：小 test 集以 **EM／ROUGE-L 為主指標**，BLEU-4 僅列參考並註明高變異。
5. 人工評估（計畫 6.2，5 分制）：S02/S06 的切分差、S14/S18 語序需母語者判定何者可接受。
