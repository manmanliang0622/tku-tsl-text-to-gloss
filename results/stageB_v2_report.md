# Stage B v2：乾淨切分重訓報告（依 HANDOFF.md）

日期：2026-07-23～24（學校 VM）
定位：**管線驗證**（HANDOFF §4：train/dev 來源仍 review_status=pending，不得宣稱「正確／正式／可散布」）。
對應：《微調訓練計畫》第 5 節 Stage B；HANDOFF.md 第 3–4 節。

## 1. 為什麼重訓（與 v1 的差別）

v1（commit d499b36）的切分含未經母語者裁定的 rule-derived 合成句、且 dev 可能有同源近似句洩漏。依 HANDOFF.md 指示，v2 改用**乾淨切分**：

| 項目 | v1 | v2（本次） |
|---|---|---|
| rule-derived 合成句 | 納入 | **排除 609 句**（`exclude_rule_derived=True`） |
| dev 洩漏 | 未控制 | **依 seg_uuid／詞條／模板整組留存，dev_group_leakage=0** |
| 切分 | train 5,788 | **train 5,129／dev 553／test 33** |
| train 真實資料占比 | ~85% | **~96%**（tslcorpus 4,392＋twtsl 497，synth 僅 240 非 rule-derived） |

test 集不變＝Stage A 相同 33 句真實已審核句，全程未進訓練。

## 2. 結果（三版同一份現行 metrics.py 重算，公平對比）

| 系統 | BLEU-4 | ROUGE-L | Exact Match | 詞彙表內率(聯集) |
|---|---|---|---|---|
| 受控基礎模型（未微調） | 33.21 | 68.15 | 39.4% | 89.3% |
| v1 微調（含 rule-derived） | 72.73 | 78.48 | 54.5% | 97.6% |
| **v2 微調（乾淨切分, ep2）** | 42.17 | **79.75** | **57.6%** | 96.2% |

**v2 在兩個穩定指標上都優於 v1**：Exact Match 57.6%>54.5%（19/33 完全正確）、ROUGE-L 79.75>78.48。詞彙表內率相當（96.2 vs 97.6）。

## 3. 關鍵發現：BLEU-4 在 n=33 小測試集上極不穩定

v2 的 EM 與 ROUGE-L 都比 v1 高，BLEU-4 卻從 72.73 掉到 42.17——這是**度量假象，非品質退步**。原因拆解（同一 metrics.py，v1 重算仍得 72.73，排除度量版本差異）：

- 整個測試集僅 79 token，**4-gram 總數只有 6 個**。
- 4-gram 精度：v1 p4=4/6、**v2 p4=0/6**；3-gram：v1 15/20、v2 9/21。
- 差異全來自 3 個長句的輕微語序／詞形差（v1 剛好全對、v2 各差一點）：

| 句 | ref | v1 | v2 |
|---|---|---|---|
| P04 | 明天/我/去/花蓮/要 | ✓ 完全命中 | 我/明天/去/花蓮/要（時間詞未置句首） |
| S14 | 我/水/喝/要 | ✓ | 我/水/要/喝（喝↔要） |
| S15 | 我/廁所/去/要 | ✓ | 我/上廁所/要（廁所→上廁所、漏「去」） |

4-gram 僅 6 個時，任一長句失手即讓 p4 崩塌，BLEU-4 隨之腰斬；而短句（占多數）對 3/4-gram 無貢獻，故 EM 上升與 BLEU-4 下降可同時發生。

**建議（寫入評估規範）**：n=33 的小測試集上以 **Exact Match 與 ROUGE-L 為主要指標**，BLEU-4 僅列參考並註明高變異。要穩定報 BLEU 應擴大真實 test 集，或改用對短序列較穩的 ChrF／sentence-BLEU 平均。P04 的時間詞句首是一個真實（但單句）的語序缺失，屬合成資料可強化的點。

## 4. 過擬合模式一致：epoch 2 最佳

各 epoch dev（eval）loss：

| epoch | v1 eval_loss | v2 eval_loss |
|---|---|---|
| 1 | 0.957 | 0.879 |
| **2** | 0.940 | **0.874（最佳）** |
| 3 | 1.258 | 1.180（過擬合回升） |

兩版皆 epoch 2 最佳、epoch 3 過擬合。v2 乾淨資料的 dev loss（0.874）低於 v1（0.940）（註：dev 集不同，非直接可比，但方向正面）。採 **checkpoint-1284（epoch 2）**。正式訓練建議固定 2 epochs 或以 dev loss early stopping。

## 5. 訓練設定（同 v1，另註）

- 基礎模型 `google/gemma-4-E4B-it`（Apache 2.0），QLoRA（4-bit nf4 + double quant，bf16），LoRA r=16 α=32。
- 3 epochs／1,926 步／約 66 分鐘；batch 2、grad_accum 4、max_len 192；GPU RTX 4060 Ti 16GB（共用生產機，訓練峰值留 ~2.5GB free）。
- 本次修正：`load_best_model_at_end=False`＋`save_total_limit=None`，訓練乾淨收尾（無 v1 的 offloaded model reload dispatch 報錯），並保留三個 epoch checkpoint 供事後挑選。

## 6. 檔案

- `results/finetuned_e4b_v2_ep2_test.jsonl`、`results/summary_finetuned_e4b_v2_ep2.json`
- VM adapter：`outputs/qlora_e4b_v2/checkpoint-1284`（epoch 2，最佳；未入庫）
- 切分 manifest：`data/splits/manifest.json`（train 5,129／dev 553／test 33，dev_group_leakage=0）

## 7. 下一步（承 HANDOFF §6）

- 人工評估（計畫 6.2，5 分制）：P04 等語序、S02「早/安靜」等單詞句多生成需手語老師判定。
- 擴大真實 test 集以穩定 BLEU。
- Stage C 多任務混訓、Stage D RAG。
- 補齊資料審核與授權後，方可宣稱正式成果。
