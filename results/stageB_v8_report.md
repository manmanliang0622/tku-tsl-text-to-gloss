# Stage B v8：結構化 JSON 目標（含 NMS）訓練報告

日期：2026-08-07
定位：目前**最佳模型**，已上線供前端測試。

## 1. 結論

以「結構化 JSON」取代「純 Gloss」作為訓練目標後，**在 33 句核心 test 上每一項指標都優於 v6**，同時獲得 v6 沒有的能力（疑問句判斷、否定判斷、非手部標記輸出）。

| 模型 | BLEU-4 | ROUGE-L | Exact Match | 可播放率 |
|---|---:|---:|---:|---:|
| 0804try（tku_json, 5,038 句） | 56.71 | 63.88 | 39.39% | 91.76% |
| v6（純 Gloss, 6,102 句） | 72.59 | 80.52 | 63.64% | 94.29% |
| **v8（JSON 目標, 6,102 句）** | **82.47** | **88.20** | **75.76%** | **97.18%** |

v8 額外能力（v6 完全沒有）：

| 指標 | v8 |
|---|---:|
| 有效 JSON 率 | 100% |
| 疑問句類型判斷正確率 | 100% |
| 否定判斷正確率 | 96.97% |
| 實際輸出 NMS 的句子比例 | 33.33% |

三者用**同一份 33 句 test、同一套 metrics 與詞彙表**評估；該 33 句對三個模型都是留存的（0804try 用的正是本專案 v4/v5 切分）。

## 2. 為什麼 JSON 目標反而讓 Gloss 更準

原本預期是「多輸出語法欄位可能犧牲 Gloss 準確度」，實測相反（EM 63.64% → 75.76%）。合理解釋：`topic`／`verb`／`time`／`question_type` 等欄位構成**輔助監督訊號**，等於要求模型先把句子的語法角色想清楚再產生語序，屬多任務學習的正向遷移。此為觀察到的結果，非事前假設。

⚠️ **不可用 dev loss 比較兩者**：v8 的 dev loss 0.1487 遠低於 v6 的 0.7061，但 JSON 目標含大量固定欄位（`subject:none`、`object:none`、`verb_type:unknown`、`agreement:none`）屬零難度填空，會系統性壓低 loss。兩者的 loss 不在同一尺度上。

## 3. 訓練設定

| 項目 | 設定 |
|---|---|
| 基礎模型 | `google/gemma-4-E4B-it`（4-bit nf4） |
| LoRA | r=16, α=32, dropout=0.05（與 v6 相同） |
| 目標格式 | 結構化 JSON（`scripts/build_json_targets.py`） |
| 資料 | train 6,102／dev 641／test 33（與 v6 完全相同的切分） |
| 超參數 | 2 epochs、batch 2、grad_accum 4、lr 2e-4、max_len 192、seed 42 |
| dev loss | epoch 1 **0.1487（最佳）**、epoch 2 0.1510 → 採 `checkpoint-763` |
| 訓練時間 | 1,526 步／1 小時 08 分 |

**變因單一**：與 v6 相比只有目標格式不同，其餘（資料、切分、超參數、seed）完全一致，故差異可歸因於目標格式。

## 4. Schema 與來源

欄位定義沿用同機 `0804try` 專案 `src/signavatar/tku_json_dataset.py`，使兩邊模型可直接對照、前端可共用解析程式。本專案有兩處改良：

1. **`nonmanual` 用真實標註**：0804try 恆填 `"none"`；本專案 train 有 530 句帶人工 NMS 文字（synth 模板＋P03），直接採用，其餘才回退啟發式。這是 v8 能輸出真實表情資訊的原因。
2. **`verb` 不取句末 token**：0804try 直接取最後一個 gloss，但本專案實測文化部語料庫「要」有 66% 位於句末、否定詞亦常在句末，直接取末位會把情態／否定詞誤標成動詞。故先剔除句末情態與否定詞再取。

## 5. 實測輸出（線上服務）

```
我要喝水      → 我/水/喝/要      question=none  negation=false nonmanual=none
你住在宜蘭嗎  → 你/宜蘭/住       question=yesno negation=false
                nonmanual=「疑問表情（眉毛上揚、身體微前傾）貫穿全句，不比『嗎』」
我今天不去學校 → 今天/我/學校/去/不  question=none negation=true  time=今天
                nonmanual=「否定可伴隨搖頭、眼睛或嘴部表情」
```

是非問句不打「嗎」、否定詞置句尾、時間詞置句首等規則皆正確，且**非手部標記已可直接餵給下游虛擬人做表情同步**（計畫第 1 節的需求至此才真正被滿足）。

推論速度 4.5–5.7 秒/句（JSON 目標較長，約為 v6 的 4 倍；仍在互動可接受範圍）。

## 6. 檔案

- `results/v8_json_ep1_test.jsonl`、`results/summary_v8_json_ep1.json`
- `scripts/build_json_targets.py`、`scripts/eval_json_model.py`
- `scripts/train_qlora.py --target json`、`scripts/serve_model.py --target json`
- VM adapter：`outputs/qlora_e4b_v8_json/checkpoint-763`（未入庫）

## 7. 界線與後續

- 33 句 test 樣本小（僅 5 個 reference 4-gram），BLEU 變異大；EM／ROUGE-L 較可靠。
- 語料庫已全數進訓練，**目前沒有大規模真實泛化測試集**；要嚴謹證明泛化需另留測試資料或做人工評估。
- `nonmanual` 的訓練訊號有 530 句來自人工標註、其餘為啟發式回退，尚未經手語老師逐句驗證。
- 下一步建議：計畫 6.2 的 5 分制人工評估；以及 SCOPE（AAAI-25）的上下文翻譯，以解決先前診斷出的「受詞前置取決於前文話題」問題。
