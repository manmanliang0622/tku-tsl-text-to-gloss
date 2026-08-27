# 線上服務切換至 v17＋約束解碼（2026-08-27）

使用者裁示「接到成果好的版本」。v17cd＝v17 adapter（k=40＋語義通道）＋
解碼時將 sign_id 鎖在候選清單內（constrained_decode.py，服務端 serve_model
已內嵌同一套邏輯）。

## 判定依據（完整參考）

| | v14 | v17cd |
|---|---|---|
| corpus BLEU／RL／EM | 16.20／52.33／2.41% | 16.36／52.92／0.60% |
| textbook BLEU／RL／EM | 22.99／60.88／14.89% | **24.47／61.16／15.60%** |
| test33 BLEU／EM | **77.8／69.7%** | 72.7／66.7% |
| 約束違反（corpus 列/ID） | 4.22%／0.70% | **0%／0%** |
| 可播放率（corpus／textbook） | 99.5／99.15% | **100／100%** |
| needs_review 校準 F1（corpus） | 0.825 | 0.827 |

約束解碼相對 v17 的品質代價只有 −0.2 BLEU（雜訊級），違反率歸零、
可播放率 100%——新格式的核心價值指標全滿。corpus EM −1.8pp（3 句）與
test33 −1 句是已知代價（v17 報告的「字面精確解被擠出首選」信號）。
教材集（最大、最接近展示情境）全面領先定勝負。

## 部署內容（0821）

- `model_service/checkpoint` ← v17script_k40sem/checkpoint-558（雜湊已驗證）；
  v14 備份在 `checkpoint.bak-v14-current`，回滾＝搬回目錄＋改回兩個常數＋重啟
- `bundle_server.py` EXPECTED_MODEL ← qlora_e4b_v17script_k40sem
- `serve_model.py` NEEDS_REVIEW_THRESHOLD 0.02 ← **0.095349**
  （v17 的 dev 依既定規則選定；門檻跟模型走，不可混用）
- 公開網頁端到端驗證：`/api/translate` 回 v17、needs_review 用新門檻、
  dropped_ids 空（約束解碼生效）

## v17cd 評估產物

`results/v17cd_*.jsonl`／`*_scriptmetrics.json`（門檻 0.095349）。
與 v17（無約束解碼）唯一差異在解碼；訓練產物同一個。
