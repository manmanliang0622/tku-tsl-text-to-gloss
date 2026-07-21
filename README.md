# TKU TSL Text-to-Gloss

淡江大學專題：手語翻譯＋虛擬人生成 — **語言模型端（中文 → 臺灣手語 Gloss）**。

整體專題管線：

```
中文 → 臺灣手語翻譯 → TSL Gloss／動作腳本 → 動作生成或動作庫檢索
     → 動作串接與平滑 → 表情、頭部、身體同步 → Avatar 骨架驅動 → 影片／3D 即時播放
```

本 repo 負責前兩段：以 **Gemma 4** 微調的 Text-to-Gloss 翻譯模型。

## 目錄

| 路徑 | 內容 |
|---|---|
| [臺灣手語翻譯語言模型_微調訓練計畫.md](臺灣手語翻譯語言模型_微調訓練計畫.md) | 完整微調訓練計畫（任務定義、資料建置、模型選型、訓練流程、評估），所有方法均標註可查證出處 |
| [data/](data/) | 第一階段產出：句級平行資料 JSONL（35 句）、地名詞彙 JSONL（38 詞）、Gloss 詞彙總表（85 個 Gloss）＋中正手語辭典爬取資料（`data/twtsl/`，3,500 詞、544 例句），欄位說明見 [data/README.md](data/README.md) |
| [scripts/build_jsonl.py](scripts/build_jsonl.py) | 標記表（xlsx/docx）→ JSONL 轉換腳本，含一致性驗證，可重跑 |
| [scripts/synthesize.py](scripts/synthesize.py) | 規則模板資料合成（25 模板 × 詞彙槽位 → 849 句，增量輸出），同步產出各批次人工審核表 |
| [scripts/scrape_twtsl.py](scripts/scrape_twtsl.py) | 中正大學《台灣手語線上辭典》爬蟲（詞條＋帶 Gloss 例句，可續跑、限速） |
| [scripts/query_tslcorpus.py](scripts/query_tslcorpus.py) | 文化部《臺灣手語語料庫》主題查詢（合成前的語序查證證據） |
| [scripts/scrape_tslcorpus_full.py](scripts/scrape_tslcorpus_full.py) | 文化部《臺灣手語語料庫》全爬（407 段→5,272 句真實平行語料，訓練用） |

## 進度

- [x] 第零階段：參考文獻查證、微調訓練計畫（2026-07-17）
- [x] 第一階段 3.2：標記表統一格式化 → JSONL（2026-07-17）
- [x] 第一階段 3.3-A/B：規則模板合成 477 句（9 模板，附審核表）（2026-07-17）
- [x] 詞彙擴充：中正手語辭典爬取 3,500 詞＋544 帶 Gloss 例句；句型擴充 T10–T25 共 372 句（batch2/3）（2026-07-20）
- [x] 分工確認：語言模型端只負責翻譯語序，影片由下游自行爬取；詞彙表改兩層聯集（2026-07-20）
- [x] 主題句型 batch4：點餐/交通/問路/日常對話/看病 19 模板 118 句，先經文化部手語語料庫＋中正辭典例句語序查證（2026-07-20）
- [x] 文化部語料庫全爬：407 段落 → 5,272 句真實 Text→Gloss 平行語料（17 主題，訓練用，VM 執行）（2026-07-21）
- [ ] 第一階段 3.3-C：手語老師／聾人顧問人工審核（`data/synth/review_sheet*.xlsx`＋twtsl 例句）
- [ ] 第一階段 3.4：train/dev/test 切分（審核通過後）
- [ ] 第二階段：Gemma 4 E4B＋QLoRA 環境建置
- [x] 第三階段 Stage A：提示法基線（fewshot BLEU-4 44.95 / EM 36.4%，見 [results/stageA_report.md](results/stageA_report.md)）（2026-07-20）
- [ ] 第三階段 Stage B–D：QLoRA SFT → 多任務混訓 → RAG
- [ ] 第四階段：BLEU/ROUGE/詞彙表內率＋人工評估

## 主要參考資料

見計畫文件第 9 節資料來源清單（SignAlignLM ACL 2025、CCL24-Eval Task 10、SCOPE AAAI-25、工研院 AI手語虛擬氣象主播、Gemma 4 官方文件等）。
