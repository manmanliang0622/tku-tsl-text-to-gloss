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
| [data/](data/) | 第一階段產出：句級平行資料 JSONL（35 句）、地名詞彙 JSONL（38 詞）、Gloss 詞彙總表（85 個 Gloss）＋中正手語辭典本地快照（`data/twtsl/`，3,500 詞、544 例句；2026-07-22 官網已顯示 4,600 詞、560 例句，重建前需更新），欄位說明見 [data/README.md](data/README.md) |
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
- [x] 第一階段 3.4：train/dev/test 切分（train 5,788／dev 503／test 33，test=Stage A 同 33 句真實句永不進訓練）（2026-07-21）
- [x] 第二階段：Gemma 4 E4B＋QLoRA 環境建置（學校 VM，RTX 4060 Ti；含 PLE CPU-offload 記憶體解法）（2026-07-21）
- [x] 第三階段 Stage A：提示法基線（fewshot BLEU-4 44.95 / EM 36.4%，見 [results/stageA_report.md](results/stageA_report.md)）（2026-07-20）
- [x] 第三階段 Stage B v1：QLoRA 微調首輪（EM 54.5%；見 [results/stageB_report.md](results/stageB_report.md)）（2026-07-22）
- [x] 第三階段 Stage B v2：乾淨切分重訓（排除 rule-derived＋dev 無洩漏，**EM 57.6% / ROUGE-L 79.75**，epoch 2 最佳；見 [results/stageB_v2_report.md](results/stageB_v2_report.md)）（2026-07-24）
- [x] 第三階段 Stage B v3：擴大真實 test 留存＋核心評估（**EM 63.6% / ROUGE-L 82.37**；585 句長評估因 GPU 驅動問題停於 16 筆）（2026-07-25）
- [x] 第三階段 Stage B v4 資料：教師審核後重切（train 5,038／dev 636／核心 test 33／擴大 test 584），加入 group bootstrap BLEU CI 與完整 leakage 檢查
- [ ] 第三階段 Stage B v4 訓練／評估：等待 VM 管理者修復 NVIDIA driver/library mismatch
- [ ] 第三階段 Stage C–D：多任務混訓 → RAG
- [ ] 第四階段：人工評估（5 分制）＋以 2 epochs 重訓正式版

## 2026-07-22 語言資料稽核

- 已以可回查的中正大學臺灣手語研究文獻、博士／碩士論文、教育部／國教院資源重新校正規則描述。
- 老師具體審核量：T13/T14 競合 5 句；`rule-derived` 609 句；詞形 27 句；現有真實句 NMS 11 句；文化部語料固定風險抽查 170 句。各維度有重疊，不能相加。
- 詳見 [資料來源.md](資料來源.md)、[臺灣手語資料審核報告_2026-07-22.md](臺灣手語資料審核報告_2026-07-22.md) 與 `outputs/019f87d9-976d-7961-84aa-a05c910dcd5c/臺灣手語老師具體審核清單_2026-07-22.xlsx`。
- [~] 第一階段 3.3-C：手語老師審核（進行中）——已逐句審 115 flagged 句、修正 108（見「2026-07-24 手語老師審核落實」）；其餘 852 句合成句與 twtsl/tslcorpus 例句仍 `pending`

## 2026-07-23 全資料逐筆審核

- 已審核 8 類共 18,440 筆，逐列加入審核狀態、訓練狀態、風險旗標、理由與人工覆核欄；原始檔未被覆蓋。
- 967 句合成句全部暫停訓練，其中 115 句已有明確語意／詞彙／順序風險。
- 文化部語料 5,272 句中，1,846 句需影片／上下文對齊，147 筆完全重複列需排除或群組化。
- 現有 split 只能作管線驗證，不得作正式訓練；原因是 `exclude_rule_derived=false` 且 train/dev 仍含 pending 資料。
- 詳見 [全資料審核報告](臺灣手語全資料審核報告_2026-07-23.md) 與 `outputs/019f87d9-976d-7961-84aa-a05c910dcd5c/臺灣手語全資料逐筆審核_2026-07-23.xlsx`。

## 2026-07-24 手語老師審核落實

- 手語老師依中正辭典第五版＋文化部語料庫，逐句審核 07-23 標記的 115 句「需修正或追加實證」：**修正 108 句、7 句留待母語者影片裁定**（T13/T14 競合 5、上學/唸書 1、感冒狀態 1）。
- 修正已套進訓練資料（`scripts/apply_teacher_review.py`，原 Gloss 保留於 `pre_review_gloss_text`）：`review_status` = teacher-reviewed 108／reviewed-pending-video 7／pending 852。其中 **29 句 attested-pattern 修正（如 上班→工作）實際改善訓練集**。
- 切分已改安全預設：`exclude_rule_derived=true` ＋ dev 去洩漏（`dev_group_leakage=0`），train 5,130／dev 553／test 33。
- **界線（勿誤解「審核完＝可正式訓練」）**：967 句中僅 108 句經老師逐句審核，其餘 852 仍 `pending`、791 句需影片；合成資料整體 `training_status` 仍暫停。現況仍為**管線驗證**，正式成果尚需更全面審核＋影片裁定＋資料授權。
- 分支：main/data 已含本次落實（commit a3be340）；model 分支由訓練端維護。

## 2026-07-25 Stage B v4

- 以 `--use-teacher-reviewed --corpus-test-ratio 0.12 --seed 42` 結合老師審核後的 synth 與擴大真實 test。
- 585 句候選全部對到老師工作簿判定；保留 584 句、排除重複列 `TC01419`，並以 machine-readable sidecar 與 SHA-256 固定評測集合。
- 擴大 test 含 37 個對話群組、1,070 個 4-gram；train/dev/test 的群組、中文、Gloss 與 pair 洩漏均為 0。
- 自動評估新增以對話群組為抽樣單位的 BLEU-4 95% bootstrap CI。詳見 [results/stageB_v4_report.md](results/stageB_v4_report.md)。
- VM 仍載入 NVIDIA 核心模組 580.159.03、磁碟／NVML 為 580.173.02，CUDA 不可用；不自行重開共用 VM，待管理者修復後再跑 2 epochs v4。

## 主要參考資料

見計畫文件第 9 節資料來源清單（SignAlignLM ACL 2025、CCL24-Eval Task 10、SCOPE AAAI-25、工研院 AI手語虛擬氣象主播、Gemma 4 官方文件等）。
