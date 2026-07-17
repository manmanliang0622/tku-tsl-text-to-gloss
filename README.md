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
| [data/](data/) | 第一階段產出：句級平行資料 JSONL（35 句）、地名詞彙 JSONL（38 詞）、Gloss 詞彙總表（85 個 Gloss），欄位說明見 [data/README.md](data/README.md) |
| [scripts/build_jsonl.py](scripts/build_jsonl.py) | 標記表（xlsx/docx）→ JSONL 轉換腳本，含一致性驗證，可重跑 |

## 進度

- [x] 第零階段：參考文獻查證、微調訓練計畫（2026-07-17）
- [x] 第一階段 3.2：標記表統一格式化 → JSONL（2026-07-17）
- [ ] 第一階段 3.3：規則模板＋LLM 改寫資料合成、人工審核
- [ ] 第一階段 3.4：train/dev/test 切分
- [ ] 第二階段：Gemma 4 E4B＋QLoRA 環境建置
- [ ] 第三階段：提示法基線 → QLoRA SFT → 多任務混訓 → RAG
- [ ] 第四階段：BLEU/ROUGE/詞彙表內率＋人工評估

## 主要參考資料

見計畫文件第 9 節資料來源清單（SignAlignLM ACL 2025、CCL24-Eval Task 10、SCOPE AAAI-25、工研院 AI手語虛擬氣象主播、Gemma 4 官方文件等）。
