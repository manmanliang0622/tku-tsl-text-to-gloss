# Stage A：提示法基線報告（未微調 Gemma 4）

日期：2026-07-20
對應計畫：《臺灣手語翻譯語言模型_微調訓練計畫.md》第 5 節 Stage A、第 6 節評估。

## 1. 實驗設定

| 項目 | 設定 |
|---|---|
| 模型 | `gemma4:e4b`（Ollama 0.32.1，9.6 GB，未微調） |
| 硬體 | MacBook Pro（Intel i5-8257U，16 GB RAM，純 CPU 推理） |
| 解碼 | temperature=0、num_predict=128、**think=false**（見第 4 節） |
| 評測集 | `data/tsl_sentences.jsonl` 33 句（35 句排除含佔位符的模板句 S24、S26） |
| 指標 | BLEU-4（Papineni 2002）、ROUGE-L（Lin 2004）、Exact Match、詞彙表內率（`scripts/metrics.py`） |

三種提示策略（設計依據：SignAlignLM 的 0-shot／rule-based prompt；CCL24-Eval Task 10 的「任務描述＋專家示例＋翻譯任務」三段式）：

- **zero**：僅任務描述
- **rules**：＋計畫 3.3 節之 7 條已查證臺灣手語語法規則
- **fewshot**：＋9 句專家示例（示例池 10 句，leave-one-out 排除當前測試句防洩漏）

## 2. 結果

| 策略 | BLEU-4 | ROUGE-L | Exact Match | 詞彙表內率（自有85） | 詞彙表內率（聯集5,049） |
|---|---|---|---|---|---|
| zero | 39.50 | 55.10 | 27.3% | 68.7% | 85.5% |
| rules | 42.56 | 57.86 | 30.3% | 76.0% | 84.8% |
| **fewshot** | **44.95** | **65.59** | **36.4%** | **81.8%** | **94.3%** |

詞彙表內率有兩個分母：自有 85 Gloss（動作庫已拍攝範圍）與聯集 5,049 詞（自有 85 ＋ 中正大學《台灣手語線上辭典》3,500 詞條含同義索引名，2026-07-20 分工確認後的合法詞彙定義）。

**提示法遞增有效**：fewshot > rules > zero 在所有指標上成立，與 CCL24-Eval（ICL 可做出可用系統）及 SignAlignLM（語言學規則提示有幫助）的結論方向一致。

## 3. 錯誤分析（fewshot，33 句，分類法依 CCL24-Eval）

| 類型 | 句數 | 例子 |
|---|---|---|
| 完全正確 | 12 | 我住在台北。→ 我/台北/住 ✓ |
| 詞彙替換/混合 | 13 | S10 我知道 → 我/**知**（ref：我/知道）；S23 → 你/**名牌/問**/什麼 |
| 語序錯誤（詞集相同） | 3 | S25 你住哪裡 → 你/**哪裡/住**（ref：你/住/哪裡）；P04 → 明天/我/**花蓮/去**/要 |
| Gloss 多增 | 3 | S16 我肚子餓 → 我/**肚子**/餓（ref：我/餓） |
| Gloss 缺失 | 2 | S14 我要喝水 → 我/水/要（漏「喝」） |

主要瓶頸是**詞彙選擇**（13/33）：模型不知道動作庫／辭典的標準詞形（「知」vs「知道」、「名牌」vs「名字」）。這直接支持計畫的兩個後續手段：
1. **Stage B 微調**——讓模型學會標準詞形與語序（Exact Match 僅 36.4%，改進空間大）；
2. **Stage D RAG 詞彙約束**——推理時把合法詞彙檢索進 prompt（ITRI 做法）。

## 4. 過程發現：Gemma 4 思考模式陷阱

Gemma 4 是思考型模型：Ollama 預設 `think=true`，思考內容會先佔用 `num_predict` 額度。第一輪實驗（num_predict=64）因此大量輸出為空（`done_reason=length`，rules 策略 23/33 句空白、BLEU 掉到 10.4），且規則越多思考越長、越容易被截斷。修正：API 請求加 `"think": false`，輸出恢復正常且單句耗時從 20–40s 降到 12–14s。第一輪無效結果存於 `results/archive_thinking_truncated/` 備查。

**對後續的提醒**：Stage B 微調與正式部署時，須明確決定思考模式開關並保持訓練／推理一致；CPU 環境不可行完整思考鏈，GPU 環境可另行實驗 think=true 是否提升品質。

## 5. 檔案

- `results/baseline_gemma4_e4b_{zero,rules,fewshot}.jsonl`：逐句預測（含原始輸出與耗時）
- `results/summary_gemma4_e4b.json`：指標彙總
- `scripts/run_baseline.py`、`scripts/metrics.py`：可重跑（`--resume` 續跑、`--limit` 冒煙測試）
