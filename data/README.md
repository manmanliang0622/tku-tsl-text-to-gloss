# TSL 平行資料（第一階段產出）

產生方式：`python3 scripts/build_jsonl.py`（標記表更新後重跑即可，腳本會自動驗證句數、詞數與同句多影片間的 Gloss 一致性）。
對應計畫：《臺灣手語翻譯語言模型_微調訓練計畫.md》第 3.2 節「統一格式化」。

## 檔案

| 檔案 | 內容 | 筆數 |
|---|---|---|
| `tsl_sentences.jsonl` | 句級平行資料（中文 ↔ Gloss ↔ NMS） | 35（P01–P05 ＋ S01–S30） |
| `tsl_words.jsonl` | 台灣地名詞彙（W01–W38） | 38 |
| `tsl_gloss_vocab.json` | Gloss 詞彙總表＝下游動作庫檢索的合法詞彙清單 | 85 個 Gloss |

## 來源標記表

- `SLR_句子標記表_台灣地名應用句子.xlsx`（v1.4，含 NMS 欄位）— P01–P05
- `SLR_V2_標記表_已修改.xlsx` — S01–S30（已驗證與原始備份、Downloads 各版本 Gloss 完全一致）
- `台灣地名與區域詞彙_手語影片命名規則.docx` 表4 — W01–W38

以上皆位於 `/Users/leo/Documents/手語影片/outputs/`。

## 欄位說明（tsl_sentences.jsonl / tsl_words.jsonl）

| 欄位 | 說明 |
|---|---|
| `id` | 句子／詞彙代碼（P/S/W 前綴，跨批次全域唯一） |
| `type` | `sentence` 或 `word` |
| `chinese` | 中文原句（模型輸入端） |
| `gloss` | Gloss 序列（list，依 Gloss序 排序） |
| `gloss_text` | Gloss 以 `/` 串接的字串（模型輸出端） |
| `nms` | 非手部標記；無則 `null`（例：P03 是非問句的疑問表情） |
| `is_template` | `true` = 含全形底線 `＿` 佔位符的模板句（S24、S26），**訓練前需展開成具體句或排除** |
| `n_videos` | 該句對應影片數（僅供追溯，訓練不使用） |
| `batch` / `source_file` | 資料批次與來源標記表檔名（追溯用） |

## synth/ — 規則模板合成資料（3.3 節產出，待審核）

產生方式：`python3 scripts/synthesize.py`。**增量輸出**：重跑不會改動既有句的 SYN 編號與內容，只把新增句接續編號寫入 jsonl，並另出 `review_sheet_batch{N}.xlsx`（不覆蓋老師填寫中的舊審核表）。

| 檔案 | 內容 |
|---|---|
| `synth/tsl_synth.jsonl` | 合成平行資料 849 句（batch1：T1–T9 共 477 句；batch2：T10–T19 自有詞彙 328 句；batch3：T20–T25 外部詞彙 44 句），**全部 `review_status: pending`，未經審核不得進訓練集** |
| `synth/review_sheet.xlsx` | batch1（SYN0001–0477）審核表 |
| `synth/review_sheet_batch{N}.xlsx` | 各增量批次審核表（batch2＝SYN0478–0805、batch3＝SYN0806–0849 含外部詞彙欄） |

模板各自標註 `rule_basis`（依據計畫 3.3 節的 7 條已查證語法規則＋自有實例句），腳本會驗證不產生表外 Gloss、不與原始 35 句重複。`confidence` 兩級：

- `attested-pattern`：句型直接複製自有標記表實例，只換地名槽位
- `rule-derived`：由規則組合推導、自有資料無實例，**審核優先**（batch2 全屬此級；其中 T13/T14 涉及規則1 與規則7 的順序競合，審核最優先）

詞彙採兩層制：第一層為自有詞彙表（有自有動作影片）；第二層為中正大學手語辭典詞條（`twtsl/twtsl_words.jsonl`，T20–T25 的槽位來源）。用到第二層詞彙的句子會在 `external_glosses` 欄列出**尚無自有動作影片**的 Gloss——這些詞下游動作庫檢索不到，需列入拍攝排程或僅供語言模型學句型用。

## twtsl/ — 中正大學《台灣手語線上辭典》爬取資料

產生方式：`python3 scripts/scrape_twtsl.py`（可中斷續跑；`--consolidate-only` 只重整不重抓）。

| 檔案 | 內容 |
|---|---|
| `twtsl/twtsl_words.jsonl` | 辭典詞條（TW 編號）：手形、位置、筆畫、動作描述、示範影片網址 |
| `twtsl/twtsl_sentences.jsonl` | 常用詞例句（TWS 編號）：**辭典自帶 Gloss 標記＋中文翻譯的真實平行語料**，`review_status: pending` |
| `twtsl/raw_details.jsonl` | 原始 API 回應（續跑斷點用） |

出處：蔡素娟、戴浩一、劉世凱、陳怡君 2026《台灣手語線上辭典（中文版第五版）》，國立中正大學手語語言學台灣研究中心。僅抓後設資料與例句文字，不下載影片；**引用／發表時須依網站規定註明出處，資料再散布前先確認授權**。詞條名的 `_A`/`_B` 後綴為方言／自由變異，`gloss` 欄已去除後綴（原名保留在 `name` 欄）。

## 已知注意事項

1. **S24（我叫＿）、S26（我住在＿）是模板句**，`is_template: true`。佔位符 Gloss（`＿`、`＿地點`）已從詞彙總表排除。下一步資料合成時可用它們當句型模板展開（如 S26 × 38 個地名詞彙）。
2. 只有 P03 帶 NMS 標註；S 批次標記表沒有 NMS 欄位。依計畫第 8 節，S 批次中的疑問句（S21–S23、S25、S27）與否定句（S09、S11、S13、S18、S29）理論上應有 NMS，屬於**待補標註**項目，需手語老師／聾人顧問確認後補入標記表再重跑腳本。
3. ~~詞彙總表中 85 個 Gloss 是「自有影片資料集拍過的動作」上限~~ **2026-07-20 分工更新**：語言模型端只負責中文→Gloss 翻譯語序，手語影片由下游端自行爬取（`twtsl_words.jsonl` 每詞附辭典示範影片網址）。合法詞彙＝自有 85 Gloss＋twtsl 3,500 詞的聯集；「詞彙表內率」（計畫 6.1 節）改以此聯集為準。合成句的 `external_glosses` 欄仍保留，用途改為標示「下游需另爬影片」的詞。
