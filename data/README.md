# TSL 平行資料（第一階段產出）

> 2026-07-23 審核狀態：專案全資料的逐筆審核版位於 `outputs/019f87d9-976d-7961-84aa-a05c910dcd5c/reviewed_data_2026-07-23/`。原始 JSONL 保留不動。現有合成句全部暫停訓練，現有 train/dev/test 只可作管線驗證；詳見根目錄 `臺灣手語全資料審核報告_2026-07-23.md`。

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
| `synth/tsl_synth.jsonl` | 合成平行資料 967 句（batch1：T1–T9 共 477 句；batch2：T10–T19 自有詞彙 328 句；batch3：T20–T25 外部詞彙 44 句；batch4：T26–T44 主題句型 118 句），**全部 `review_status: pending`，未經審核不得進訓練集** |
| `synth/review_sheet.xlsx` | batch1（SYN0001–0477）審核表 |
| `synth/review_sheet_batch{N}.xlsx` | 各增量批次審核表（batch2＝SYN0478–0805、batch3＝SYN0806–0849、batch4＝SYN0850–0967，含外部詞彙欄） |

batch4 為主題句型（點餐、交通、問路、日常對話、看病），每個模板做法：先查文化部《臺灣手語語料庫》真實聾人語料（`scripts/query_tslcorpus.py` → `refs/tslcorpus_evidence.jsonl`，1,082 句證據）與中正辭典例句核對語序，再合成；`rule_basis` 附語料庫句代碼（如 G3C12/a94）可回查。

模板各自標註 `rule_basis`（依據計畫 3.3 節的 7 條已查證語法規則＋自有實例句），腳本會驗證不產生表外 Gloss、不與原始 35 句重複。`confidence` 三級：

- `attested-pattern`：句型直接複製自有標記表實例，只換地名槽位
- `corpus-attested`：句型直接複製公開語料庫／辭典的真實例句（非自有資料），只換槽位
- `rule-derived`：由規則組合推導、無逐字實例，**審核優先**（其中 T13/T14 涉及規則1 與規則7 的順序競合，審核最優先）

注意：Gloss 使用本地中正辭典快照的**辭典主詞條**（公車→公共汽車、痛→疼、騎腳踏車→自行車、讀書→唸書），下游才可依相同 key 檢索；這不代表唯一「正式名」或唯一打法。對應中文同義索引在 `aliases` 欄，A/B 為自由變異、N/S 為南北方言，仍須老師確認實際選用詞形。

## vocab/ — 統一 Gloss 主詞彙表

產生方式：`python3 scripts/build_vocab.py`（合併自有＋twtsl＋語料庫 Gloss）。

| 檔案 | 內容 |
|---|---|
| `vocab/gloss_master.jsonl` | 7,002 個 Gloss 字形，每個標 `sources`（own／twtsl／twtsl-alias／corpus）、`corpus_freq`、`has_dict_entry`、`norm_has_dict_entry`（異體字正規化後）、`twtsl_ids` |
| `vocab/coverage.json` | 覆蓋率摘要（供評估腳本引用） |

**關鍵數字**：語料庫真實用到 4,680 個不同 Gloss，其中僅 **30.7%（type）** 有辭典詞條，但**依出現次數計為 70.7%（token）**——高頻詞多半查得到，OOV 集中在長尾詞組（沒辦法、手語翻譯）、體標記（完、先、之後）與異體字（臺↔台）。「詞彙表內率」（計畫 6.1 節）宜以 token 覆蓋 70.7% 為現況基線。正規化僅供比對，不改寫原資料。

## tslcorpus/ — 文化部《臺灣手語語料庫》全爬平行語料 ★真實訓練資料

產生方式：`python3 scripts/scrape_tslcorpus_full.py`（VM 執行；可續跑，`--consolidate-only` 只重整）。

| 檔案 | 內容 | 筆數 |
|---|---|---|
| `tslcorpus/parallel.jsonl` | 全語料庫 407 段落逐句抓取的**真實聾人平行語料**（中文↔Gloss token 陣列↔主題↔影片網址），`review_status: pending` | 5,272 句 |

這是目前**最大宗的真實 Text→Gloss 平行資料**（17 主題：休閒娛樂／公共服務／身體醫療／交通旅遊／教育學習／餐飲烹飪／自然環境／日常起居／社交人際／文化／安全／科技／社會／商店購物／情緒態度／職業／主題訪談），Gloss 由文化部標注、中位長度 4 詞、含 3,818 句短句（2–6 詞）。相較合成句與辭典例句，這批最適合直接進微調訓練集（審核後）。

- 每句附 `corpus_id`（篇章/句 ID，如 G2D1P1/a81）、`seg_uuid`、`speaker`、`film_url` 可回查原始影片。
- Gloss token 已去除來源夾帶的中文標點；保留 `++`（重複貌）等語言學標記。
- 說明：API 的主題過濾參數實測無效（每個主題都回全部 407 段），故爬蟲只列舉一次、主題改由每段回應的 `theme_data` 判定（見腳本註解）。
- 續跑檔 `refs/tslcorpus_raw.jsonl`（一段一行原始回應，4.6MB）已列入 `.gitignore`，不入庫。

出處：文化部《臺灣手語語料庫（測試版）》。僅抓文字標記（Gloss／中文）供學術研究，不下載影片；**引用／散布須依網站著作權聲明辦理**。

## refs/ — 語序查證證據

`refs/tslcorpus_evidence.jsonl`：文化部《臺灣手語語料庫》按主題關鍵詞抓取的例句（Gloss＋中文對照＋篇章代碼），僅供句型查證，© 文化部，引用／散布依其著作權聲明。（全語料庫已由 `tslcorpus/parallel.jsonl` 完整涵蓋，此檔保留供 batch4 模板的規則依據回溯。）

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
2. 只有 P03 帶 NMS 文字；S 批次標記表沒有 NMS 欄位。具體審核量為 **11 句**：S21–S23、S25、S27 與 S09、S11、S13、S18、S29 共 10 句待補；P03 的既有 NMS 形式與作用範圍仍須驗證。補標時應記錄 NMS 形式、開始 Gloss、結束 Gloss 及可否省略，再重跑腳本。
3. ~~詞彙總表中 85 個 Gloss 是「自有影片資料集拍過的動作」上限~~ **2026-07-20 分工更新**：語言模型端只負責中文→Gloss 翻譯語序，手語影片由下游端自行爬取（`twtsl_words.jsonl` 每詞附辭典示範影片網址）。合法詞彙＝自有 85 Gloss＋twtsl 3,500 詞的聯集；「詞彙表內率」（計畫 6.1 節）改以此聯集為準。合成句的 `external_glosses` 欄仍保留，用途改為標示「下游需另爬影片」的詞。
4. **2026-07-22 版本差異**：本地 `twtsl` 快照為 3,500 詞、544 例句；中正辭典官網已顯示第五版 4,600 詞、560 例句。重建 `gloss_master.jsonl` 或重新計算覆蓋率前，需先更新本地快照。
5. 文化部語料的中文字元數與 Gloss 詞數差異只能用來挑人工抽查樣本，不能直接當作錯誤標籤。可重現稽核門檻與固定 170 句抽查表見專案根目錄的 2026-07-22 審核報告與審核工作簿。
