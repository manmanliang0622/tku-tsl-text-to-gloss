---
license: gemma
base_model: google/gemma-4-E4B-it
library_name: peft
pipeline_tag: translation
language:
- zh
tags:
- peft
- lora
- qlora
- gemma
- sign-language
- taiwan-sign-language
- tsl
- gloss
- translation
metrics:
- bleu
- rouge
- exact_match
model-index:
- name: tsl-gloss-e4b-v4-lora
  results:
  - task:
      type: translation
      name: Chinese → Taiwan Sign Language Gloss (Text-to-Gloss)
    metrics:
    - type: bleu
      value: 18.61
      name: BLEU-4 (extended 584-sentence test, 95% CI 15.48–22.49)
    - type: rouge
      value: 55.40
      name: ROUGE-L (extended 584)
    - type: exact_match
      value: 9.25
      name: Exact Match % (extended 584)
    - type: bleu
      value: 80.00
      name: BLEU-4 (core 33-sentence test, 95% CI 65.74–88.77)
---

# tsl-gloss-e4b-v4-lora

QLoRA adapter that translates **Chinese sentences → Taiwan Sign Language (TSL) gloss order**.
Fine-tuned from `google/gemma-4-E4B-it`. Tamkang University (淡江大學) student project.

> 中文 → 臺灣手語 Gloss 詞序翻譯的 QLoRA adapter。此 README 為 Hugging Face 格式 model card；
> 完整中文說明另見專案 repo 的 [`MODEL_CARD.md`](https://github.com/manmanliang0622/tku-tsl-text-to-gloss/blob/main/MODEL_CARD.md)。

## Model Details

- **Base model:** `google/gemma-4-E4B-it` (Gemma 4 E4B, instruction-tuned)
- **Adapter:** LoRA r=16, alpha=32, dropout=0.05; 4-bit **nf4** + double quant, **bf16** compute
- **Target modules:** `language_model` 的 `q/k/v/o/gate/up/down_proj`
- **Task:** Text-to-Gloss (輸入中文，輸出以 `/` 分隔的 gloss token 序列)
- **Language:** Chinese input → Taiwan Sign Language gloss
- **Developed by:** 淡江大學專題（TKU TSL Text-to-Gloss）
- **License:** Gemma (見下方 License & Attribution)

## Intended Use & Limitations（務必閱讀）

- **是什麼：** Text→Gloss 的**詞彙／語序層**候選模型。
- **不是什麼：** **不輸出、也不保證 NMS（非手部標記：表情／搖頭／揚眉）、手形、地區變體。**
  這些屬「影片軌」，需臺灣手語母語者／聾人顧問看影片裁定，不在本模型範圍。
- **成熟度：** 內部候選 / 管線驗證。核心 33 句指標高，但涵蓋較廣真實對話的 584 句 BLEU 僅 18.61；**尚非最終成果**，勿當通用可用模型。
- 自動指標高 ≠ 手語文法正確；正式品質須經手語老師 5 分制人工評估。

## Training Data & Attribution（散布須標明出處）

| 來源 | 出處／授權 |
|---|---|
| 文化部臺灣手語語料庫（測試版） | © 文化部臺灣手語語料庫。訓練＋散布授權已確認合法（2026-08-04），須標明出處。<https://tslcorpus.moc.gov.tw/> |
| 中正大學台灣手語線上辭典（第五版） | 蔡素娟、戴浩一、劉世凱、陳怡君。2026。《台灣手語線上辭典（中文版第五版）》。嘉義：國立中正大學手語語言學台灣研究中心。訓練＋散布授權已確認合法（2026-08-04），須標明出處。<https://twtsl.ccu.edu.tw/> |
| 自有標記表 | 專案自製（35 句／38 詞） |
| 規則模板合成句 | 合成；經手語老師 2026-07-24 gloss 層審核（108 句修正，7 句待影片者排除） |

**授權 ≠ 品質背書**：NMS／手形／地區變體正確性仍待母語者影片裁定。

## Training Procedure

Split（seed 42 固定，可重現）：
```bash
python3 scripts/split_data.py --use-teacher-reviewed --corpus-test-ratio 0.12 --seed 42
# train 5,038 / dev 636 / core test 33 / extended test 584；洩漏檢查全為 0
```

Train：
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 scripts/train_qlora.py \
  --model google/gemma-4-E4B-it \
  --output outputs/qlora_e4b_v4_teacher_holdout \
  --epochs 2 --batch 2 --grad-accum 4 --max-len 192 --lr 2e-4 --seed 42
```
- 1,260 steps，約 43 分 31 秒，final train loss 3.459。
- **選模僅依最低 dev loss** → `checkpoint-630`（epoch 1, dev loss 0.9821940660）＜ `checkpoint-1260`（1.0088934898）。未用 test 選模。

## Evaluation

```bash
python3 scripts/eval_model.py \
  --adapter outputs/qlora_e4b_v4_teacher_holdout/checkpoint-630 \
  --test-file test_corpus.jsonl --tag finetuned_e4b_v4_teacher_ep1_corpus \
  --batch-size 8 --resume --bootstrap-samples 1000 --bootstrap-seed 42
```

| Metric | Core 33 | Extended 584 |
|---|---:|---:|
| BLEU-4 | 80.00 | **18.61** |
| BLEU-4 95% CI | 65.74–88.77 | **15.48–22.49** |
| ROUGE-L | 73.92 | 55.40 |
| Exact Match | 48.48% | 9.25% (54/584) |
| Union in-vocab rate | 95.12% | 69.74% |
| bootstrap groups | 33 | 37 |

兩套 test 句型／長度／難度不同，分數不可當同分布比較；泛化描述以 584 句為主。

## Getting the Weights

此 model card 目錄**未附 adapter 權重**。取得方式：
1. **重跑**：依上方三步（seed 42 固定）重現同一 adapter。
2. **上傳**：若要在此 HF repo 提供權重，將
   `outputs/qlora_e4b_v4_teacher_holdout/checkpoint-630/`（`adapter_model.safetensors` ＋ `adapter_config.json`）
   放到本目錄後推送（HF Hub 內建 LFS）。

## How to Use（權重就位後）

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "google/gemma-4-E4B-it"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(model, "<this-repo-or-local-path>")
# 輸入中文，輸出以 / 分隔的 TSL gloss 序列
```

## License & Attribution

- **Model weights** 為 Google Gemma 衍生物 → 適用 **Gemma Terms of Use**（散布須隨附條款、含使用限制）。公開權重前請確認 Gemma 條款。
- **Training data** 使用須標明出處（見上表）。

## Citation

- 文化部臺灣手語語料庫（測試版）。文化部。<https://tslcorpus.moc.gov.tw/>
- 蔡素娟、戴浩一、劉世凱、陳怡君。2026。《台灣手語線上辭典（中文版第五版）》。嘉義：國立中正大學手語語言學台灣研究中心。<https://twtsl.ccu.edu.tw/>
- Base model: Google Gemma 4 E4B（Gemma Terms of Use）。
