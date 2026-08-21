# 上傳 Hugging Face（private）— v14 / tsl-script-v1

備料已完成，`upload/` 裡就是要傳的三個檔，**缺的只有登入這一步**。

    upload/README.md                 6.6 KB   會渲染成 model card
    upload/adapter_config.json       1.7 KB   base_model 已改為 google/gemma-4-E4B-it
    upload/adapter_model.safetensors 140 MB   checkpoint-558（epoch 1，dev loss 0.1957）

已刻意**不含** optimizer.pt／scheduler.pt／rng_state.pth／trainer_state.json／
training_args.bin——那些是訓練狀態，對使用者無用且會讓 repo 肥四倍。
也不含任何原始語料。

## 為什麼要你自己登入

`huggingface-cli login` 要貼 **write token**，那是機密憑證。
**不要把 token 貼進對話、寫進 repo、或交給任何自動化流程。**

## 步驟

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login          # 貼 write token（https://huggingface.co/settings/tokens）
```

登入後（在專案根目錄）：

```bash
huggingface-cli upload <你的HF帳號>/tsl-gloss-e4b-v14-script \
  models/tsl-gloss-e4b-v14-script/upload . \
  --repo-type model --private
```

`--private`：repo 不存在時建為私有。**先私有是專案原本的規劃**——
轉公開容易，轉回來難（公開過的東西可能已被快取）。

## 驗證

開 `https://huggingface.co/<你的HF帳號>/tsl-gloss-e4b-v14-script`，
應有 README.md（渲染成 model card）、adapter_config.json、adapter_model.safetensors。

## ⚠️ 轉公開前必須先處理的兩件事

1. **Google Gemma Terms of Use**。adapter 是 Gemma 衍生物，散布須遵守其條款
   （含使用限制）。MODEL_CARD.md 明寫「公開權重前請先確認 Gemma 條款；
   本卡不代為認定」——**這是專案自己標記的未解決閘門，不是形式**。
2. **model card 的出處完整性**：文化部臺灣手語語料庫、中正辭典
   （蔡素娟、戴浩一、劉世凱、陳怡君 2026，中正大學）。已寫在 README，
   轉公開前再確認一次沒被改掉。

轉公開：repo 網頁 → Settings → Change visibility → Public。

## 別人怎麼用（重要）

**這個 adapter 單獨下載沒有用。** 它輸出 `sign_id`，需要搭配同一套候選檢索器與
17,078 筆 sign_id 總表才有意義，而那兩者不在 HF repo 裡。README 已把這點寫在
最前面，不要拿掉那段——否則下載的人會以為模型壞了。

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("google/gemma-4-E4B-it", device_map={"": 0})
model = PeftModel.from_pretrained(base, "<你的HF帳號>/tsl-gloss-e4b-v14-script")
```

⚠️ `device_map` 含任何 `"cpu"` 項目會讓 accelerate 掛 offload hook，
每 token 慢到約 35 秒（全放 GPU 是 0.06 秒/token）。
