# 上傳 Hugging Face（private）備忘

> 目前**不上傳**。此檔僅備妥指令，要分享時再照做。
> 權重本機路徑：`models/tsl-gloss-e4b-v4-lora/checkpoint-630/`（已 gitignore，不進本 repo）。

## 先決條件
- HF 帳號；到 <https://huggingface.co/settings/tokens> 產生一個 **write** 權限 token。
- 已安裝 CLI：`pip install -U "huggingface_hub[cli]"`。

## 步驟

```bash
# 0) 登入（貼上 write token；token 屬機密，勿寫進 repo 或訊息）
huggingface-cli login

# 1) 準備乾淨上傳資料夾：只放 HF 需要的三個檔
#    （不要傳 optimizer.pt / scheduler.pt / rng_state.pth / trainer_state.json / training_args.bin）
mkdir -p models/tsl-gloss-e4b-v4-lora/upload
cp models/tsl-gloss-e4b-v4-lora/README.md \
   models/tsl-gloss-e4b-v4-lora/checkpoint-630/adapter_model.safetensors \
   models/tsl-gloss-e4b-v4-lora/checkpoint-630/adapter_config.json \
   models/tsl-gloss-e4b-v4-lora/upload/

# 2) 建 **private** repo 並整包上傳（--private：repo 不存在時建為私有）
huggingface-cli upload <你的HF帳號>/tsl-gloss-e4b-v4-lora \
  models/tsl-gloss-e4b-v4-lora/upload . \
  --repo-type model --private
```

## 驗證
- 開 `https://huggingface.co/<你的HF帳號>/tsl-gloss-e4b-v4-lora`（private，只有你／授權者看得到）。
- 應有：`README.md`（會渲染成 model card）、`adapter_model.safetensors`、`adapter_config.json`。

## 之後要轉公開
- 到該 repo 網頁 **Settings → Change visibility → Public**。
- 轉公開前確認 **Gemma Terms of Use**（Gemma 衍生權重散布條款），並確認 model card 出處完整
  （文化部臺灣手語語料庫；中正辭典：蔡素娟等 2026，中正大學）。

## 別人載入方式（權重就位後）
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = "google/gemma-4-E4B-it"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(model, "<你的HF帳號>/tsl-gloss-e4b-v4-lora")
```
（private repo 需先 `huggingface-cli login` 或帶 `token=`。）
