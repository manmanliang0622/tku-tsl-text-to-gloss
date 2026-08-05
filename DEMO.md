# 前端測試語言模型（中文 → TSL Gloss）

把微調後的 Gemma 4 接到 `手語前端優化/index.html#generation`，在網頁上直接測試翻譯成效。

模型跑在學校 VM（有 GPU），前端在自己的 Mac；兩者以 **SSH 通道**連接，
不需要開放 VM 對外埠口。

## 步驟

### 1. VM：啟動推論服務

```bash
cd ~/tku-tsl-text-to-gloss
. .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
setsid nohup python3 scripts/serve_model.py \
  --adapter outputs/qlora_e4b_v6_all/checkpoint-763 \
  --port 8018 > serve.log 2>&1 < /dev/null &
```

等 `serve.log` 出現 `[serve] 監聽 127.0.0.1:8018` 表示模型載入完成（約 1–2 分鐘）。

### 2. Mac：開 SSH 通道（此視窗保持開著）

```bash
ssh -p 2288 -N -L 8018:localhost:8018 b310ai@<VM 位址>
```

驗證：

```bash
curl -s http://127.0.0.1:8018/health
curl -s -X POST http://127.0.0.1:8018/translate \
  -H 'Content-Type: application/json' \
  -d '{"text":"我要喝水"}'
```

### 3. 開啟前端

用瀏覽器開 `手語前端優化/index.html`，切到 **TEXT TO SIGN**（`#generation`），
輸入句子按「轉成手語」。

- 狀態列顯示「語言模型（Gemma 4 微調）翻譯完成，耗時 X s」＝**用的是真模型**。
- 顯示「未連上語言模型…改用內建對照表」＝通道沒開或服務沒起，前端自動退回原本的
  硬編碼樣本，畫面仍可展示。

## 換模型 / 換位址

- 換 adapter：重啟 `serve_model.py` 時指定不同 `--adapter`。
- 換 API 位址：瀏覽器 console 執行
  `localStorage.setItem('glossApi','http://127.0.0.1:9000')` 後重整。

## 注意

- 服務只綁 `127.0.0.1`，僅能經 SSH 通道存取，不對外網暴露。
- GPU 為共用生產機，測試完請結束服務：`pkill -f serve_model.py`。
- 前端以 `file://` 開啟時 Origin 為 `null`，故服務回 `Access-Control-Allow-Origin: *`。
