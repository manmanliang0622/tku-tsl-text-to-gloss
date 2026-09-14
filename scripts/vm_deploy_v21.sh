#!/usr/bin/env bash
# 把線上服務從 v19 換成 v21。與 v19 部署（deploy-bak-v19-0907-0501）同一套四件事：
# checkpoint、candidate_config.json、serve_model.py（門檻）、bundle_server.py（EXPECTED_MODEL），
# 另加 script_schema.py（serve_model 的宣告相依，這版有改）。先備份、再換、再重啟。
set -u
B="$HOME/0821_bundle"; MS="$B/model_service"; STAGE="$HOME/deploy-stage-v21"
TS=$(date +%m%d-%H%M); BAK="$HOME/deploy-bak-v21-$TS"
echo "=== 1/5 備份 → $BAK ==="
mkdir -p "$BAK" && cp "$B/bundle_server.py" "$MS/candidate_config.json" "$MS/scripts/serve_model.py" "$MS/scripts/script_schema.py" "$BAK/" && cp -r "$MS/checkpoint" "$BAK/checkpoint" && ls "$BAK"
echo "=== 2/5 換 checkpoint（v18 的 checkpoint.old 移除，原件在 ~/outputs/qlora_e4b_v18script）==="
rm -rf "$MS/checkpoint.old" && mv "$MS/checkpoint" "$MS/checkpoint.old" && cp -r "$HOME/outputs/qlora_e4b_v21script/checkpoint-349" "$MS/checkpoint" && du -sh "$MS/checkpoint" "$MS/checkpoint.old"
echo "=== 3/5 換 candidate_config、serve_model、script_schema、EXPECTED_MODEL ==="
cp "$HOME/tsl-v18/data/splits_script_v21/candidate_config.json" "$MS/candidate_config.json"
cp "$STAGE/serve_model.py" "$STAGE/script_schema.py" "$MS/scripts/"
sed -i 's/^EXPECTED_MODEL = "qlora_e4b_v19script"/EXPECTED_MODEL = "qlora_e4b_v21script"/' "$B/bundle_server.py"
grep -n "^EXPECTED_MODEL" "$B/bundle_server.py"; grep -n "^NEEDS_REVIEW_THRESHOLD" "$MS/scripts/serve_model.py" | cut -c1-70
"$B/.venv/bin/python3" -m py_compile "$B/bundle_server.py" "$MS/scripts/serve_model.py" "$MS/scripts/script_schema.py" && echo "py_compile ok"
cd "$MS/scripts" && for f in comitative.py constrained_decode.py eval_video_coverage.py gloss_fallback.py prompt_common.py rag_retrieve.py script_schema.py sign_candidates.py train_qlora.py serve_model.py; do [ -f "$f" ] || echo "缺 $f"; done; echo "相依檢查完畢"
echo "=== 4/5 重啟：先砍父程序 bundle_server，再砍 serve_model；看門狗 2 分鐘內接回 ==="
pkill -f "[b]undle_server.py --host 127.0.0.1 --port 8084"; sleep 2; pkill -f "[s]erve_model.py"; sleep 3
pkill -9 -f "[b]undle_server.py --host 127.0.0.1 --port 8084" 2>/dev/null; pkill -9 -f "[s]erve_model.py" 2>/dev/null
echo "砍完 $(date +%T)，等看門狗…"
for i in $(seq 1 20); do
  sleep 15
  h=$(curl -s -m 5 http://127.0.0.1:8084/health 2>/dev/null)
  if echo "$h" | grep -q '"status": "ok"'; then echo "OK at $(date +%T)（第 $i 次輪詢）"; break; fi
  echo "  $(date +%T) $(echo "$h" | grep -o '"model_process_alive": [a-z]*' || echo '後端未起')"
done
echo "=== 5/5 驗證 ==="
curl -s -m 5 http://127.0.0.1:8084/health | cut -c1-260; echo
grep -hE "候選參數|Traceback|Error|門檻|NEEDS_REVIEW|載入" "$B/logs/v11-model.log" "$B/logs/v11-model.err.log" 2>/dev/null | tail -6 | cut -c1-160
curl -s -m 60 -X POST http://127.0.0.1:8084/translate -H "Content-Type: application/json" -d '{"text":"我住在台北"}' | cut -c1-220; echo
tail -2 "$HOME/tsl-autopublish/autopublish.log" | cut -c1-140
