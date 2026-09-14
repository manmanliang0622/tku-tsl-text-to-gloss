#!/usr/bin/env bash
# 把常用語句整句對照（phrasebook.py）上線。只換 serve_model.py＋新增 phrasebook.py，
# checkpoint／門檻／候選參數都不動。先把兩個檔放到 ~/deploy-stage-phrasebook/ 再跑。
set -u
B="$HOME/0821_bundle"; MS="$B/model_service"; STAGE="$HOME/deploy-stage-phrasebook"
TS=$(date +%m%d-%H%M); BAK="$HOME/deploy-bak-phrasebook-$TS"
echo "=== 看門狗 log 最後時間（搶不到鎖是靜默退出，要看這個）==="
tail -1 "$HOME/tsl-autopublish/autopublish.log" | cut -c1-140; date +%T
echo "=== 1/4 備份 → $BAK ==="
mkdir -p "$BAK" && cp "$MS/scripts/serve_model.py" "$BAK/" && ls "$BAK"
echo "=== 2/4 換檔 ==="
cp "$STAGE/serve_model.py" "$STAGE/phrasebook.py" "$MS/scripts/"
"$B/.venv/bin/python3" -m py_compile "$MS/scripts/serve_model.py" "$MS/scripts/phrasebook.py" && echo "py_compile ok"
md5sum "$MS/scripts/serve_model.py" "$MS/scripts/phrasebook.py"
echo "=== 3/4 重啟：先砍父程序 bundle_server，再砍 serve_model；看門狗 2 分鐘內接回 ==="
pkill -f "[b]undle_server.py --host 127.0.0.1 --port 8084"; sleep 2; pkill -f "[s]erve_model.py"; sleep 3
pkill -9 -f "[b]undle_server.py --host 127.0.0.1 --port 8084" 2>/dev/null; pkill -9 -f "[s]erve_model.py" 2>/dev/null
echo "砍完 $(date +%T)，等看門狗…"
for i in $(seq 1 24); do
  sleep 15
  h=$(curl -s -m 5 http://127.0.0.1:8084/health 2>/dev/null)
  if echo "$h" | grep -q '"status": "ok"'; then echo "OK at $(date +%T)（第 $i 次輪詢）"; break; fi
  echo "  $(date +%T) $(echo "$h" | grep -o '"model_process_alive": [a-z]*' || echo '後端未起')"
done
echo "=== 4/4 驗證 ==="
curl -s -m 5 http://127.0.0.1:8084/health | cut -c1-200; echo
grep -h "常用語句對照\|phrasebook\|Traceback" "$B/logs/v11-model.log" "$B/logs/v11-model.err.log" 2>/dev/null | tail -3 | cut -c1-200
for t in "請再說一次" "請再說一次。" "請你再說一次" "我住在台北"; do
  curl -s -m 60 -X POST http://127.0.0.1:8084/translate -H "Content-Type: application/json" \
    -d "{\"text\":\"$t\"}" | python3 -c "import sys,json;d=json.load(sys.stdin);print('$t →',d.get('gloss_text'),d.get('source'),d.get('phrasebook_id',''))"
done
