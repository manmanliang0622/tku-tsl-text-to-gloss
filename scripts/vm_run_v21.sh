#!/usr/bin/env bash
# v21：訓練 → 推論 → 選門檻 → 評分 → 產 v19/v21 盲測表。一條龍，中途不放開 GPU。
#
# 與 v19 唯一差異：資料用新的 --min-coverage 預設（豁免「整詞缺但單字都有」的 OOV，
# commit f9692ed）。教「不要拆字」的例子由 19 列回到 58 列；其餘設定完全相同
# （k 60、LOGO、min-coverage 0.8、schema V3、無前文）。
# 資料：splits_script_v21。驗收重點：test_textbook 的 TB0081/TB0371/TB0390/TB0392/TB0393
# （調香師、毛利人、羅馬競技場）——看 oov_items 是否列整詞、sign_ids 是否不再有碎片。
#
# 前置（**本腳本不動線上服務**）：先建 training.lock、停掉佔卡的線上模型，
# 再跑這支。顯存不足直接退出。跑完自動解鎖，看門狗 2 分鐘內接回。
set -u
REPO="$HOME/tsl-v18"
LOCK="$HOME/tsl-autopublish/training.lock"
OUT="$HOME/outputs/qlora_e4b_v21script"
DATA="$REPO/data/splits_script_v21"
PY="$HOME/unsloth-venv/bin/python"
exec > >(tee -a "$HOME/v21_run.log") 2>&1
echo "=== v21 開始 $(date '+%F %T') ==="

cleanup() {
  rc=$?
  echo "--- 移除訓練鎖（rc=$rc）$(date '+%F %T') ---"
  rm -f "$LOCK"
  echo "看門狗最多 2 分鐘內把線上服務接回。"
  exit $rc
}
trap cleanup EXIT INT TERM

free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
echo "GPU 可用 ${free} MiB（訓練峰值需 ~11000）"
if [ "$free" -lt 11500 ]; then
  echo "✗ 顯存不足，請先停掉線上模型。"
  exit 2
fi
# 2026-09-09 實證：只砍 serve_model 子程序沒用，bundle_server 秒級重生它，
# 訓練在 CPU tokenization 那 1 分鐘裡被搶走 9.2GB，model.to(device) 直接 OOM。
# 要先砍父程序 bundle_server（鎖在，看門狗會以 /clips-only 模式接回前端）。
if pgrep -f "[s]erve_model.py" >/dev/null; then
  echo "✗ 線上模型程序還活著（bundle_server 會自動重生子程序，要先砍父程序 bundle_server）。"
  exit 2
fi
touch "$LOCK"

cd "$REPO" || exit 1
[ -f "$DATA/train.jsonl" ] || { echo "✗ 找不到 $DATA/train.jsonl"; exit 1; }
echo "--- 1/5 訓練（$(wc -l < "$DATA/train.jsonl") 列）---"
$PY scripts/train_script_qlora.py --data "$DATA" --output "$OUT" \
    --epochs 2 --max-len 768 --seed 42 || { echo "✗ 訓練失敗"; exit 1; }

CKPT=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | head -1)
[ -z "$CKPT" ] && { echo "✗ 找不到 checkpoint"; exit 1; }
echo "--- 選用 $CKPT（epoch 1，依既定規則）---"

echo "--- 2/5 推論（不放開 GPU）---"
$PY scripts/infer_script_model.py --adapter "$CKPT" --data "$DATA" \
    --split dev test test_corpus test_textbook --tag v21script \
    || { echo "✗ 推論失敗"; exit 1; }

echo "--- 3/5 在 dev 上選門檻（只能用 dev）---"
$PY scripts/nr_threshold.py select "$REPO/results/v21script_dev.jsonl" \
    | tee "$REPO/results/v21_nr_threshold.txt"
THR=$(grep -oE "t=0\.[0-9]+" "$REPO/results/v21_nr_threshold.txt" | head -1 | cut -d= -f2)
[ -z "$THR" ] && THR=0.0421
echo "使用門檻 $THR"

echo "--- 4/5 評分 ---"
for s in dev test test_corpus test_textbook; do
  $PY scripts/eval_script_format.py --pred "$REPO/results/v21script_$s.jsonl" \
      --threshold "$THR" --overwrite > /dev/null && echo "  ✓ $s"
done

echo "--- 4b 專有名詞五句（v19 vs v21）---"
$PY - <<'PYEOF'
import json
ids = {"TB0081","TB0371","TB0390","TB0392","TB0393"}
for tag in ("v19script","v21script"):
    print(f"[{tag}]")
    for l in open(f"results/{tag}_test_textbook.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r.get("id") in ids:
            print(" ", json.dumps(r, ensure_ascii=False)[:500])
PYEOF

echo "--- 5/5 產 v19 vs v21 盲測表 ---"
$PY -c "import openpyxl" 2>/dev/null && \
  $PY scripts/make_ab_eval_sheet.py --a v19script --b v21script \
      --n-core 8 --n-corpus 40 --n-textbook 40 \
  || echo "  （openpyxl 不在，盲測表改在本機產）"

echo "=== 完成 $(date '+%F %T') ==="
echo "指標：$REPO/results/v21script_*_scriptmetrics.json"
