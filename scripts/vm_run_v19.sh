#!/usr/bin/env bash
# v19：訓練 → 推論 → 選門檻 → 評分 → 產 v18/v19 盲測表。一條龍，中途不放開 GPU。
#
# 前置（**本腳本不動線上服務**）：先建 training.lock、停掉佔卡的線上模型，
# 再跑這支。顯存不足會直接退出，不會亂跑。跑完自動解鎖，看門狗 2 分鐘內接回。
#
# 為什麼串成一支：2026-08-23 的教訓——訓練一結束顯存釋放，看門狗 2 分鐘內
# 把線上模型載回 GPU，後發的推論只搶到殘渣、被 offload 到 CPU（331 秒/句）。
set -u
REPO="$HOME/tsl-v18"                 # worktree 名稱沿用，內容已是最新 branch
LOCK="$HOME/tsl-autopublish/training.lock"
OUT="$HOME/outputs/qlora_e4b_v19script"
DATA="$REPO/data/splits_script_v19"
PY="$HOME/unsloth-venv/bin/python"
exec > >(tee -a "$HOME/v19_run.log") 2>&1
echo "=== v19 開始 $(date '+%F %T') ==="

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
touch "$LOCK"

echo "--- 1/5 訓練（5,545 列，約為 v18 的 61%，時間應短於 v18）---"
cd "$REPO" || exit 1
$PY scripts/train_script_qlora.py --data "$DATA" --output "$OUT" \
    --epochs 2 --max-len 768 --seed 42 || { echo "✗ 訓練失敗"; exit 1; }

CKPT=$(ls -d "$OUT"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | head -1)
[ -z "$CKPT" ] && { echo "✗ 找不到 checkpoint"; exit 1; }
echo "--- 選用 $CKPT（epoch 1，依既定規則）---"

echo "--- 2/5 推論（不放開 GPU）---"
$PY scripts/infer_script_model.py --adapter "$CKPT" --data "$DATA" \
    --split dev test test_corpus test_textbook --tag v19script \
    || { echo "✗ 推論失敗"; exit 1; }

echo "--- 3/5 在 dev 上選門檻（只能用 dev）---"
$PY scripts/nr_threshold.py select "$REPO/results/v19script_dev.jsonl" \
    | tee "$REPO/results/v19_nr_threshold.txt"
THR=$(grep -oE "t=0\.[0-9]+" "$REPO/results/v19_nr_threshold.txt" | head -1 | cut -d= -f2)
[ -z "$THR" ] && THR=0.0421
echo "使用門檻 $THR"

echo "--- 4/5 評分 ---"
for s in dev test test_corpus test_textbook; do
  $PY scripts/eval_script_format.py --pred "$REPO/results/v19script_$s.jsonl" \
      --threshold "$THR" --overwrite > /dev/null && echo "  ✓ $s"
done

echo "--- 5/5 產 v18 vs v19 盲測表 ---"
# openpyxl 只在 unsloth venv 裡；沒有就跳過，不擋整條流程
$PY -c "import openpyxl" 2>/dev/null && \
  $PY scripts/make_ab_eval_sheet.py --a v18script --b v19script \
      --n-core 8 --n-corpus 40 --n-textbook 40 \
  || echo "  （openpyxl 不在，盲測表改在本機產）"

echo "=== 完成 $(date '+%F %T') ==="
echo "指標：$REPO/results/v19script_*_scriptmetrics.json"
