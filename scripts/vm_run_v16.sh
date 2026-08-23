#!/usr/bin/env bash
# v16：k=60 + 語義通道換血（v14 配方：dropout 0.05、無 nr 加權）。
# 訓練結束立刻接推論，不留 GPU 空窗（2026-08-23 教訓：看門狗 2 分鐘內會把 v14 載回來）。
set -u
cd ~/tku-tsl-text-to-gloss/scripts
DATA=~/tku-tsl-text-to-gloss/data/splits_script_k60sem
OUT=~/outputs/qlora_e4b_v16script_k60sem
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
~/unsloth-venv/bin/python train_unsloth.py --format script --data "$DATA" --output "$OUT" \
  --epochs 2 --batch 2 --grad-accum 8 --lora-dropout 0.05 > ~/v16_train.log 2>&1
BEST=$(grep BEST_EPOCH ~/v16_train.log | sed -E 's/.*epoch=([0-9.]+).*/\1/')
mapfile -t CKPTS < <(ls -d "$OUT"/checkpoint-* | sort -t- -k2 -n)
if [ -z "${BEST:-}" ] || [ "${#CKPTS[@]}" -eq 0 ]; then echo "TRAIN_FAILED" > ~/v16_infer.log; exit 1; fi
if awk "BEGIN{exit !($BEST < 1.5)}"; then CK="${CKPTS[0]}"; else CK="${CKPTS[-1]}"; fi
echo "USING $CK (BEST_EPOCH=$BEST)" > ~/v16_infer.log
for s in test test_corpus test_textbook dev; do
  echo "=== $s ===" >> ~/v16_infer.log
  ~/unsloth-venv/bin/python infer_script_model.py --adapter "$CK" --split "$s" --tag v16script \
    --data "$DATA" >> ~/v16_infer.log 2>&1 || echo "INFER_FAIL_$s" >> ~/v16_infer.log
done
echo ALL_DONE >> ~/v16_infer.log
