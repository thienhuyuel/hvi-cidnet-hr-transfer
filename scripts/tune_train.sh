#!/bin/bash
# Hyper-parameter selection on a LOLv2-real TRAIN subset (every 23rd pair = 30 images). Never touches test.
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for r in 4 8 16; do for e in 1e-4 1e-3 1e-2; do
  $PY -W ignore -m study.transfer --split train --stride 23 --scales 2 4 --methods r3_gf r2b_gain_gf --gf_r $r --gf_eps $e --tag tune_gf_r${r}_e${e}
done; done
for sp in 8 16 32; do for lam in 1e-6 1e-4 1e-2; do
  $PY -W ignore -m study.transfer --split train --stride 23 --scales 2 4 --methods r5_bgu_rgb r6_bgu_hvi --sp $sp --lam $lam --tag tune_bgu_sp${sp}_lam${lam}
done; done
