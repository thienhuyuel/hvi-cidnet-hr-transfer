#!/bin/bash
# Round 3 (reviewer item 3): separate selection for RGB (R5) and HVI (R6), grid extended past the round-2 border.
# TRAIN subset only (every 23rd pair, 30 images). Selection: mean PSNR-to-teacher over x2 and x4, done per method.
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for sp in 2 4 8; do for lam in 1e-10 1e-8 1e-6; do for nb in 8 16; do for bl in 0 1; do
  $PY -W ignore -m study.transfer --split train --stride 23 --scales 2 4 --methods r5_bgu_rgb r6_bgu_hvi \
      --sp $sp --lam $lam --nb $nb --blur $bl --tag tune3_sp${sp}_lam${lam}_nb${nb}_bl${bl}
done; done; done; done
