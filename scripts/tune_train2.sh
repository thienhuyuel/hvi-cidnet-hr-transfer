#!/bin/bash
# Round 2: extend the grid where round 1 hit the boundary. TRAIN subset only; selection metric = fidelity to teacher.
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for r in 1 2 4; do for e in 0.01 0.03 0.1; do
  $PY -W ignore -m study.transfer --split train --stride 23 --scales 2 4 --methods r1_bicubic r3_gf --gf_r $r --gf_eps $e --tag tune2_gf_r${r}_e${e}
done; done
for sp in 4 8; do for lam in 1e-8 1e-6; do for bl in 1 2; do
  $PY -W ignore -m study.transfer --split train --stride 23 --scales 2 4 --methods r5_bgu_rgb r6_bgu_hvi --sp $sp --lam $lam --blur $bl --tag tune2_bgu_sp${sp}_lam${lam}_bl${bl}
done; done; done
$PY -W ignore -m study.transfer --split train --stride 23 --scales 2 4 --methods r5_bgu_rgb r6_bgu_hvi --sp 8 --lam 1e-6 --nb 16 --tag tune2_bgu_sp8_lam1e-6_nb16
