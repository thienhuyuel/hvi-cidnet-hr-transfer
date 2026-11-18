#!/bin/bash
# Round 4. TRAIN subset only (every 23rd pair, 30 images), scales 2 4.
# Part A: R7 (gain-aware denoising) grid, 12 configs, R5 params fixed (sp=2 nb=8 blur=1 lam=1e-8).
#   Selection rule (fixed before looking at test): max mean PSNR-to-teacher over x2 and x4;
#   LPIPS-GT only overrides if it clearly (consistently) argues against.
# Part B: sigma_s=1 check for R5 (lam 1e-8) and R6 (lam 1e-10), nb=8, blur 0 and 1.
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$(dirname "$0")/.."
PY=.venv/bin/python
for r in 2 4; do for eps in 0.001 0.004 0.016; do for gg in "1 4" "2 8"; do
  set -- $gg
  $PY -W ignore -m study.transfer --split train --stride 23 --scales 2 4 --methods r7_bgu_rgb_dn \
      --sp 2 --lam 1e-8 --nb 8 --blur 1 --dn_r $r --dn_eps $eps --g0 $1 --g1 $2 --tag tune4_r${r}_e${eps}_g$1-$2
done; done; done
for bl in 0 1; do
  $PY -W ignore -m study.transfer --split train --stride 23 --scales 2 4 --methods r5_bgu_rgb \
      --sp 1 --lam 1e-8 --nb 8 --blur $bl --tag tune4_sp1_rgb_bl${bl}
  $PY -W ignore -m study.transfer --split train --stride 23 --scales 2 4 --methods r6_bgu_hvi \
      --sp 1 --lam 1e-10 --nb 8 --blur $bl --tag tune4_sp1_hvi_bl${bl}
done
