#!/bin/bash
# Frozen R7 + R5 (lam 1e-8) and R6 (lam 1e-10) on the TEST split, scales 2 4 8. R7 params selected on train (tune_train4.sh).
export PYTORCH_ENABLE_MPS_FALLBACK=1
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY -W ignore -m study.transfer --split test --scales 2 4 8 --methods r5_bgu_rgb r7_bgu_rgb_dn \
    --sp 2 --lam 1e-8 --nb 8 --blur 1 --dn_r 2 --dn_eps 0.016 --g0 2 --g1 8 --tag test_r7
$PY -W ignore -m study.transfer --split test --scales 2 4 8 --methods r6_bgu_hvi \
    --sp 2 --lam 1e-10 --nb 8 --blur 1 --tag test_r7_hvi
