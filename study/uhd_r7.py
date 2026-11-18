"""Recompute ONLY the R7 rows of the UHD-LL evaluation with the train-selected R7 parameters
(dn_r=2, dn_eps=0.016, g0=2, g1=8; sp=2 nb=8 blur=1 lam=1e-8) instead of the run-time defaults (dn_r=4, dn_eps=4e-3) used by uhd_eval.py.
Reuses cached teacher PNGs (results/uhd_cache/teacher_*.png); writes results/uhd_cache/r7sel_*.json, then run
`python uhd_eval.py --aggregate` to rebuild uhd_perimg.csv / uhd_summary.csv (old R7 rows are kept as r7_bgu_rgb_dn_default)."""
import glob, json, os, time, cv2, numpy as np, torch
from study.common import *
from study import transfer, uhd_eval as U

P = dict(sp=2, nb=8, blur=1, lam=1e-8, dn_r=2, dn_eps=0.016, g0=2.0, g1=8.0)
R = transfer.Runner(torch.device('mps'), 'lolv2_real_bestpsnr')
for lo, hi in U.subset():
    name = os.path.basename(lo); out = f'{U.CACHE}/r7sel_{name[:-4]}.json'
    if os.path.exists(out): continue
    I8, G8 = U.load_hr(lo, 1), U.load_hr(hi, 1)
    T8 = cv2.cvtColor(cv2.imread(f'{U.CACHE}/teacher_{name[:-4]}.png'), cv2.COLOR_BGR2RGB)
    I_hr = to_tensor(I8)[0]; H, W = I8.shape[:2]; rows = []
    for s in [4, 8, 2]:
        lr8 = cv2.resize(I8, (W // s, H // s), interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter(); I_lr, O_lr, Oh = R.enhance(lr8); t_net = time.perf_counter() - t0
        t0 = time.perf_counter(); o = transfer.build('r7_bgu_rgb_dn', R, I_hr, I_lr, O_lr, Oh, dict(P)); P8 = to_uint8(o[None])
        t_tr = time.perf_counter() - t0
        rows.append(dict(name=name, scale=s, method='r7_bgu_rgb_dn', t_net_lr=round(t_net, 3), t_transfer=round(t_tr, 3), **U.metrics(P8, G8, T8)))
    json.dump(rows, open(out, 'w')); print(name, flush=True)
