"""Reproduce the paper-table metrics of a released checkpoint (full resolution).
python reproduce.py --set lolv2_real --device mps [--use_GT_mean]
Appends a summary row to results/reproduce.csv and per-image rows to results/reproduce_<set>_perimg.csv
"""
import argparse, csv, os, time
import numpy as np, torch
from study.common import *

p = argparse.ArgumentParser()
p.add_argument('--set', default='lolv2_real', choices=list(SETS))
p.add_argument('--ckpt', default='lolv2_real_bestpsnr')
p.add_argument('--device', default='mps')
p.add_argument('--use_GT_mean', action='store_true')
p.add_argument('--limit', type=int, default=0)
a = p.parse_args()

dev = torch.device(a.device)
net = load_model(dev, a.ckpt)
cfg = config_str(net, a.ckpt, a.device, a.use_GT_mean)
pairs = load_pairs(a.set)[:a.limit or None]
os.makedirs('results', exist_ok=True)
rows = []
for lo, hi in pairs:
    x, gt = read(lo), read(hi)
    t = time.perf_counter()
    y = to_uint8(run_cidnet(net, to_tensor(x), dev))
    dt = time.perf_counter() - t
    m = all_metrics(y, gt, a.use_GT_mean)
    rows.append(dict(name=os.path.basename(lo), time_s=round(dt, 3), **m))
    print(rows[-1], flush=True)

mean = {k: float(np.mean([r[k] for r in rows])) for k in ('psnr', 'ssim', 'lpips', 'time_s')}
print('MEAN', mean, '\n', cfg)
with open(f'results/reproduce_{a.set}_perimg.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
new = not os.path.exists('results/reproduce.csv')
with open('results/reproduce.csv', 'a', newline='') as f:
    w = csv.writer(f)
    if new:
        w.writerow(['timestamp', 'set', 'n', 'psnr', 'ssim', 'lpips', 'mean_time_s', 'config'])
    w.writerow([time.strftime('%F %T'), a.set, len(rows), round(mean['psnr'], 4), round(mean['ssim'], 4),
                round(mean['lpips'], 4), round(mean['time_s'], 3), cfg])
