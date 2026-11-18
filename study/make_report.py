"""Build paper/tables.tex and figures from results/*.csv (no hand-typed numbers)."""
import csv, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('paper', exist_ok=True)
S = list(csv.DictReader(open('results/transfer_summary.csv')))
def get(tag, scale, method):
    return next(r for r in S if r['tag'] == tag and r['scale'] == str(scale) and r['method'] == method)
f = lambda r, k, d=2: f'{float(r[k]):.{d}f}'

# ---- Table 1: reproduction
rep = list(csv.DictReader(open('results/reproduce.csv')))
r0 = next(r for r in rep if 'use_GT_mean=False' in r['config'])
t1 = r'''\begin{table}[!htbp]
\centering
\caption{Reproduction of the released LOLv2-real \texttt{best\_PSNR} checkpoint (full resolution, no GT-mean rescaling). ``Reported'' is the value in the official repository README.}
\label{tab:repro}
\begin{tabular}{lccc}
\toprule
 & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ \\
\midrule
Reported (README) & 23.904 & 0.8656 & 0.1219 \\
Ours (MPS, $n{=}%s$) & %s & %s & %s \\
\bottomrule
\end{tabular}
\end{table}
''' % (r0['n'], f(r0, 'psnr', 3), f(r0, 'ssim', 4), f(r0, 'lpips', 4))

# ---- Table 2: ablation ladder
methods = [('r1_bicubic', 'R1 Bicubic upsampling of $O_{lr}$'), ('r2a_gain', 'R2 Gain ratio, bicubic'),
           ('r2b_gain_gf', 'R2$'+"'"+'$ Gain ratio, guided filter'), ('r3_gf', 'R3 Guided-filter upsampling'),
           ('r5_bgu_rgb', 'R5 Bilateral-grid affine, RGB'), ('r6_bgu_hvi', 'R6 Bilateral-grid affine, HVI'),
           ('r7_bgu_rgb_dn', 'R7 R5 + gain-aware denoising')]
rows = []
R3TAG = {'r5_bgu_rgb': 'test_r3_rgb', 'r6_bgu_hvi': 'test_r3_hvi', 'r7_bgu_rgb_dn': 'test_r7'}
for sc in (2, 4, 8):
    ms = methods if sc != 8 else [x for x in methods if x[0] in ('r1_bicubic', 'r5_bgu_rgb', 'r6_bgu_hvi', 'r7_bgu_rgb_dn')]
    vals = {m: get(R3TAG.get(m, 'test_final' if sc != 8 else 'test_r3_rgb'), sc, m) for m, _ in ms}
    best = {k: (max if k.startswith(('psnr', 'ssim')) else min)(float(v[k]) for v in vals.values())
            for k in ('psnr_teacher', 'ssim_teacher', 'psnr_gt', 'ssim_gt', 'lpips_gt')}
    rows.append(r'\multicolumn{7}{l}{\textit{Downscale factor $\times%d$}} \\' % sc)
    for m, name in ms:
        v = vals[m]; cells = []
        for k, d in (('psnr_teacher', 2), ('ssim_teacher', 3), ('psnr_gt', 2), ('ssim_gt', 3), ('lpips_gt', 3)):
            c = f(v, k, d); cells.append(r'\textbf{%s}' % c if abs(float(v[k]) - best[k]) < 10 ** -(d + 1) * 0.5 else c)
        tt = float(v['t_transfer']) * 1000
        rows.append(f'{name} & ' + ' & '.join(cells) + f' & {tt:.0f} \\\\')
tl = get('test_final', 1, 'teacher_fullres')
rows.append(r'\midrule')
rows.append(r'Teacher (full-res CIDNet) & -- & -- & %s & %s & %s & -- \\' % (f(tl, 'psnr_gt'), f(tl, 'ssim_gt', 3), f(tl, 'lpips_gt', 3)))
for t in (512, 384, 256, 128):
    v = get('test_tiles2' if t in (512, 384) else 'test_tiles', 1, f'r4_tile{t}')
    rows.append(r'R4 Tiling, %d px (overlap %d) & %s & %s & %s & %s & %s & %.0f \\' % (
        t, t // 4, f(v, 'psnr_teacher'), f(v, 'ssim_teacher', 3), f(v, 'psnr_gt'), f(v, 'ssim_gt', 3), f(v, 'lpips_gt', 3), float(v['t_net_lr']) * 1000))
t2 = r'''\begin{table}[!htbp]
\centering
\caption{Ablation ladder on LOLv2-real test (100 images of $600{\times}400$). Left block: fidelity to the full-resolution teacher; right block: quality against ground truth. $t$ is the transfer time (ms) excluding the network (grid fit and slicing on CPU; for R6 the HVI conversions run on MPS); for R4 it is the full tiled inference time (MPS). Best per scale in bold (R4 and teacher excluded). R5 and R6 use hyper-parameters selected separately for each color space on the training subset ($\sigma_s{=}2$, $n_b{=}8$, one smoothing pass, $\lambda{=}10^{-8}$ for RGB and $10^{-10}$ for HVI). R7 uses the R5 grid and denoising parameters selected on the same subset (guided-filter radius 2, $\epsilon{=}0.016$, gain ramp 2 to 8).}
\label{tab:ablation}
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lcc|ccc|c}
\toprule
 & \multicolumn{2}{c|}{vs.\ teacher} & \multicolumn{3}{c|}{vs.\ GT} & $t$ \\
Method & PSNR & SSIM & PSNR & SSIM & LPIPS$\downarrow$ & (ms) \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
''' % '\n'.join(rows)

# ---- Table 3: end-to-end cost on a 10.8 MP image
P = [r for r in csv.DictReader(open('results/pipeline.csv')) if r['mode'] in ('full', 'tile')] + list(csv.DictReader(open('results/pipeline_r3.csv')))
names = {'full': 'Full-resolution CIDNet', 'tile': 'Tiled CIDNet (R4)', 'lowres_rgb': 'Low-res + BGU, RGB (R5)', 'lowres_hvi': 'Low-res + BGU, HVI (R6)'}
order = [('full', '0'), ('tile', '512'), ('tile', '256'), ('lowres_rgb', '2'), ('lowres_hvi', '2'), ('lowres_rgb', '4'), ('lowres_hvi', '4'), ('lowres_rgb', '8'), ('lowres_hvi', '8')]
rr = []
for m, par in order:
    r = next(x for x in P if x['mode'] == m and x['param'] == par)
    cfg = {'full': '--', 'tile': f'{par}\\,px'}.get(m, f'$1/{par}$')
    net = float(r['time_net_s']); tr = f"{float(r['time_transfer_s']):.1f}" if r['time_transfer_s'] else '--'
    rr.append(f"{names[m]} & {cfg} & {net:.1f} & {tr} & {float(r['time_total_s']):.1f} & {float(r['mps_driver_gb']):.1f} \\\\")
t3 = r'''\begin{table}[!htbp]
\centering
\caption{End-to-end cost on one $4032{\times}2688$ (10.8\,MP) image on an Apple M1 (32\,GB, PyTorch MPS), one cold run per row. Network time $t_{net}$ includes downscaling; transfer time $t_{tr}$ is the bilateral-grid fit and slicing on CPU. Peak is the MPS driver allocation.}
\label{tab:cost}
\begin{tabular}{llrrrr}
\toprule
Pipeline & Setting & $t_{net}$ (s) & $t_{tr}$ (s) & Total (s) & Peak (GB) \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
''' % '\n'.join(rr)
U = list(csv.DictReader(open('results/uhd_summary.csv')))
ug = lambda sc, m: next(r for r in U if r['scale'] == str(sc) and r['method'] == m)
ur = []
tu = ug(1, 'teacher_fullres')
ur.append(r'Teacher (full-res CIDNet) & -- & -- & -- & %s & %s & %s & %.1f \\' % (f(tu, 'psnr_gt'), f(tu, 'ssim_gt', 3), f(tu, 'lpips_gt', 3), float(tu['t_net_lr'])))
for t in (512, 256):
    v = ug(1, f'r4_tile{t}')
    ur.append(r'R4 Tiling & %d\,px & %s & %s & %s & %s & %s & %.1f \\' % (t, f(v, 'psnr_teacher'), f(v, 'ssim_teacher', 3), f(v, 'psnr_gt'), f(v, 'ssim_gt', 3), f(v, 'lpips_gt', 3), float(v['t_net_lr'])))
UN = {'r1_bicubic': 'R1 Bicubic', 'r5_bgu_rgb': 'R5 BGU, RGB', 'r6_bgu_hvi': 'R6 BGU, HVI', 'r7_bgu_rgb_dn': 'R7 R5 + denoising'}
for sc in (2, 4, 8):
    ur.append(r'\midrule')
    for m in UN:
        v = ug(sc, m)
        tt = '%.1f' % (float(ug(sc, 'r5_bgu_rgb')['t_net_lr']) + float(v['t_transfer']))
        ur.append(r'%s & $1/%d$ & %s & %s & %s & %s & %s & %s \\' % (UN[m], sc, f(v, 'psnr_teacher'), f(v, 'ssim_teacher', 3), f(v, 'psnr_gt'), f(v, 'ssim_gt', 3), f(v, 'lpips_gt', 3), tt))
t4 = r'''\begin{table}[!htbp]
\centering
\caption{Real high-resolution evaluation on UHD-LL test ($n{=}%s$ pairs of $3840{\times}2160$, every 5th of the 150 sorted test pairs, fixed before running). Same checkpoint and frozen hyper-parameters as Table~\ref{tab:ablation}; no re-tuning on UHD-LL. LPIPS is computed at full 4K resolution. $t$ is the mean end-to-end time per image on the M1 (low-resolution network + transfer; for R4 and the teacher, the whole inference); timings are noisy because other jobs shared the machine. Teacher peak MPS allocation: %s\,GB on average (max %s\,GB).}
\label{tab:uhd}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{ll|cc|ccc|r}
\toprule
 & & \multicolumn{2}{c|}{vs.\ teacher} & \multicolumn{3}{c|}{vs.\ GT} & $t$ \\
Method & Setting & PSNR & SSIM & PSNR & SSIM & LPIPS$\downarrow$ & (s) \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
''' % (tu['n'], f(tu, 'mps_driver_gb_mean', 1), f(tu, 'mps_driver_gb_max', 1), '\n'.join(ur))
open('paper/tables.tex', 'w').write('% generated by make_report.py -- do not edit\n' + t1 + '\n' + t2 + '\n' + t3 + '\n' + t4)

# ---- Figure: memory / time vs megapixels (single-pass full-res inference)
sc = [r for r in csv.DictReader(open('results/scaling.csv')) if r['status'] == 'ok']
mp = [float(r['mpix']) for r in sc]; tm = [float(r['time_s']) for r in sc]; mem = [float(r['mps_driver_gb']) for r in sc]
fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.4))
ax[0].loglog(mp, mem, 'o-'); ax[0].set_ylim(top=60); ax[0].axhline(32, ls='--', c='gray', lw=.8); ax[0].text(0.18, 36, 'unified memory 32 GB', fontsize=7, c='gray')
ax[0].set_xlabel('image size (MP)'); ax[0].set_ylabel('MPS memory (GB)')
ax[1].loglog(mp, tm, 'o-', c='C1'); ax[1].set_xlabel('image size (MP)'); ax[1].set_ylabel('time (s, cold run)')
for a in ax: a.grid(alpha=.3, which='both')
plt.tight_layout(); plt.savefig('paper/fig_scaling.pdf')
print(open('paper/tables.tex').read())

# ---- Figure: quality/cost trade-off. Cost measured at 10.8 MP; quality measured on LOLv2-real test at the same factor / tile size.
def q(tag, sc, m): r = get(tag, sc, m); return float(r['psnr_teacher']), float(r['lpips_gt'])
pts = []
cost = {(r['mode'], r['param']): float(r['time_total_s']) for r in P}
for d, sc in (('2', 2), ('4', 4), ('8', 8)):
    pts.append(('BGU-RGB', d, cost[('lowres_rgb', d)], *q('test_r3_rgb', sc, 'r5_bgu_rgb')))
    pts.append(('BGU-HVI', d, cost[('lowres_hvi', d)], *q('test_r3_hvi', sc, 'r6_bgu_hvi')))
for t, tag in (('512', 'test_tiles2'), ('256', 'test_tiles')):
    pts.append(('Tiling', t, cost[('tile', t)], *q(tag, 1, f'r4_tile{t}')))
tl = get('test_final', 1, 'teacher_fullres')
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
mk = {'BGU-RGB': ('o', 'C0'), 'BGU-HVI': ('s', 'C1'), 'Tiling': ('^', 'C2')}
for name in mk:
    P_ = sorted([p for p in pts if p[0] == name], key=lambda p: p[2])
    for k, (axi, j) in enumerate(((ax[0], 3), (ax[1], 4))):
        axi.plot([p[2] for p in P_], [p[j] for p in P_], marker=mk[name][0], c=mk[name][1], label=name, lw=1)
        for p in P_:
            axi.annotate(('1/' + p[1]) if name != 'Tiling' else p[1], (p[2], p[j]), fontsize=6, xytext=(3, 2), textcoords='offset points')
ax[1].scatter([cost[('full', '0')]], [float(tl['lpips_gt'])], marker='*', c='k', s=60, label='Full-res', zorder=5)
ax[0].set_ylabel('PSNR to teacher (dB)'); ax[1].set_ylabel('LPIPS to GT (lower is better)')
for a in ax: a.set_xscale('log'); a.set_xlabel('time at 10.8 MP (s)'); a.grid(alpha=.3, which='both')
ax[1].legend(fontsize=6)
plt.tight_layout(); plt.savefig('paper/fig_pareto.pdf')
