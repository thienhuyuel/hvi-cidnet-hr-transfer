"""Paired Wilcoxon tests on UHD-LL per-image results (results/uhd_perimg.csv) -> results/paired_tests_uhd.txt"""
import pandas as pd
from scipy.stats import wilcoxon
d = pd.read_csv('results/uhd_perimg.csv'); out = []
for sc in (2, 4, 8):
    for a, b in (('r6_bgu_hvi', 'r5_bgu_rgb'), ('r7_bgu_rgb_dn', 'r5_bgu_rgb')):
        x = d[(d.scale == sc) & (d.method == a)].set_index('name'); y = d[(d.scale == sc) & (d.method == b)].set_index('name')
        for k in ('psnr_teacher', 'ssim_teacher', 'psnr_gt', 'ssim_gt', 'lpips_gt'):
            df = (x[k] - y[k]).dropna()
            out.append(f'x{sc} {a}-{b} {k}: n={len(df)} mean diff {df.mean():+.4f}  frac>0 {(df > 0).mean():.2f}  wilcoxon p={wilcoxon(df).pvalue:.1e}')
t = d[d.method == 'r4_tile512'].set_index('name')
for sc in (2, 4, 8):
    r = d[(d.scale == sc) & (d.method == 'r5_bgu_rgb')].set_index('name')
    out.append(f'x{sc} r5 vs r4_tile512 psnr_teacher: r5 better on {(r.psnr_teacher > t.psnr_teacher).sum()}/{len(r)}')
open('results/paired_tests_uhd.txt', 'w').write('\n'.join(out) + '\n'); print('\n'.join(out[-3:]))
