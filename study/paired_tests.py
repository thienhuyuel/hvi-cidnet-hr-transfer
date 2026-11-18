"""Paired Wilcoxon signed-rank tests between methods from per-image CSVs (results/transfer_<tag>_perimg.csv).
usage: python paired_tests.py <csv> <out.txt> A:B [A:B ...]   (diff = A minus B per image; frac>0 = share of images with A>B)"""
import sys, csv
import numpy as np
from scipy.stats import wilcoxon
src, out, pairs = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = list(csv.DictReader(open(src)))
mets = ['psnr_teacher', 'ssim_teacher', 'lpips_gt', 'psnr_gt', 'ssim_gt']
lines = []
for s in sorted({int(r['scale']) for r in rows if r['scale'] != '1'}):
    for pr in pairs:
        a, b = pr.split(':')
        da = {r['name']: r for r in rows if int(r['scale']) == s and r['method'] == a}
        db = {r['name']: r for r in rows if int(r['scale']) == s and r['method'] == b}
        names = sorted(da)
        for m in mets:
            d = np.array([float(da[n][m]) - float(db[n][m]) for n in names])
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            lines.append(f'x{s} {a}-{b} {m}: n={len(d)} mean diff {d.mean():+.4f}  frac>0 {np.mean(d > 0):.2f}  wilcoxon p={p:.1e}')
open(out, 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines))
