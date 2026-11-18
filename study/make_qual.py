"""Qualitative figure: test images with index 0 and 50 (sorted), downscale x4, fixed crop. -> paper/fig_qual.png"""
import cv2, numpy as np, torch
from study.transfer import *
dev = torch.device('mps'); R = Runner(dev, 'lolv2_real_bestpsnr')
p = dict(gf_r=1, gf_eps=0.01, sp=2, nb=8, lam=1e-8, blur=1)  # round-3 selection (lam 1e-10 for HVI)
lam = {'r5_bgu_rgb': 1e-8, 'r6_bgu_hvi': 1e-10}
CROP = (220, 120, 180, 120)  # x, y, w, h (fixed a priori)
cols = ['Input', 'Teacher', 'R1 bicubic', 'R5 BGU-RGB', 'R6 BGU-HVI', 'Ground truth']
def lab(im, t):
    im = np.ascontiguousarray(im); cv2.putText(im, t, (6, 20), 0, 0.6, (255, 255, 0), 2); return im
rows = []
for lo, hi in load_pairs('lolv2_real')[::50]:
    I8, G8 = read(lo), read(hi); H, W = I8.shape[:2]
    I_hr, O, _ = R.enhance(I8)
    lr = cv2.resize(I8, (W // 4, H // 4), interpolation=cv2.INTER_AREA); I_lr, O_lr, Oh = R.enhance(lr)
    ims = [I8, to_uint8(O[None])] + [to_uint8(build(m, R, I_hr, I_lr, O_lr, Oh, {**p, 'lam': lam.get(m, p['lam'])})[None]) for m in ('r1_bicubic', 'r5_bgu_rgb', 'r6_bgu_hvi')] + [G8]
    x, y, w, h = CROP
    full = [lab(cv2.rectangle(im.copy(), (x, y), (x + w, y + h), (255, 0, 0), 2), c) for im, c in zip(ims, cols)]
    crop = [cv2.resize(im[y:y + h, x:x + w], (W // 2, int(H / 2 * h / w * 0.0 + (W // 2) * h / w)), interpolation=cv2.INTER_NEAREST) for im in ims]
    full = [cv2.resize(f, (W // 2, H // 2), interpolation=cv2.INTER_AREA) for f in full]
    rows += [np.hstack(full), np.hstack(crop)]
w = max(r.shape[1] for r in rows)
rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
cv2.imwrite('paper/fig_qual.png', np.vstack(rows)[..., ::-1])
