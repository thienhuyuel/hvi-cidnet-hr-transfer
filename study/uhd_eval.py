"""UHD-LL (4K) evaluation of low-res -> high-res transfer of a CIDNet adjustment.

Subset (fixed before looking at any result): the sorted test inputs of UHD-LL, every 5th  -> 30 of 150 pairs.
HR = native 3840x2160 (--hr_div 1) or a fixed INTER_AREA downscale of both low and GT (--hr_div 2 -> 1920x1080).
teacher = CIDNet(HR);  O_lr = CIDNet(INTER_AREA downscale of HR by s);  methods build O_hr from (I_hr, I_lr, O_lr).
Methods: r1_bicubic, r5_bgu_rgb, r6_bgu_hvi, [r7_bgu_rgb_dn] at s in --scales; r4 tiling (overlap = tile/4).
Frozen hyper-parameters (paper): sp=2 nb=8 blur=1, lam=1e-8 (RGB, R7) / 1e-10 (HVI).
Metrics as transfer.py (uint8, PSNR/SSIM vs teacher and GT, LPIPS-alex vs GT on CPU at the evaluated HR size).

  python uhd_eval.py                       # run (resumable, per-image cache in results/uhd_cache/)
  python uhd_eval.py --aggregate           # only rebuild results/uhd_perimg.csv + uhd_summary.csv from the cache
  python uhd_eval.py --fig                 # paper/fig_uhd.png from cached crops
"""
import argparse, csv, glob, gc, json, os, resource, time
import cv2, numpy as np, torch
from study.common import *
from study import transfer

UHD = os.path.join(DATA, 'UHD-LL', 'testing_set')
CACHE = 'results/uhd_cache'
FIG_IMAGES = [3, 17]                        # indices into the 30-image subset (fixed in advance)
FIG_CROP = (1440, 720, 640, 400)            # x0, y0, w, h in 3840x2160 coordinates (centre region), fixed in advance
FIG_SCALE = 4
FIG_KEYS = ['input', 'teacher', 'r1_bicubic_s4', 'r5_bgu_rgb_s4', 'r4_tile512', 'gt']
P_RGB = dict(sp=2, nb=8, blur=1, lam=1e-8)
P_HVI = dict(sp=2, nb=8, blur=1, lam=1e-10)
P_R7 = dict(sp=2, nb=8, blur=1, lam=1e-8, dn_r=4, dn_eps=4e-3, g0=2.0, g1=8.0)  # R7 defaults of transfer.py at run time
KEYS = ['t_net_lr', 't_transfer', 'psnr_gt', 'ssim_gt', 'lpips_gt', 'psnr_teacher', 'ssim_teacher']


def subset():
    names = open('results/uhd_testlist.txt').read().split()  # 150 sorted test input names (Drive listing)
    assert len(names) == 150
    ins = [f'{UHD}/input/{n}' for n in names[::5]]
    return [(p, p.replace('/input/', '/gt/')) for p in ins]


def metrics(P8, G8, T8):
    mg = all_metrics(P8, G8)
    return dict(psnr_gt=mg['psnr'], ssim_gt=mg['ssim'], lpips_gt=mg['lpips'],
                psnr_teacher=float(psnr(P8, T8)), ssim_teacher=ssim(P8, T8))


def crop(a, div):
    x0, y0, w, h = [v // div for v in FIG_CROP]
    return a[y0:y0 + h, x0:x0 + w].copy()


def load_hr(path, div):
    a = read(path)
    if div > 1:
        a = cv2.resize(a, (a.shape[1] // div, a.shape[0] // div), interpolation=cv2.INTER_AREA)
    return a


def run_image(R, idx, lo, hi, a, methods):
    name = os.path.basename(lo)
    I8, G8 = load_hr(lo, a.hr_div), load_hr(hi, a.hr_div)
    H, W = I8.shape[:2]
    rows, crops = [], {}
    keep = idx in FIG_IMAGES
    # ---- teacher: full-res CIDNet, time + peak memory ----
    gc.collect(); torch.mps.empty_cache()
    base_driver = torch.mps.driver_allocated_memory()
    t0 = time.perf_counter()
    I_hr, O_t, _ = R.enhance(I8)
    t_teacher = time.perf_counter() - t0
    mem_driver = torch.mps.driver_allocated_memory() / 2 ** 30
    mem_cur = torch.mps.current_allocated_memory() / 2 ** 30
    T8 = to_uint8(O_t[None])
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30
    mt = all_metrics(T8, G8)
    rows.append(dict(name=name, scale=1, method='teacher_fullres', t_net_lr=round(t_teacher, 3), t_transfer=0,
                     psnr_gt=mt['psnr'], ssim_gt=mt['ssim'], lpips_gt=mt['lpips'], psnr_teacher=99.0, ssim_teacher=1.0,
                     mps_driver_gb=round(mem_driver, 2), mps_alloc_gb=round(mem_cur, 2), rss_gb=round(rss, 2)))
    if keep:
        crops['input'] = crop(I8, a.hr_div); crops['teacher'] = crop(T8, a.hr_div); crops['gt'] = crop(G8, a.hr_div)
    os.makedirs(CACHE, exist_ok=True)
    cv2.imwrite(f'{CACHE}/teacher_{name[:-4]}.png', cv2.cvtColor(T8, cv2.COLOR_RGB2BGR))
    del O_t
    # ---- tiling ----
    for tsz in a.tiles:
        t0 = time.perf_counter()
        y = run_tiled(R.net, to_tensor(I8), R.dev, tsz, tsz // 4)
        dt = time.perf_counter() - t0
        P8 = to_uint8(y)
        rows.append(dict(name=name, scale=1, method=f'r4_tile{tsz}', t_net_lr=round(dt, 3), t_transfer=0, **metrics(P8, G8, T8)))
        if keep and tsz == 512:
            crops['r4_tile512'] = crop(P8, a.hr_div)
        del y, P8
    # ---- transfer methods ----
    for s in a.scales:
        lr8 = cv2.resize(I8, (W // s, H // s), interpolation=cv2.INTER_AREA)
        t0 = time.perf_counter()
        I_lr, O_lr, O_hvi_lr = R.enhance(lr8)
        t_net = time.perf_counter() - t0
        for m in methods:
            p = P_HVI if m == 'r6_bgu_hvi' else (P_R7 if m == 'r7_bgu_rgb_dn' else P_RGB)
            t0 = time.perf_counter()
            out = transfer.build(m, R, I_hr, I_lr, O_lr, O_hvi_lr, dict(p))
            P8 = to_uint8(out[None])
            t_tr = time.perf_counter() - t0
            rows.append(dict(name=name, scale=s, method=m, t_net_lr=round(t_net, 3), t_transfer=round(t_tr, 3),
                             **metrics(P8, G8, T8)))
            if keep and s == FIG_SCALE and f'{m}_s{s}' in FIG_KEYS:
                crops[f'{m}_s{s}'] = crop(P8, a.hr_div)
            del out, P8
            gc.collect()
    return rows, crops


def read_cache():
    rows = []
    for f in sorted(glob.glob(f'{CACHE}/rows_*.json')):
        rows += json.load(open(f))
    sel = sorted(glob.glob(f'{CACHE}/r7sel_*.json'))
    if sel:  # uhd_r7.py: R7 recomputed with the train-selected params replaces the run-time defaults (kept as *_default)
        for r in rows:
            if r['method'] == 'r7_bgu_rgb_dn':
                r['method'] = 'r7_bgu_rgb_dn_default'
        for f in sel:
            rows += json.load(open(f))
    return rows


def aggregate(cfg):
    rows = read_cache()
    if not rows:
        return
    cols = ['name', 'scale', 'method'] + KEYS + ['mps_driver_gb', 'mps_alloc_gb', 'rss_gb']
    with open('results/uhd_perimg.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, cols, extrasaction='ignore', restval=''); w.writeheader(); w.writerows(rows)
    out = []
    names = sorted({r['name'] for r in rows})
    with open('results/uhd_summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n', 'scale', 'method'] + KEYS + ['mps_driver_gb_mean', 'mps_driver_gb_max', 'config'])
        for s in sorted({r['scale'] for r in rows}):
            for m in dict.fromkeys(r['method'] for r in rows if r['scale'] == s):
                sel = [r for r in rows if r['scale'] == s and r['method'] == m]
                vals = [round(float(np.mean([r[k] for r in sel])), 4) for k in KEYS]
                mem = [r['mps_driver_gb'] for r in sel if r.get('mps_driver_gb', '') != '']
                w.writerow([len(sel), s, m] + vals + [round(float(np.mean(mem)), 2) if mem else '',
                                                      round(float(np.max(mem)), 2) if mem else '', cfg])
                out.append((s, m, len(sel), vals))
    for s, m, n, vals in out:
        print(f'x{s} {m:16s} n={n} ' + ' '.join(f'{k}={v}' for k, v in zip(KEYS, vals)))
    # paired differences R6 - R5 and tile vs teacher
    get = lambda s, m, k: {r['name']: r[k] for r in rows if r['scale'] == s and r['method'] == m}
    for s in sorted({r['scale'] for r in rows if r['method'] == 'r6_bgu_hvi'}):
        for k in ('psnr_teacher', 'ssim_teacher', 'psnr_gt', 'ssim_gt', 'lpips_gt'):
            a6, a5 = get(s, 'r6_bgu_hvi', k), get(s, 'r5_bgu_rgb', k)
            d = np.array([a6[n] - a5[n] for n in a6 if n in a5])
            if len(d) > 1:
                print(f'paired R6-R5 x{s} {k}: mean={d.mean():+.4f} se={d.std(ddof=1) / len(d) ** .5:.4f} '
                      f'wins(R6>R5)={(d > 0).sum()}/{len(d)}')


def make_fig():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ttl = dict(input='input', teacher='CIDNet full-res (teacher)', r1_bicubic_s4='R1 bicubic (x4)',
               r5_bgu_rgb_s4='R5 bilateral RGB (x4)', r4_tile512='R4 tiling 512', gt='ground truth')
    names = sorted(glob.glob(f'{CACHE}/crops_*.npz'))
    assert names, 'no crops cached'
    fig, ax = plt.subplots(len(names), len(FIG_KEYS), figsize=(3.2 * len(FIG_KEYS), 2.1 * len(names)), squeeze=False)
    for i, f in enumerate(names):
        z = np.load(f)
        for j, k in enumerate(FIG_KEYS):
            ax[i, j].imshow(z[k]); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            if i == 0: ax[i, j].set_title(ttl[k], fontsize=8)
        ax[i, 0].set_ylabel(os.path.basename(f)[6:-4], fontsize=7)
    plt.tight_layout(pad=0.3)
    plt.savefig('paper/fig_uhd.png', dpi=200)
    print('wrote paper/fig_uhd.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='mps')
    ap.add_argument('--hr_div', type=int, default=1, help='1 = native 4K; 2 = both low and GT downscaled to 1920x1080')
    ap.add_argument('--scales', type=int, nargs='*', default=[4, 8, 2])
    ap.add_argument('--tiles', type=int, nargs='*', default=[512, 256])
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--aggregate', action='store_true'); ap.add_argument('--fig', action='store_true')
    a = ap.parse_args()
    if a.fig:
        return make_fig()
    methods = ['r1_bicubic', 'r5_bgu_rgb', 'r6_bgu_hvi'] + (['r7_bgu_rgb_dn'] if hasattr(transfer, 'r7_bgu_rgb_dn') else [])
    dev = torch.device(a.device)
    R = transfer.Runner(dev, 'lolv2_real_bestpsnr')
    cfg = (config_str(R.net, 'lolv2_real_bestpsnr', a.device) + f' hr_div={a.hr_div} subset=every5th/150 '
           f'rgb={P_RGB} hvi={P_HVI} r7={P_R7} eps_gain={transfer.EPS_GAIN} tiles={a.tiles} overlap=tile/4')
    os.makedirs(CACHE, exist_ok=True)
    pairs = subset()[:a.limit or None]
    if not a.aggregate:
        print(cfg, flush=True)
        for idx, (lo, hi) in enumerate(pairs):
            name = os.path.basename(lo)
            fn = f'{CACHE}/rows_{name[:-4]}.json'
            if os.path.exists(fn):
                continue
            t0 = time.time()
            for attempt in range(4):  # MPS masked-assignment in PHVIT occasionally raises a shape mismatch: retry
                try:
                    rows, crops = run_image(R, idx, lo, hi, a, methods); break
                except RuntimeError as e:
                    print(f'  retry {attempt + 1} {name}: {str(e)[:90]}', flush=True)
                    gc.collect(); torch.mps.empty_cache()
            else:
                raise RuntimeError(f'{name} failed 4 times')
            if crops:
                np.savez(f'{CACHE}/crops_{name[:-4]}.npz', **crops)
            json.dump(rows, open(fn, 'w'))
            print(f'[{idx + 1}/{len(pairs)}] {name} {time.time() - t0:.0f}s teacher={rows[0]["t_net_lr"]}s '
                  f'mps_driver={rows[0]["mps_driver_gb"]}GB', flush=True)
            gc.collect(); torch.mps.empty_cache()
    aggregate(cfg)


if __name__ == '__main__':
    main()
