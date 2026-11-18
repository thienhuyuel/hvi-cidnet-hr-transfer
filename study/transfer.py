"""Low-res -> high-res transfer of a CIDNet adjustment (ablation ladder, rungs 1-3, 5, 6, 7).

Per image:  I_hr (LOL, 600x400, treated as 'HR')  ->  I_lr = INTER_AREA downscale by s.
  teacher = CIDNet(I_hr);  O_lr = CIDNet(I_lr);  each method builds O_hr from (I_hr, I_lr, O_lr).
Methods:  r1_bicubic | r2a_gain | r2b_gain_gf | r3_gf | r5_bgu_rgb | r6_bgu_hvi
Metrics (uint8, measure.py protocol): vs teacher (PSNR/SSIM) and vs GT (PSNR/SSIM/LPIPS).
Hyper-parameters are chosen on --split train subsets, never on test.
"""
import argparse, csv, math, os, time
import cv2, numpy as np, torch, torch.nn.functional as F
from study.common import *

EPS_GAIN = 0.02
METHODS = ['r1_bicubic', 'r2a_gain', 'r2b_gain_gf', 'r3_gf', 'r5_bgu_rgb', 'r6_bgu_hvi']


# ---------- helpers ----------
def up(x, hw):  # CHW float tensor -> bicubic to (H,W)
    return F.interpolate(x[None], size=hw, mode='bicubic', align_corners=False)[0]


def gf(guide_hwc, src_chw, r, eps):  # cv2 guided filter, guide HxWx3 float32, src CHW tensor
    s = src_chw.permute(1, 2, 0).numpy().astype(np.float32)
    o = cv2.ximgproc.guidedFilter(guide_hwc.astype(np.float32), s, r, eps)
    return torch.from_numpy(o).permute(2, 0, 1)


def log_guide(i, floor=1e-3):  # intensity (H,W) in [0,1] -> log-scaled [0,1]
    return (torch.log(i.clamp(0, 1) + floor) - math.log(floor)) / (math.log(1 + floor) - math.log(floor))


# ---------- bilateral-grid affine (own implementation of the BGU idea, Chen et al. 2016) ----------
def bgu_fit_apply(src_lr, dst_lr, g_lr, src_hr, g_hr, sp=16, nb=8, lam=1e-4, blur=1):
    """Fit per-cell 3x4 affine src->dst on a bilateral grid (x, y, guide), slice at HR, apply to src_hr."""
    dt = torch.float64
    C, H, W = src_lr.shape
    GH, GW = (H - 1) // sp + 2, (W - 1) // sp + 2
    ys, xs = torch.meshgrid(torch.arange(H, dtype=dt), torch.arange(W, dtype=dt), indexing='ij')
    gx, gy, gz = (xs / sp).flatten(), (ys / sp).flatten(), (g_lr.to(dt).flatten().clamp(0, 1) * (nb - 1))
    v = torch.cat([src_lr.reshape(C, -1).to(dt), torch.ones(1, H * W, dtype=dt)], 0)  # 4,N
    d = dst_lr.reshape(C, -1).to(dt)                                                    # 3,N
    feat = torch.cat([(v[:, None] * v[None]).reshape(16, -1), (d[:, None] * v[None]).reshape(12, -1)], 0)  # 28,N
    feat = torch.cat([feat, torch.ones(1, H * W, dtype=dt)], 0)                          # + count
    grid = torch.zeros(29, nb * GH * GW, dtype=dt)
    x0, y0, z0 = gx.floor(), gy.floor(), gz.floor()
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                w = ((1 - (gx - x0)) if dx == 0 else (gx - x0)) * ((1 - (gy - y0)) if dy == 0 else (gy - y0)) \
                    * ((1 - (gz - z0)) if dz == 0 else (gz - z0))
                xi = (x0 + dx).long().clamp(0, GW - 1); yi = (y0 + dy).long().clamp(0, GH - 1)
                zi = (z0 + dz).long().clamp(0, nb - 1)
                grid.index_add_(1, (zi * GH + yi) * GW + xi, feat * w)
    g5 = grid.reshape(1, 29, nb, GH, GW)
    k = torch.tensor([1., 2., 1.], dtype=dt) / 4
    for _ in range(blur):  # separable [1 2 1] smoothing in x, y, z
        for ax in range(3):
            shp = [1, 1, 1]; shp[ax] = 3
            kk = k.reshape(1, 1, *shp).repeat(29, 1, 1, 1, 1)
            pad = [0, 0, 0, 0, 0, 0]; pad[2 * (2 - ax):2 * (2 - ax) + 2] = [1, 1]
            g5 = F.conv3d(F.pad(g5, pad, mode='replicate'), kk, groups=29)
    grid = g5.reshape(29, -1).T  # M,29
    A = grid[:, :16].reshape(-1, 4, 4); Bm = grid[:, 16:28].reshape(-1, 3, 4); cnt = grid[:, 28]
    # global fit as prior for sparse cells
    Ag, Bg = A.sum(0), Bm.sum(0)
    Xg = torch.linalg.solve(Ag + 1e-9 * torch.eye(4, dtype=dt), Bg.T).T  # 3,4
    mu = (lam * cnt + 1e-8)[:, None, None]
    Ar = A + mu * torch.eye(4, dtype=dt)
    Br = Bm + mu * Xg[None]
    X = torch.linalg.solve(Ar, Br.transpose(1, 2)).transpose(1, 2)  # M,3,4
    coef = X.reshape(nb, GH, GW, 12).permute(3, 0, 1, 2)[None].float()  # 1,12,nb,GH,GW
    # slice at HR
    Hh, Wh = src_hr.shape[1:]
    sh, sw = Hh / H, Wh / W
    yh = ((torch.arange(Hh).float() + 0.5) / sh - 0.5).clamp(0, H - 1) / sp
    xh = ((torch.arange(Wh).float() + 0.5) / sw - 0.5).clamp(0, W - 1) / sp
    zz = g_hr.clamp(0, 1) * (nb - 1)
    nx = 2 * xh[None, :].expand(Hh, Wh) / (GW - 1) - 1
    ny = 2 * yh[:, None].expand(Hh, Wh) / (GH - 1) - 1
    nz = 2 * zz / (nb - 1) - 1
    samp = torch.stack([nx, ny, nz], -1)[None, None]  # 1,1,Hh,Wh,3
    m = F.grid_sample(coef, samp, mode='bilinear', padding_mode='border', align_corners=True)[0, :, 0]  # 12,Hh,Wh
    m = m.reshape(3, 4, Hh, Wh)
    out = (m[:, :3] * src_hr[None]).sum(1) + m[:, 3]
    return out


# ---------- pipeline ----------
class Runner:
    def __init__(self, device, ckpt):
        self.dev = device
        self.net = load_model(device, ckpt)
        self.cap = {}
        orig = self.net.trans.PHVIT
        def hook(h):  # capture CIDNet's output in HVI space (before HVI->RGB)
            self.cap['hvi'] = h.detach().float().cpu()
            return orig(h)
        self.net.trans.PHVIT = hook

    def enhance(self, a_uint8):
        x = to_tensor(a_uint8)
        y = run_cidnet(self.net, x, self.dev)
        h, w = x.shape[2:]
        return x[0], y[0], self.cap['hvi'][0, :, :h, :w]

    def hvit(self, x):
        with torch.no_grad():
            return self.net.trans.HVIT(x[None].to(self.dev)).float().cpu()[0]

    def phvit(self, h):
        with torch.no_grad():
            return self.net.trans.PHVIT(h[None].to(self.dev)).float().cpu()[0]


def build(method, R, I_hr, I_lr, O_lr, O_hvi_lr, p):
    hw = I_hr.shape[1:]
    if method == 'r1_bicubic':
        return up(O_lr, hw)
    if method in ('r2a_gain', 'r2b_gain_gf'):
        g = (O_lr + EPS_GAIN) / (I_lr + EPS_GAIN)
        gu = up(g, hw)
        if method == 'r2b_gain_gf':
            gu = gf(log_guide(I_hr).permute(1, 2, 0).numpy(), gu, p['gf_r'], p['gf_eps'])
        return I_hr * gu
    if method == 'r3_gf':
        return gf(log_guide(I_hr).permute(1, 2, 0).numpy(), up(O_lr, hw), p['gf_r'], p['gf_eps'])
    if method == 'r5_bgu_rgb':
        gl, gh = log_guide(I_lr.max(0)[0]), log_guide(I_hr.max(0)[0])
        return bgu_fit_apply(I_lr, O_lr, gl, I_hr, gh, p['sp'], p['nb'], p['lam'], p['blur'])
    if method == 'r6_bgu_hvi':
        Hl, Hh = R.hvit(I_lr), R.hvit(I_hr)  # note: HVIT(I_hr) last -> density_k identical anyway
        gl, gh = log_guide(Hl[2]), log_guide(Hh[2])
        oh = bgu_fit_apply(Hl, O_hvi_lr, gl, Hh, gh, p['sp'], p['nb'], p['lam'], p['blur'])
        return R.phvit(oh)
    if method == 'r7_bgu_rgb_dn':
        return r7_bgu_rgb_dn(I_hr, I_lr, O_lr, p)
    raise ValueError(method)


def r7_bgu_rgb_dn(I_hr, I_lr, O_lr, p):
    """R7 gain-aware denoising (training-free). R5 with the affine applied to a blend of I_hr and a denoised copy.
    D_hr = self-guided guided filter of I_hr (radius dn_r, eps dn_eps); local gain G = (Y(O_lr)+e)/(Y(I_lr)+e)
    (Y = channel mean, e=EPS_GAIN) upsampled bicubic; w = clip((G-g0)/(g1-g0),0,1); I_in = (1-w) I_hr + w D_hr;
    out = BGU affine (fit at low-res exactly as R5) applied to I_in."""
    hw = I_hr.shape[1:]
    ihwc = I_hr.permute(1, 2, 0).numpy().astype(np.float32)
    D = torch.from_numpy(cv2.ximgproc.guidedFilter(ihwc, ihwc, p['dn_r'], p['dn_eps'])).permute(2, 0, 1)
    G = up(((O_lr.mean(0) + EPS_GAIN) / (I_lr.mean(0) + EPS_GAIN))[None], hw)[0]
    w = ((G - p['g0']) / (p['g1'] - p['g0'])).clamp(0, 1)
    I_in = (1 - w) * I_hr + w * D
    gl, gh = log_guide(I_lr.max(0)[0]), log_guide(I_hr.max(0)[0])  # guide stays the ORIGINAL I_hr (as R5)
    return bgu_fit_apply(I_lr, O_lr, gl, I_in, gh, p['sp'], p['nb'], p['lam'], p['blur'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', default='lolv2_real')
    ap.add_argument('--split', default='test', choices=['test', 'train'])
    ap.add_argument('--ckpt', default='lolv2_real_bestpsnr')
    ap.add_argument('--device', default='mps')
    ap.add_argument('--scales', type=int, nargs='*', default=[2, 4])
    ap.add_argument('--tiles', type=int, nargs='*', default=[], help='tile sizes for rung 4 (overlap = tile/4)')
    ap.add_argument('--methods', nargs='+', default=METHODS)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--stride', type=int, default=1, help='take every k-th pair (train subsets)')
    ap.add_argument('--gf_r', type=int, default=8); ap.add_argument('--gf_eps', type=float, default=1e-3)
    ap.add_argument('--sp', type=int, default=16); ap.add_argument('--nb', type=int, default=8)
    ap.add_argument('--lam', type=float, default=1e-4); ap.add_argument('--blur', type=int, default=1)
    ap.add_argument('--dn_r', type=int, default=4); ap.add_argument('--dn_eps', type=float, default=4e-3)  # R7
    ap.add_argument('--g0', type=float, default=2.0); ap.add_argument('--g1', type=float, default=8.0)      # R7
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    p = dict(gf_r=a.gf_r, gf_eps=a.gf_eps, sp=a.sp, nb=a.nb, lam=a.lam, blur=a.blur)

    if 'r7_bgu_rgb_dn' in a.methods:
        p.update(dn_r=a.dn_r, dn_eps=a.dn_eps, g0=a.g0, g1=a.g1)
    dev = torch.device(a.device)
    R = Runner(dev, a.ckpt)
    name = a.set
    if a.split == 'train':
        d = DATA + '/LOL-v2/Real_captured/Train'
        SETS['lolv2_real_train'] = (d + '/Low', d + '/Normal'); name = 'lolv2_real_train'
    pairs = load_pairs(name)[::a.stride][:a.limit or None]
    cfg = config_str(R.net, a.ckpt, a.device) + f' eps_gain={EPS_GAIN} ' + ' '.join(f'{k}={v}' for k, v in p.items())
    os.makedirs('results', exist_ok=True)
    rows = []
    for lo, hi in pairs:
        I8, G8 = read(lo), read(hi)
        I_hr, O_hr_t, _ = R.enhance(I8)
        T8 = to_uint8(O_hr_t[None])
        rec = dict(name=os.path.basename(lo))
        for tsz in a.tiles:
            t0 = time.perf_counter()
            y = run_tiled(R.net, to_tensor(I8), dev, tsz, tsz // 4)
            dt = time.perf_counter() - t0
            P8 = to_uint8(y); mg = all_metrics(P8, G8)
            rows.append(dict(name=rec['name'], scale=1, method=f'r4_tile{tsz}', t_net_lr=round(dt, 4), t_transfer=0,
                             psnr_gt=mg['psnr'], ssim_gt=mg['ssim'], lpips_gt=mg['lpips'],
                             psnr_teacher=float(psnr(P8, T8)), ssim_teacher=ssim(P8, T8)))
        for s in a.scales:
            H, W = I8.shape[:2]
            lr8 = cv2.resize(I8, (W // s, H // s), interpolation=cv2.INTER_AREA)
            t0 = time.perf_counter()
            I_lr, O_lr, O_hvi_lr = R.enhance(lr8)
            t_net = time.perf_counter() - t0
            for m in a.methods:
                t0 = time.perf_counter()
                out = build(m, R, I_hr, I_lr, O_lr, O_hvi_lr, p)
                t_tr = time.perf_counter() - t0
                P8 = to_uint8(out[None])
                mg = all_metrics(P8, G8)
                rows.append(dict(name=rec['name'], scale=s, method=m, t_net_lr=round(t_net, 4), t_transfer=round(t_tr, 4),
                                 psnr_gt=mg['psnr'], ssim_gt=mg['ssim'], lpips_gt=mg['lpips'],
                                 psnr_teacher=float(psnr(P8, T8)), ssim_teacher=ssim(P8, T8)))
        mt = all_metrics(T8, G8)
        rows.append(dict(name=rec['name'], scale=1, method='teacher_fullres', t_net_lr=0, t_transfer=0,
                         psnr_gt=mt['psnr'], ssim_gt=mt['ssim'], lpips_gt=mt['lpips'], psnr_teacher=99.0, ssim_teacher=1.0))
        print(rec['name'], flush=True)

    tag = a.tag or f'{name}'
    per = f'results/transfer_{tag}_perimg.csv'
    with open(per, 'w', newline='') as f:
        w = csv.DictWriter(f, rows[0].keys()); w.writeheader(); w.writerows(rows)
    keys = ['t_net_lr', 't_transfer', 'psnr_gt', 'ssim_gt', 'lpips_gt', 'psnr_teacher', 'ssim_teacher']
    out_csv = 'results/transfer_summary.csv'
    new = not os.path.exists(out_csv)
    with open(out_csv, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['timestamp', 'tag', 'split', 'n', 'scale', 'method'] + keys + ['config'])
        for s in sorted({r['scale'] for r in rows}):
            for m in dict.fromkeys(r['method'] for r in rows if r['scale'] == s):
                sel = [r for r in rows if r['scale'] == s and r['method'] == m]
                vals = [round(float(np.mean([r[k] for r in sel])), 4) for k in keys]
                w.writerow([time.strftime('%F %T'), tag, a.split, len(sel), s, m] + vals + [cfg])
                print(f'x{s} {m:16s} ' + ' '.join(f'{k}={v}' for k, v in zip(keys, vals)))
    print(cfg)


if __name__ == '__main__':
    main()
