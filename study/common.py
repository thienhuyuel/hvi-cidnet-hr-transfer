"""Shared helpers: model loading with the official per-checkpoint config, and the
metrics of measure.py (PSNR / MATLAB-style SSIM on uint8, LPIPS-alex)."""
import glob, os
import cv2, numpy as np, torch, torch.nn.functional as F
from PIL import Image
from net.CIDNet import CIDNet

DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Dataset'))
SETS = {  # name -> (low dir, GT dir)
    'lolv2_real': (f'{DATA}/LOL-v2/Real_captured/Test/Low', f'{DATA}/LOL-v2/Real_captured/Test/Normal'),
    'lolv1': (f'{DATA}/LOLv1/eval15/low', f'{DATA}/LOLv1/eval15/high'),
}
# eval.py: LOLv2_real best_PSNR -> gated2=True, alpha=0.8 (gated stays False)
CKPT = {'lolv2_real_bestpsnr': dict(path='weights/LOLv2_real/best_PSNR.pth', gated=False, gated2=True, alpha=0.8)}


def load_model(device, ckpt='lolv2_real_bestpsnr', gated=None, gated2=None, alpha=None):
    c = CKPT[ckpt]
    net = CIDNet().to(device)
    net.load_state_dict(torch.load(c['path'], map_location='cpu', weights_only=True))
    net.eval()
    net.trans.gated = c['gated'] if gated is None else gated
    net.trans.gated2 = c['gated2'] if gated2 is None else gated2
    net.trans.alpha = c['alpha'] if alpha is None else alpha
    return net


def config_str(net, ckpt, device, use_gt_mean=False):
    t = net.trans
    return (f'ckpt={ckpt} gated={t.gated} gated2={t.gated2} alpha={t.alpha} alpha_s={t.alpha_s} '
            f'gamma=1.0 use_GT_mean={use_gt_mean} device={device}')


def load_pairs(name):
    lo, hi = SETS[name]
    # LOLv2 names GT normalNNNNN.png for lowNNNNN.png (same index); LOLv1 uses identical names
    pairs = [(p, os.path.join(hi, os.path.basename(p).replace('low', 'normal', 1)
                              if name.startswith('lolv2') else os.path.basename(p)))
             for p in sorted(glob.glob(lo + '/*.png'))]
    assert all(os.path.exists(g) for _, g in pairs), 'missing GT for some inputs'
    return pairs


def read(p):
    return np.asarray(Image.open(p).convert('RGB'))


def to_tensor(a):  # HWC uint8 -> 1CHW float
    return torch.from_numpy(a).permute(2, 0, 1).float().div(255)[None]


def to_uint8(t):  # 1CHW float -> HWC uint8 (clamp+round, like ToPILImage)
    return (t.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)


@torch.no_grad()
def run_cidnet(net, x, device):
    """x: 1CHW float in [0,1] on CPU. Pads to multiple of 8 (reflect), crops back."""
    h, w = x.shape[2:]
    ph, pw = (-h) % 8, (-w) % 8
    xp = F.pad(x, (0, pw, 0, ph), 'reflect') if (ph or pw) else x
    y = net(xp.to(device))
    if device.type == 'mps':
        torch.mps.synchronize()
    return y[:, :, :h, :w].float().cpu()


def psnr(a, b):
    d = a.astype(np.float32) - b.astype(np.float32)
    return 10.0 * np.log10(255.0 * 255.0 / (np.mean(d * d) + 1e-8))


def _ssim1(i1, i2):
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    i1, i2 = i1.astype(np.float64), i2.astype(np.float64)
    k = cv2.getGaussianKernel(11, 1.5); win = np.outer(k, k.T)
    f = lambda z: cv2.filter2D(z, -1, win)[5:-5, 5:-5]
    m1, m2 = f(i1), f(i2)
    s1, s2, s12 = f(i1 ** 2) - m1 ** 2, f(i2 ** 2) - m2 ** 2, f(i1 * i2) - m1 * m2
    return (((2 * m1 * m2 + C1) * (2 * s12 + C2)) / ((m1 ** 2 + m2 ** 2 + C1) * (s1 + s2 + C2))).mean()


def ssim(a, b):
    return float(np.mean([_ssim1(a[..., c], b[..., c]) for c in range(3)]))


_lp = None
def lpips_fn():
    global _lp
    if _lp is None:
        import lpips
        _lp = lpips.LPIPS(net='alex', verbose=False)
    return _lp


@torch.no_grad()
def lpips_score(a, b):
    import lpips
    return float(lpips_fn()(lpips.im2tensor(b), lpips.im2tensor(a)).item())  # (ref, pred) as measure.py


def gt_mean_fix(pred, gt):
    mp = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY).mean()
    mt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY).mean()
    return np.clip(pred * (mt / mp), 0, 255)  # float, as in measure.py


def all_metrics(pred, gt, use_gt_mean=False):
    if use_gt_mean:
        pred = gt_mean_fix(pred, gt)
    return dict(psnr=float(psnr(pred, gt)), ssim=ssim(pred, gt), lpips=lpips_score(pred, gt))


@torch.no_grad()
def run_tiled(net, x, device, tile=256, overlap=64):
    """Overlapping-tile CIDNet inference with linear-ramp feather blending. x: 1CHW float CPU."""
    _, C, H, W = x.shape
    stride = tile - overlap
    ys = list(range(0, max(H - tile, 0) + stride, stride)) if H > tile else [0]
    xs = list(range(0, max(W - tile, 0) + stride, stride)) if W > tile else [0]
    acc = torch.zeros(1, C, H, W); wsum = torch.zeros(1, 1, H, W)

    def ramp(n, lo, hi):  # lo/hi: feather at this edge?
        r = torch.ones(n)
        k = min(overlap, n // 2)
        if lo: r[:k] = torch.linspace(1 / (k + 1), 1, k)
        if hi: r[n - k:] = torch.linspace(1, 1 / (k + 1), k)
        return r
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            y0c, x0c = max(0, y1 - tile), max(0, x1 - tile)  # keep full tile size at borders
            out = run_cidnet(net, x[:, :, y0c:y1, x0c:x1], device)
            h, w = y1 - y0c, x1 - x0c
            wt = (ramp(h, y0c > 0, y1 < H)[:, None] * ramp(w, x0c > 0, x1 < W)[None, :])[None, None]
            acc[:, :, y0c:y1, x0c:x1] += out * wt
            wsum[:, :, y0c:y1, x0c:x1] += wt
    return acc / wsum
