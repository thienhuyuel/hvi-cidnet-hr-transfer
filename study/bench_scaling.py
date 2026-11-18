"""Measure CIDNet latency and peak memory vs. input resolution.

Driver:  python bench_scaling.py --input IMG --device mps --full_long_edge 4032
Each (scale) runs in a fresh subprocess so ru_maxrss is a clean per-run peak.
Results are appended to results/scaling.csv (one row per run, config included).
"""
import argparse, csv, json, os, platform, resource, subprocess, sys, time

W = 'weights/LOLv2_real/best_PSNR.pth'
FIELDS = ['timestamp', 'device', 'weights', 'input', 'src_size', 'full_long_edge',
          'denom', 'h', 'w', 'mpix', 'status', 'time_s', 'time_warm_s',
          'peak_rss_gb', 'mps_driver_gb', 'mps_alloc_gb', 'error']


def run_single(a):
    import numpy as np, torch, torch.nn.functional as F
    from PIL import Image
    from net.CIDNet import CIDNet
    dev = torch.device(a.device)
    img = Image.open(a.input).convert('RGB')
    if a.full_long_edge:
        s = a.full_long_edge / max(img.size)
        img = img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC)
    if a.denom > 1:
        img = img.resize((max(8, img.width // a.denom), max(8, img.height // a.denom)), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().div(255)[None]
    h, w = x.shape[2:]
    ph, pw = (-h) % 8, (-w) % 8
    x = F.pad(x, (0, pw, 0, ph), 'reflect')
    from common import load_model
    net = load_model(dev)  # official LOLv2_real best_PSNR config (gated2, alpha=0.8)
    sync = (lambda: torch.mps.synchronize()) if dev.type == 'mps' else (lambda: None)
    row = dict(h=h, w=w, mpix=round(h * w / 1e6, 3), status='ok', error='')
    try:
        with torch.no_grad():
            xd = x.to(dev)
            t = time.perf_counter(); y = net(xd); sync(); row['time_s'] = round(time.perf_counter() - t, 3)
            if a.warm:
                t = time.perf_counter(); y = net(xd); sync(); row['time_warm_s'] = round(time.perf_counter() - t, 3)
            y.float().cpu()
    except Exception as e:  # OOM / unsupported op: record, do not hide
        row.update(status='fail', error=repr(e)[:300])
    div = 1 << 30
    row['peak_rss_gb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / div, 3)  # bytes on macOS
    if dev.type == 'mps':
        row['mps_driver_gb'] = round(torch.mps.driver_allocated_memory() / div, 3)
        row['mps_alloc_gb'] = round(torch.mps.current_allocated_memory() / div, 3)
    print('ROW ' + json.dumps(row), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--device', default='mps', choices=['mps', 'cpu'])
    p.add_argument('--weights', default=W)
    p.add_argument('--full_long_edge', type=int, default=0,
                   help='resize source to this long edge first (synthetic large image; logged)')
    p.add_argument('--denoms', type=int, nargs='+', default=[8, 4, 2, 1])
    p.add_argument('--warm', action='store_true', help='also time a 2nd (warm) pass')
    p.add_argument('--timeout', type=int, default=900)
    p.add_argument('--out', default='results/scaling.csv')
    p.add_argument('--denom', type=int, help=argparse.SUPPRESS)
    p.add_argument('--single', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.single:
        return run_single(a)

    from PIL import Image
    src = Image.open(a.input).size
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    new = not os.path.exists(a.out)
    env = dict(os.environ)
    print(f'host={platform.platform()} torch device={a.device} '
          f'MPS_FALLBACK={env.get("PYTORCH_ENABLE_MPS_FALLBACK")}')
    with open(a.out, 'a', newline='') as f:
        wr = csv.DictWriter(f, FIELDS)
        if new:
            wr.writeheader()
        for d in a.denoms:
            cmd = [sys.executable, '-m', 'study.bench_scaling', '--single', '--input', a.input, '--device', a.device,
                   '--weights', a.weights, '--full_long_edge', str(a.full_long_edge), '--denom', str(d)]
            if a.warm:
                cmd.append('--warm')
            row = dict(timestamp=time.strftime('%F %T'), device=a.device, weights=a.weights,
                       input=a.input, src_size=f'{src[0]}x{src[1]}', full_long_edge=a.full_long_edge, denom=d)
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout, env=env)
                line = [l for l in r.stdout.splitlines() if l.startswith('ROW ')]
                if line:
                    row.update(json.loads(line[-1][4:]))
                else:
                    row.update(status='crash', error=(r.stderr.strip().splitlines() or ['no output'])[-1][:300])
            except subprocess.TimeoutExpired:
                row.update(status='timeout', error=f'>{a.timeout}s')
            wr.writerow(row); f.flush()
            print({k: row.get(k) for k in ('denom', 'h', 'w', 'status', 'time_s', 'peak_rss_gb', 'mps_driver_gb', 'error')}, flush=True)


if __name__ == '__main__':
    main()
