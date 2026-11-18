"""End-to-end time / peak memory of full-res, tiled and low-res+transfer pipelines on a large image.
Each configuration runs in a fresh subprocess (clean peak RSS).  Results -> results/pipeline.csv
python bench_pipeline.py --input IMG --full_long_edge 4032
"""
import argparse, csv, json, os, resource, subprocess, sys, time

FIELDS = ['timestamp', 'input', 'h', 'w', 'mpix', 'device', 'mode', 'param', 'status', 'time_net_s', 'time_transfer_s',
          'time_total_s', 'peak_rss_gb', 'mps_driver_gb', 'error']


def single(a):
    import cv2, numpy as np, torch
    from PIL import Image
    from common import load_model, to_tensor, to_uint8, run_cidnet, run_tiled
    import transfer
    dev = torch.device(a.device)
    img = Image.open(a.input).convert('RGB')
    s = a.full_long_edge / max(img.size)
    img = img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC)
    I8 = np.asarray(img).copy()
    H, W = I8.shape[:2]
    row = dict(h=H, w=W, mpix=round(H * W / 1e6, 2), status='ok', error='')
    try:
        if a.mode == 'full':
            R = transfer.Runner(dev, 'lolv2_real_bestpsnr')
            t = time.perf_counter(); R.enhance(I8); row['time_net_s'] = round(time.perf_counter() - t, 2)
        elif a.mode == 'tile':
            net = load_model(dev)
            t = time.perf_counter(); run_tiled(net, to_tensor(I8), dev, a.param, a.param // 4)
            row['time_net_s'] = round(time.perf_counter() - t, 2)
        else:  # lowres_rgb / lowres_hvi, param = downscale denominator
            R = transfer.Runner(dev, 'lolv2_real_bestpsnr')
            p = dict(gf_r=1, gf_eps=0.01, sp=2, nb=8, blur=1, lam=1e-8 if a.mode == 'lowres_rgb' else 1e-10)  # round-3 selection
            t0 = time.perf_counter()
            lr = cv2.resize(I8, (W // a.param, H // a.param), interpolation=cv2.INTER_AREA)
            I_lr, O_lr, Oh = R.enhance(lr)
            I_hr = to_tensor(I8)[0]
            t1 = time.perf_counter()
            out = transfer.build('r5_bgu_rgb' if a.mode == 'lowres_rgb' else 'r6_bgu_hvi', R, I_hr, I_lr, O_lr, Oh, p)
            to_uint8(out[None])
            t2 = time.perf_counter()
            row.update(time_net_s=round(t1 - t0, 2), time_transfer_s=round(t2 - t1, 2))
        row['time_total_s'] = round(time.perf_counter() - t0, 2) if a.mode.startswith('lowres') else row['time_net_s']
    except Exception as e:
        row.update(status='fail', error=repr(e)[:300])
    row['peak_rss_gb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, 2)
    if dev.type == 'mps':
        row['mps_driver_gb'] = round(torch.mps.driver_allocated_memory() / 2 ** 30, 2)
    print('ROW ' + json.dumps(row), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True); p.add_argument('--full_long_edge', type=int, default=4032)
    p.add_argument('--device', default='mps'); p.add_argument('--timeout', type=int, default=900)
    p.add_argument('--out', default='results/pipeline.csv')
    p.add_argument('--modes', nargs='*', default=None, help='subset of modes to run')
    p.add_argument('--mode', help=argparse.SUPPRESS); p.add_argument('--param', type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument('--single', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.single:
        return single(a)
    cfgs = [('lowres_rgb', 8), ('lowres_hvi', 8), ('lowres_rgb', 4), ('lowres_hvi', 4), ('lowres_rgb', 2), ('lowres_hvi', 2),
            ('tile', 256), ('tile', 512), ('full', 0)]
    new = not os.path.exists(a.out)
    with open(a.out, 'a', newline='') as f:
        w = csv.DictWriter(f, FIELDS)
        if new: w.writeheader()
        for mode, par in [c for c in cfgs if not a.modes or c[0] in a.modes]:
            row = dict(timestamp=time.strftime('%F %T'), input=a.input, device=a.device, mode=mode, param=par)
            cmd = [sys.executable, '-W', 'ignore', '-m', 'study.bench_pipeline', '--single', '--input', a.input, '--full_long_edge', str(a.full_long_edge),
                   '--device', a.device, '--mode', mode, '--param', str(par)]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
                ln = [l for l in r.stdout.splitlines() if l.startswith('ROW ')]
                row.update(json.loads(ln[-1][4:]) if ln else dict(status='crash', error=(r.stderr.strip().splitlines() or ['?'])[-1][:300]))
            except subprocess.TimeoutExpired:
                row.update(status='timeout', error=f'>{a.timeout}s')
            w.writerow(row); f.flush(); print(row, flush=True)


if __name__ == '__main__':
    main()
