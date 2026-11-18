# Enhance Small, Apply Large: Low-to-High-Resolution Transfer for HVI-CIDNet

Course project (Multimedia, HCMUS K35). Built on [Fediory/HVI-CIDNet](https://github.com/Fediory/HVI-CIDNet)
(HVI: A New Color Space for Low-light Image Enhancement, arXiv 2502.20272); the original README is kept in
[`README_ORIGINAL.md`](README_ORIGINAL.md) (only a note on the new script location was added). Paper: [`paper/main.pdf`](paper/main.pdf) (LaTeX source in `paper/`).

**What the project does.**
1. Reproduces the released HVI-CIDNet LOLv2-real `best_PSNR` checkpoint at full resolution (PSNR / SSIM / LPIPS).
2. Measures CIDNet latency and peak memory versus input resolution on Apple Silicon (MPS).
3. Runs CIDNet once at low resolution and transfers the adjustment to the high-resolution image without any training
   (rungs R1 bicubic, R2 gain-ratio, R3 guided filter, R4 overlapped tiling, R5 bilateral-grid affine fit in RGB,
   R6 the same fit in HVI space, R7 R5 with gain-aware denoising) and compares them with the full-resolution output
   ("teacher") and the ground truth, on LOLv2-real and on 30 real 4K pairs of UHD-LL.

**Main findings.** The bilateral-grid transfer is 3.0 to 3.4 dB closer to the teacher than bicubic upsampling; fitting
in HVI instead of RGB brings no practical gain (at most 0.04 dB on 4K) and is slower; gain-aware denoising lowers
fidelity; overlapping tiling is nearly exact at 600x400 but far from the teacher on real 4K images, where the
low-resolution RGB transfer is the better cheap pipeline. See the paper for full numbers and limitations.

No network is trained; only the released checkpoint is used.

## 1. Environment (macOS, Apple Silicon)

Tested with Python 3.11, PyTorch 2.5.1 (MPS). Pinned versions are in `requirements-macos.txt`
(`requirements.txt` is the upstream file and is left untouched). See also `docs/RUN_MAC.md` (Vietnamese, demo notes).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-macos.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1      # the tuning scripts set this themselves
```

Notes:
- `opencv-contrib-python` is required (not plain `opencv-python`): `study/transfer.py` uses `cv2.ximgproc.guidedFilter`.
- `lpips` fetches the AlexNet backbone from torchvision the first time it runs, so the first run needs internet.
- Scripts default to `--device mps`; pass `--device cpu` to run without a GPU (much slower).

## 2. Checkpoint

The scripts expect `weights/LOLv2_real/best_PSNR.pth` (included in this repository). Original links from `README_ORIGINAL.md`
("LOLv2_real, v2 best PSNR"):

- Baidu Pan: https://pan.baidu.com/s/1PFQ49oa_n7ywTGLl3TUb3A?pwd=yixu (code `yixu`)
- One Drive: https://1drv.ms/u/s!AoPRJmiD24UpgnLTGGGFKmXTWBu4?e=JnzL7z
- All checkpoints: https://pan.baidu.com/s/1rvQcQPwsYbtLIYwB3XgjaA?pwd=yixu (code `yixu`) /
  https://1drv.ms/f/s!AoPRJmiD24UpgloqG-S67l1BX0cG?e=0b4GL0
- Hugging Face copy used by `upstream/eval_hf.py`: https://huggingface.co/Fediory/HVI-CIDNet-LOLv2-real-bestPSNR

Per-checkpoint settings (`gated2=True`, `alpha=0.8` for this checkpoint) follow the official `upstream/eval.py` and are set in `study/common.py`.

## 3. Data

`study/common.py` looks for the dataset next to the repository, at `../Dataset` (i.e. outside this folder):

```
../Dataset/LOL-v2/Real_captured/
    Train/Low/lowNNNNN.png   Train/Normal/normalNNNNN.png    689 pairs
    Test/Low/lowNNNNN.png    Test/Normal/normalNNNNN.png     100 pairs   (about 564 MB in total)
```

A low image `lowNNNNN.png` is paired with `normalNNNNN.png` of the same index (`load_pairs` in `study/common.py`).
`bash scripts/get_data.sh` explains how to download LOLv2 (manual, Baidu/One Drive) and fetches the UHD-LL test set;
`bash scripts/get_data.sh check` verifies the file counts.

`data_manifest/` lists exactly which pairs the experiments use (tab-separated, paths relative to `../Dataset`):

| File | Content |
|------|---------|
| `lolv2_real_train.txt` | 689 train pairs |
| `lolv2_real_test.txt` | 100 test pairs |
| `tune_subset.txt` | 30 train pairs = every 23rd of the sorted train list (`--split train --stride 23`); used only for hyper-parameter selection |

The test split is never used for selecting hyper-parameters.

## 4. Reproducing tables and figures

All commands run from the repository root with the venv active. Every output row records checkpoint, flags,
device and hyper-parameters. Raw logs are already in `results/`.

**Table: reproduction of the checkpoint** (full resolution, `results/reproduce.csv`)
```bash
python -m study.reproduce --set lolv2_real --device mps                 # no GT-mean (main number)
python -m study.reproduce --set lolv2_real --device mps --use_GT_mean   # optional, matches the "GT mean" convention
```

**Hyper-parameter selection** (train subset only; produces `results/transfer_tune*_perimg.csv`, `tune*.log`)
```bash
bash scripts/tune_train.sh      # round 1: guided filter and BGU grids
bash scripts/tune_train2.sh     # round 2: grid extended past the round-1 border
bash scripts/tune_train3.sh     # round 3: separate selection for RGB (R5) and HVI (R6); summary in results/tune3_selection.txt
```

**Table: ablation ladder on LOLv2-real test (100 images)** (`results/transfer_summary.csv`)
```bash
# R1, R2a, R2b, R3, R5, R6 at x2 and x4 (tag test_final)
python -W ignore -m study.transfer --split test --scales 2 4 --gf_r 1 --gf_eps 0.01 --sp 4 --nb 8 --lam 1e-8 --blur 1 --tag test_final

# round-3 parameters, x2/x4/x8 (RGB and HVI selected separately)
python -W ignore -m study.transfer --split test --scales 2 4 8 --methods r1_bicubic r5_bgu_rgb --sp 2 --nb 8 --lam 1e-8  --blur 1 --tag test_r3_rgb
python -W ignore -m study.transfer --split test --scales 2 4 8 --methods r6_bgu_hvi            --sp 2 --nb 8 --lam 1e-10 --blur 1 --tag test_r3_hvi

# R4: overlapped tiling (overlap = tile/4), full resolution only
python -W ignore -m study.transfer --split test --scales --methods r1_bicubic --tiles 128 256 --tag test_tiles
python -W ignore -m study.transfer --split test --scales --methods r1_bicubic --tiles 384 512 --tag test_tiles2

# paired Wilcoxon tests between methods (reads per-image CSV, A minus B)
python -m study.paired_tests results/transfer_test_final_perimg.csv results/paired_tests_test_final.txt r6_bgu_hvi:r5_bgu_rgb
```
Useful switches: `--limit N` (first N pairs), `--stride K` (every K-th pair), `--methods ...`
(`r1_bicubic r2a_gain r2b_gain_gf r3_gf r5_bgu_rgb r6_bgu_hvi`). `python -m study.transfer -h` lists all options.
The flags above were inferred from the `config` column of `results/transfer_summary.csv`; the tiling runs used the
default (unused) transfer parameters.

**Figure/table: latency and peak memory versus resolution** (`results/scaling.csv`)
```bash
python -m study.bench_scaling --input ../Dataset/LOL-v2/Real_captured/Test/Low/low00690.png \
       --device mps --full_long_edge 4032 --denoms 16 8 4 2 1 --out results/scaling.csv
# add --warm for a second (warm) pass: results/scaling_with_warm_pass.csv
```
The 600x400 input is upscaled to a 4032-px long edge (12 MP-class photo) and evaluated at 1/16 ... 1/1 of it;
each resolution runs in a fresh subprocess so peak RSS is a clean per-run value.

**Table/figure: end-to-end cost of pipelines on a large image** (`results/pipeline.csv`, `results/pipeline_r3.csv`)
```bash
python -m study.bench_pipeline --input ../Dataset/LOL-v2/Real_captured/Test/Low/low00690.png --full_long_edge 4032 --device mps --out results/pipeline.csv
# --modes lowres_rgb lowres_hvi ...  to run a subset
```

**All paper tables and figures** (reads `results/*.csv`, writes `paper/tables.tex`, `paper/fig_scaling.pdf`, `paper/fig_pareto.pdf`)
```bash
python -m study.make_report
```

**Qualitative figure** (`paper/fig_qual.png`)
```bash
python -m study.make_qual
```

Paper: `paper/main.tex` includes `paper/tables.tex` and the figures; `paper_overleaf/` is the same set ready to upload to Overleaf.

## 5. R7 gain-aware denoising

R7 = R5 applied to a blend of the input and a guided-filter-denoised copy, weighted by the low-res gain (see paper, Method). Selected on the train subset: `--dn_r 2 --dn_eps 0.016 --g0 2 --g1 8`.

```bash
bash scripts/tune_train4.sh          # 12-config selection on 30 train images -> results/tune_train4.log
bash scripts/run_test_r7.sh          # LOLv2-real test, scales 2 4 8 -> results/transfer_test_r7*_perimg.csv
.venv/bin/python -m study.paired_tests results/transfer_test_r7_all_perimg.csv results/paired_tests_r7.txt \
    r7_bgu_rgb_dn:r5_bgu_rgb r6_bgu_hvi:r5_bgu_rgb r7_bgu_rgb_dn:r6_bgu_hvi
```
`transfer_test_r7_all_perimg.csv` = rows of `transfer_test_r7_perimg.csv` + the `r6_bgu_hvi` rows of `transfer_test_r7_hvi_perimg.csv` (same header).

## 6. UHD-LL real-HR evaluation

Data: UHD-LL `testing_set` (150 pairs, 3840x2160) at `../Dataset/UHD-LL/testing_set/{input,gt}` (`scripts/get_data.sh uhdll`, or download the folder manually if Google Drive quota blocks gdown). Subset: every 5th of the sorted names (`results/uhd_testlist.txt`), 30 pairs. Needs ~20 GB unified memory for the 4K teacher; ~4 min/image on an M1.

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
.venv/bin/python -W ignore -m study.uhd_eval > results/uhd_run.log 2>&1   # resumable (results/uhd_cache/)
.venv/bin/python -W ignore -m study.uhd_r7  > results/uhd_r7.log  2>&1    # R7 with the selected params
.venv/bin/python -W ignore -m study.uhd_eval --aggregate                  # results/uhd_perimg.csv, uhd_summary.csv
.venv/bin/python -m study.uhd_eval --fig                                  # paper/fig_uhd.png
.venv/bin/python -W ignore -m study.paired_tests_uhd                      # results/paired_tests_uhd.txt
.venv/bin/python -m study.make_report                                     # paper/tables.tex (Table 4 = UHD-LL)
```

## 7. Packaging

```bash
bash scripts/make_submission.sh --dry-run --with-data   # list contents, create nothing
bash scripts/make_submission.sh                         # submission/hvi-cidnet-code.zip
bash scripts/make_submission.sh --with-data             # also submission/lolv2-real-captured.zip (about 564 MB)
bash scripts/make_submission.sh --with-weights          # also embed best_PSNR.pth in the code zip
```

## 8. Repository layout

```
study/            project code (Python package; run from the repo root: python -m study.<module>)
  common.py         model loading, metrics, dataset pairs
  transfer.py       transfer methods R1-R7 and the ablation driver
  reproduce.py      checkpoint reproduction (Table 1)
  bench_scaling.py  memory/time vs resolution (Fig. 2)
  bench_pipeline.py end-to-end cost of each pipeline (Table 3)
  uhd_eval.py, uhd_r7.py, uhd_brightness.py   UHD-LL 4K evaluation (Table 4, Fig. 5)
  paired_tests.py, paired_tests_uhd.py        Wilcoxon paired tests
  make_report.py, make_qual.py                tables and figures for the paper
scripts/          shell entry points: tune_train*.sh, run_test_r7.sh, get_data.sh, make_submission.sh, run_demo.command
results/          per-image CSVs, summaries and logs behind every number in the paper
paper/            LaTeX source and the compiled PDF
data_manifest/    exact train/test/tuning image lists
weights/          released LOLv2-real best_PSNR checkpoint
docs/             notes for running the demo on macOS (Vietnamese)

upstream/         original HVI-CIDNet scripts: train, eval, eval_hf, eval_SID_blur, measure*, net_test, app
                  (run as python -m upstream.<script>; app.py and eval_hf.py carry small macOS/CPU fixes)
net/ loss/ data/  original CIDNet model, losses and data loaders (unchanged; imported by study/ and upstream/)
pic/ examples/    original README images and demo example
```
