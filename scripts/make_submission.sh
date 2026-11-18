#!/bin/bash
# Build submission archives in ./submission/ :
#   hvi-cidnet-code.zip        source code + results CSVs + paper sources (always)
#   lolv2-real-captured.zip    ../Dataset/LOL-v2/Real_captured (only with --with-data; ~564 MB)
# Usage: bash scripts/make_submission.sh [--dry-run] [--with-data] [--with-weights]
#   --dry-run       list what would be zipped, create nothing
#   --with-weights  also include weights/LOLv2_real/best_PSNR.pth (small; otherwise download, see README.md)
set -e
cd "$(dirname "$0")/.."
DRY=0; DATA=0; WEIGHTS=0
for a in "$@"; do case "$a" in
  --dry-run) DRY=1 ;; --with-data) DATA=1 ;; --with-weights) WEIGHTS=1 ;;
  *) echo "unknown option $a"; exit 1 ;; esac; done

OUT=submission; mkdir -p "$OUT"
EXCL=("results/uhd_cache/*"  '.venv/*' '*/__pycache__/*' '__pycache__/*' '*.pyc' '*.DS_Store' '.git/*' '.claude/*'
       'submission/*' 'output_hf/*' 'paper_overleaf.zip' 'pic/*' '*.pth' '*.pt' '*.ckpt' '*.safetensors'
       '.cache/*' '*/.ipynb_checkpoints/*' '*.log.tmp' )
if [ "$DRY" = 1 ]; then
  echo "== would create $OUT/hvi-cidnet-code.zip with:"
  zip -r -sf "$OUT/.dry.zip" . -x "${EXCL[@]}" | grep -v '/$' | grep -v 'Scanning files' | sed 's/^ *//' | grep -v '^Total'
  [ "$WEIGHTS" = 1 ] && echo weights/LOLv2_real/best_PSNR.pth
  [ "$DATA" = 1 ] && echo "== would create $OUT/lolv2-real-captured.zip: $(find ../Dataset/LOL-v2/Real_captured -name '*.png' | wc -l | tr -d ' ') png files, $(du -sh ../Dataset/LOL-v2/Real_captured | cut -f1)"
  rmdir "$OUT" 2>/dev/null || true
  exit 0
fi

rm -f "$OUT/hvi-cidnet-code.zip"
zip -r -q "$OUT/hvi-cidnet-code.zip" . -x "${EXCL[@]}"
[ "$WEIGHTS" = 1 ] && zip -q "$OUT/hvi-cidnet-code.zip" weights/LOLv2_real/best_PSNR.pth
echo "created $OUT/hvi-cidnet-code.zip ($(du -h "$OUT/hvi-cidnet-code.zip" | cut -f1))"

if [ "$DATA" = 1 ]; then
  rm -f "$OUT/lolv2-real-captured.zip"
  ( cd ../Dataset && zip -r -q "$(cd - >/dev/null; pwd)/$OUT/lolv2-real-captured.zip" LOL-v2/Real_captured -x '*.DS_Store' )
  echo "created $OUT/lolv2-real-captured.zip ($(du -h "$OUT/lolv2-real-captured.zip" | cut -f1))"
fi
