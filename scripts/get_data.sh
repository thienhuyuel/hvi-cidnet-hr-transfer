#!/bin/bash
# Fetch datasets into ../Dataset (sibling of this repo), the location expected by common.py.
#   LOL-v2 Real_captured : manual download (Baidu Pan / OneDrive offer no stable direct link)
#   UHD-LL test set      : Google Drive folder, fetched with gdown
# Usage: bash scripts/get_data.sh [lolv2|uhdll|all]
set -e
cd "$(dirname "$0")/.."
DATA="$(cd .. && pwd)/Dataset"
mkdir -p "$DATA"
what="${1:-all}"

lolv2() {
  if [ -d "$DATA/LOL-v2/Real_captured/Train/Low" ] && [ -d "$DATA/LOL-v2/Real_captured/Test/Low" ]; then
    echo "LOL-v2 Real_captured already present: $DATA/LOL-v2/Real_captured"; return
  fi
  cat <<EOF
LOL-v2 must be downloaded manually (links from the original README_ORIGINAL.md):
  Baidu Pan : https://pan.baidu.com/s/17KTa-6GUUW22Q49D5DhhWw?pwd=yixu   (code: yixu)
  One Drive : https://1drv.ms/u/c/2985db836826d183/EYPRJmiD24UggCmCAQAAAAABEbg62rx0FG21FwLQq0jzLg?e=Im12UA
Unpack so that these exist:
  $DATA/LOL-v2/Real_captured/Train/{Low,Normal}   (689 pairs: lowNNNNN.png / normalNNNNN.png)
  $DATA/LOL-v2/Real_captured/Test/{Low,Normal}    (100 pairs)
Then verify with:  bash scripts/get_data.sh check
EOF
}

uhdll() {
  # UHD-LL (Li et al., ICLR 2023, "Embedding Fourier for UHD Low-Light Image Enhancement"), test set.
  # Needed only for the "UHD-LL real-HR evaluation" section.
  command -v gdown >/dev/null || { echo "install gdown first: pip install gdown"; exit 1; }
  gdown --folder "https://drive.google.com/drive/folders/1IneTwBsSiSSVXGoXQ9_hE1cO2d4Fd4DN" -O "$DATA/UHD-LL"
  echo "UHD-LL saved to $DATA/UHD-LL (4K images; large). Check the folder layout before running experiments."
}

check() {
  for s in Train Test; do for d in Low Normal; do
    p="$DATA/LOL-v2/Real_captured/$s/$d"
    printf '%s: %s files\n' "$p" "$(ls "$p" 2>/dev/null | wc -l | tr -d ' ')"
  done; done
  echo "expected: Train 689/689, Test 100/100"
}

case "$what" in
  lolv2) lolv2 ;;
  uhdll) uhdll ;;
  check) check ;;
  all)   lolv2; uhdll ;;
  *) echo "usage: $0 [lolv2|uhdll|check|all]"; exit 1 ;;
esac
