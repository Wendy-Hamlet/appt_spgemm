#!/bin/bash
# Build for sm_80 (A100) even on H100 so results port. Uses CUDA 12.8 toolkit.
set -e
CUDA=${CUDA_HOME:-/usr/local/cuda-12.8}
HERE=$(dirname "$(readlink -f "$0")")
"$CUDA/bin/nvcc" -O3 -arch=sm_80 -std=c++17 \
  -I"$CUDA/include" -L"$CUDA/lib64" \
  "$HERE/cusparse_baseline.cu" -o "$HERE/cusparse_baseline" -lcusparse
echo "built $HERE/cusparse_baseline"
