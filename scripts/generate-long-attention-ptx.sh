#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
"${NVCC:-/usr/local/cuda/bin/nvcc}" -ptx -arch=sm_86 -O3 --use_fast_math "$root/scripts/kernels/attention_long.cu" -o "$tmp/long.ptx"
{
 printf 'package ptx\n\n// LongAttentionPTX is generated from scripts/kernels/attention_long.cu.\n// Regenerate with scripts/generate-long-attention-ptx.sh (CUDA 12.8, sm_86).\nconst LongAttentionPTX = `'
 cat "$tmp/long.ptx"
 printf '`\n'
} > "$root/backends/nvidia/ptx/attention_long_generated.go"
gofmt -w "$root/backends/nvidia/ptx/attention_long_generated.go"
