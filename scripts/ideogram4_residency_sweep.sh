#!/usr/bin/env bash
set -euo pipefail

steps=${IDEOGRAM4_SWEEP_STEPS:-2}
out_csv=${IDEOGRAM4_SWEEP_CSV:-/workspace/tmp/ideogram4/residency_sweep.csv}
windows=${IDEOGRAM4_SWEEP_WINDOWS:-0 2 4 8}
starts=${IDEOGRAM4_SWEEP_STARTS:-0}
model=${IDEOGRAM4_MODEL:-/srv/piclaw-dev/workspace/tmp/ideogram4-cat-model}
prompt_file=${IDEOGRAM4_PROMPT_FILE:-prompts/ideogram4/cat.json}
base_out_dir=${IDEOGRAM4_SWEEP_OUT_DIR:-/workspace/tmp/ideogram4/sweep}
mkdir -p "$(dirname "$out_csv")" "$base_out_dir" "${GOTMPDIR:-/workspace/tmp}"

echo 'window,start,steps,rc,qwen_s,denoise_s,vae_s,total_s,kernels,h2d,h2d_bytes,d2h,d2h_bytes,mallocs,malloc_bytes,log,out' > "$out_csv"

run_one() {
  local window=$1 start=$2
  local tag="w${window}_s${start}_${steps}steps"
  local log="${base_out_dir}/${tag}.log"
  local out="${base_out_dir}/${tag}.png"
  rm -f "$log" "$out"
  local rc=0
  GO_PHERENCE_IDEOGRAM4_GPU_STATS=1 \
  GO_PHERENCE_IDEOGRAM4_TIMING=1 \
  GO_PHERENCE_IDEOGRAM4_GPU_DIT_VECTOR=1 \
  GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_WINDOW="$window" \
  GO_PHERENCE_IDEOGRAM4_GPU_LAYER_CACHE_START="$start" \
  go run ./cmd/image/ideogram4gen \
    -model "$model" \
    -prompt "$(cat "$prompt_file")" \
    -out "$out" \
    -width "${IDEOGRAM4_WIDTH:-256}" \
    -height "${IDEOGRAM4_HEIGHT:-256}" \
    -steps "$steps" \
    -guidance "${IDEOGRAM4_GUIDANCE:-7.0}" \
    -mu "${IDEOGRAM4_MU:-0.0}" \
    -std "${IDEOGRAM4_STD:-1.75}" \
    -seed "${IDEOGRAM4_SEED:-2026060803}" \
    -gpu -gpu-fp8 -gpu-fp8-cache -gpu-residency "${IDEOGRAM4_GPU_RESIDENCY:-phase}" \
    -timing > "$log" 2>&1 || rc=$?

  local qwen denoise vae total stats
  qwen=$(grep -m1 'timing qwen_condition=' "$log" | sed -E 's/.*qwen_condition=([^ ]+).*/\1/' | python3 -c 'import sys;import re; s=sys.stdin.read().strip();
import math
m=re.match(r"(?:(\d+)m)?([0-9.]+)s",s); print((float(m.group(1) or 0)*60+float(m.group(2))) if m else "")' 2>/dev/null || true)
  denoise=$(grep -m1 'timing denoise=' "$log" | sed -E 's/.*denoise=([^ ]+).*/\1/' | python3 -c 'import sys,re; s=sys.stdin.read().strip(); m=re.match(r"(?:(\d+)m)?([0-9.]+)s",s); print((float(m.group(1) or 0)*60+float(m.group(2))) if m else "")' 2>/dev/null || true)
  vae=$(grep -m1 'timing vae_decode=' "$log" | sed -E 's/.*vae_decode=([^ ]+).*/\1/' | python3 -c 'import sys,re; s=sys.stdin.read().strip(); m=re.match(r"(?:(\d+)m)?([0-9.]+)s",s); print((float(m.group(1) or 0)*60+float(m.group(2))) if m else "")' 2>/dev/null || true)
  total=$(grep -m1 'timing generate=' "$log" | sed -E 's/.*generate=([^ ]+).*/\1/' | python3 -c 'import sys,re; s=sys.stdin.read().strip(); m=re.match(r"(?:(\d+)m)?([0-9.]+)s",s); print((float(m.group(1) or 0)*60+float(m.group(2))) if m else "")' 2>/dev/null || true)
  stats=$(grep -m1 'gpu_stats denoise' "$log" || true)
  local kernels h2d h2db d2h d2hb mallocs mallocb
  kernels=$(echo "$stats" | sed -nE 's/.*kernels=([0-9]+).*/\1/p')
  h2d=$(echo "$stats" | sed -nE 's/.* h2d=([0-9]+).*/\1/p')
  h2db=$(echo "$stats" | sed -nE 's/.*h2d_bytes=([0-9]+).*/\1/p')
  d2h=$(echo "$stats" | sed -nE 's/.* d2h=([0-9]+).*/\1/p')
  d2hb=$(echo "$stats" | sed -nE 's/.*d2h_bytes=([0-9]+).*/\1/p')
  mallocs=$(echo "$stats" | sed -nE 's/.*mallocs=([0-9]+).*/\1/p')
  mallocb=$(echo "$stats" | sed -nE 's/.*malloc_bytes=([0-9]+).*/\1/p')
  echo "$window,$start,$steps,$rc,$qwen,$denoise,$vae,$total,$kernels,$h2d,$h2db,$d2h,$d2hb,$mallocs,$mallocb,$log,$out" >> "$out_csv"
  echo "sweep $tag rc=$rc denoise=${denoise}s log=$log"
}

for w in $windows; do
  for s in $starts; do
    run_one "$w" "$s"
  done
done

echo "wrote $out_csv"
