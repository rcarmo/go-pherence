#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-${1:-}}"
if [[ -z "${MODEL}" ]]; then
  echo "usage: MODEL=/path/to/diffusiongemma-FP8 $0" >&2
  echo "       $0 /path/to/diffusiongemma-FP8" >&2
  exit 2
fi

CANVAS="${CANVAS:-16}"
STEPS="${STEPS:-2}"
MAX_NEW="${MAX_NEW:-1}"
PROMPT_IDS="${PROMPT_IDS:-2,3}"
Q80_BUDGET_GIB="${Q80_BUDGET_GIB:-2.0}"
LM_HEAD_TOP_K="${LM_HEAD_TOP_K:-8}"
RETAIN_SELECTED_EXPERT_LAYERS="${RETAIN_SELECTED_EXPERT_LAYERS:-0}"
A100_WORKERS="${A100_WORKERS:-6}"
K3_THREADS="${K3_THREADS:-8}"
LOG_DIR="${LOG_DIR:-/tmp}"
TAG="${TAG:-$(date +%Y%m%d-%H%M%S)}"
LOG="${LOG_DIR}/diffusiongemma-k3-profile-${TAG}.log"

K3_HOME="${K3_HOME:-/home/me}"
if [[ -z "${HOME:-}" || "${HOME}" == "/home/agent" ]]; then
  export HOME="${K3_HOME}"
fi
if [[ -z "${TMPDIR:-}" || ! -d "${TMPDIR}" ]]; then
  export TMPDIR="/tmp"
fi
export GOCACHE="${GOCACHE:-${K3_HOME}/.cache/go-build}"
export GOMODCACHE="${GOMODCACHE:-${K3_HOME}/go/pkg/mod}"
export IME2_Q80_TCM="${IME2_Q80_TCM:-1}"
export GO_PHERENCE_DIFFUSIONGEMMA_K3="${GO_PHERENCE_DIFFUSIONGEMMA_K3:-1}"
export GO_PHERENCE_DIFFUSIONGEMMA_K3_A100_Q8="${GO_PHERENCE_DIFFUSIONGEMMA_K3_A100_Q8:-1}"
export GO_PHERENCE_DIFFUSIONGEMMA_K3_THREADS="${GO_PHERENCE_DIFFUSIONGEMMA_K3_THREADS:-${K3_THREADS}}"
export GO_PHERENCE_DIFFUSIONGEMMA_K3_A100_WORKERS="${GO_PHERENCE_DIFFUSIONGEMMA_K3_A100_WORKERS:-${A100_WORKERS}}"

EXTRA_FLAGS=()
if [[ "${SKIP_EVICTION:-1}" == "1" ]]; then
  EXTRA_FLAGS+=("-skip-eviction")
fi
if [[ "${A100_LMHEAD:-1}" == "1" ]]; then
  EXTRA_FLAGS+=("-k3-a100-lmhead")
fi
if [[ "${A100_LMHEAD_PREFETCH:-1}" == "1" ]]; then
  EXTRA_FLAGS+=("-k3-a100-lmhead-prefetch")
fi
if [[ -n "${A100_LMHEAD_CANDIDATES:-128}" ]]; then
  EXTRA_FLAGS+=("-k3-a100-lmhead-candidates" "${A100_LMHEAD_CANDIDATES:-128}")
fi
if [[ "${Q80_PREFETCH:-0}" == "1" ]]; then
  EXTRA_FLAGS+=("-k3-q80-prefetch")
fi
if [[ "${Q80_PREFETCH_EXPERTS:-0}" == "1" ]]; then
  EXTRA_FLAGS+=("-k3-q80-prefetch-experts")
fi
if [[ "${Q80_SELECTED_PREFETCH:-0}" == "1" ]]; then
  EXTRA_FLAGS+=("-k3-q80-selected-prefetch")
fi

mkdir -p "${LOG_DIR}"
echo "diffusiongemma K3 profile log: ${LOG}" >&2

go run ./cmd/diffusiongemmarun \
  -model "${MODEL}" \
  -prompt-ids "${PROMPT_IDS}" \
  -max-new "${MAX_NEW}" \
  -canvas "${CANVAS}" \
  -denoise-steps "${STEPS}" \
  -cpu-dispatcher \
  -allow-slow-cpu \
  -lm-head-top-k "${LM_HEAD_TOP_K}" \
  -dispatch-progress \
  -k3-q80-residency-budget-gib "${Q80_BUDGET_GIB}" \
  -k3-q80-retain-selected-expert-layers "${RETAIN_SELECTED_EXPERT_LAYERS}" \
  "${EXTRA_FLAGS[@]}" \
  -json 2>&1 | tee "${LOG}"

printf '\n=== diffusiongemma K3 profile summary ===\n'
printf 'log=%s\n' "${LOG}"
grep "K3 Q80 residency budget\|K3 Q80 prewarmed" "${LOG}" || true
printf '\n-- decoder layer totals --\n'
grep "CPU dispatcher: completed layer" "${LOG}" | awk '
function ms(x){ if(x ~ /ms$/){sub("ms","",x); return x+0}; if(x ~ /s$/){sub("s","",x); return 1000*x}; return x+0 }
{
  pass = int((NR-1)/30) + 1
  for(i=1;i<=NF;i++){
    if($i ~ /^elapsed=/){split($i,a,"="); sum[pass]+=ms(a[2]); count[pass]++}
    if($i ~ /^q80_bytes=/){split($i,q,"="); if(q[2]>maxq) maxq=q[2]}
  }
}
END{
  for(p=1;p<=length(count);p++) printf("pass%d_layers=%d pass%d_ms=%.0f\n", p, count[p], p, sum[p]);
  total=0; for(p in sum) total+=sum[p];
  printf("decoder_total_ms=%.0f max_q80_bytes=%s\n", total, maxq)
}'
printf '\n-- tail --\n'
grep "completed tail op=lm_head\|completed self_conditioning" "${LOG}" || true
printf '\n-- output --\n'
awk 'BEGIN{s=0} /^\{/{s=1} s{print}' "${LOG}" | jq -c '{error:.error, generated:.result.generated, steps:.result.canvases[0].steps}' 2>/dev/null || true
