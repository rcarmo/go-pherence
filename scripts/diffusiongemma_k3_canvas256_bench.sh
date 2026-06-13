#!/usr/bin/env bash
set -euo pipefail

# Canonical K3 DiffusionGemma comparison harness.
# Uses a real text/chat prompt, canvas=256, and the sparse/A100 LM-head preset.
# This intentionally does not accept prompt_ids: prompt-ID shortcuts make results
# incomparable with llama.cpp's text-prompt run.

MODEL=${MODEL:-/home/me/models/diffusiongemma-26B-A4B-it-FP8}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-18110}
URL=${SERVER_URL:-http://${HOST}:${PORT}}
PROMPT=${PROMPT:-Say hello briefly.}
# Use enough generated tokens to evaluate coherence. Set MAX_TOKENS=1 only for
# low-level microbenchmarks.
MAX_TOKENS=${MAX_TOKENS:-16}
CANVAS=${CANVAS:-256}
# llama.cpp accepted --diffusion-steps 256 but DiffusionGemma EB metadata capped
# max_steps at 48 and the reference prompt stopped after 11 effective steps.
REQUESTED_DIFFUSION_STEPS=${REQUESTED_DIFFUSION_STEPS:-256}
DENOISE_STEPS=${DENOISE_STEPS:-11}
SAMPLER_MODE=${SAMPLER_MODE:-entropy_bound}
SEED=${SEED:-1}
LM_HEAD_TOP_K=${LM_HEAD_TOP_K:-64}
RETURN_STEPS=${RETURN_STEPS:-0}
OUT=${OUT:-/tmp/diffusiongemma_k3_canvas256_bench.jsonl}
SERVER_LOG=${SERVER_LOG:-/tmp/diffusiongemma_k3_canvas256_bench.server.log}
SERVER_PID=""

if [[ -n "${PROMPT_IDS:-}" ]]; then
  echo "error: this canonical benchmark refuses PROMPT_IDS; use PROMPT text" >&2
  exit 2
fi

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ -z "${SERVER_URL:-}" ]]; then
  SERVER_ARGS=${SERVER_ARGS:-"-allow-slow-cpu -cpu-dispatcher -add-bos -generation-prompt -k3 -k3-a100-q8 -k3-a100-lmhead -k3-a100-lmhead-prefetch -lm-head-top-k ${LM_HEAD_TOP_K} -k3-q80-residency-budget-gib ${Q80_BUDGET_GIB:-2.0} -k3-q80-retain-selected-expert-layers ${RETAIN_SELECTED_EXPERT_LAYERS:-30} -k3-q80-selected-prefetch"}
  echo "starting diffusiongemmaserver on ${HOST}:${PORT}" >&2
  # shellcheck disable=SC2086
  go run ./cmd/diffusiongemmaserver -model "${MODEL}" -listen "${HOST}:${PORT}" -max-new "${MAX_TOKENS}" -canvas "${CANVAS}" -denoise-steps "${DENOISE_STEPS}" ${SERVER_ARGS} >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 180); do
    if curl -fsS "${URL}/healthz" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "server exited early; log follows" >&2
      cat "${SERVER_LOG}" >&2 || true
      exit 1
    fi
    sleep 1
  done
fi

body=$(jq -cn \
  --arg prompt "${PROMPT}" \
  --argjson max_tokens "${MAX_TOKENS}" \
  --argjson canvas "${CANVAS}" \
  --argjson requested_diffusion_steps "${REQUESTED_DIFFUSION_STEPS}" \
  --argjson denoise_steps "${DENOISE_STEPS}" \
  --arg sampler_mode "${SAMPLER_MODE}" \
  --argjson seed "${SEED}" \
  --argjson return_steps "$(if [[ "${RETURN_STEPS}" == "1" ]]; then echo true; else echo false; fi)" \
  '{model:"diffusiongemma-k3",messages:[{role:"user",content:$prompt}],max_tokens:$max_tokens,canvas_length:$canvas,diffusion_steps:$requested_diffusion_steps,denoise_steps:$denoise_steps,sampler_mode:$sampler_mode,seed:$seed,return_diffusion_steps:$return_steps,stream:false}')

tmp=$(mktemp)
start_ns=$(date +%s%N)
curl -fsS -H 'Content-Type: application/json' -d "${body}" "${URL}/v1/chat/completions" >"${tmp}"
end_ns=$(date +%s%N)
wall_ms=$(((end_ns-start_ns)/1000000))
mkdir -p "$(dirname "${OUT}")"
jq -c \
  --arg prompt "${PROMPT}" \
  --argjson wall_ms "${wall_ms}" \
  --arg sampler_mode "${SAMPLER_MODE}" \
  --argjson requested_diffusion_steps "${REQUESTED_DIFFUSION_STEPS}" \
  --argjson lm_head_top_k "${LM_HEAD_TOP_K}" \
  '{prompt:$prompt,wall_ms:$wall_ms,sampler_mode:$sampler_mode,requested_diffusion_steps:$requested_diffusion_steps,lm_head_top_k:$lm_head_top_k,server_latency_ms:.usage.latency_ms,prompt_tokens:.usage.prompt_tokens,completion_tokens:.usage.completion_tokens,generated_token_ids:.generated_token_ids,text:.choices[0].message.content,diffusion_stats:.diffusion_stats,generated_tok_s:.diffusion_stats.generated_tokens_per_second,canvas_pos_s:.diffusion_stats.canvas_positions_per_second}' \
  "${tmp}" | tee -a "${OUT}"
rm -f "${tmp}"

echo "wrote ${OUT}" >&2
if [[ -n "${SERVER_PID}" ]]; then
  echo "server log: ${SERVER_LOG}" >&2
fi
