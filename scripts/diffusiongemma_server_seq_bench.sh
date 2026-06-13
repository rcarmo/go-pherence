#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-/home/me/models/diffusiongemma-26B-A4B-it-FP8}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-18080}
URL=${SERVER_URL:-http://${HOST}:${PORT}}
REQUESTS=${REQUESTS:-3}
MAX_TOKENS=${MAX_TOKENS:-1}
CANVAS=${CANVAS:-16}
DENOISE_STEPS=${DENOISE_STEPS:-2}
SEED=${SEED:-1}
PROMPT_IDS=${PROMPT_IDS:-2,106,107}
RETURN_STEPS=${RETURN_STEPS:-1}
STREAM=${STREAM:-0}
OUT=${OUT:-/tmp/diffusiongemma_server_seq_bench.jsonl}
SERVER_LOG=${SERVER_LOG:-/tmp/diffusiongemma_server_seq_bench.server.log}
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ -z "${SERVER_URL:-}" ]]; then
  SERVER_ARGS=${SERVER_ARGS:-"-k3 -allow-slow-cpu -cpu-dispatcher -k3-a100-q8 -k3-q80-residency-budget-gib ${Q80_BUDGET_GIB:-2.0} -k3-q80-retain-selected-expert-layers ${RETAIN_SELECTED_EXPERT_LAYERS:-30} -k3-q80-selected-prefetch"}
  if [[ "${MAX_DISPATCH_LAYERS:-}" != "" ]]; then
    SERVER_ARGS+=" -max-dispatch-layers ${MAX_DISPATCH_LAYERS}"
  fi
  if [[ "${TAIL_AFTER_MAX_LAYERS:-0}" == "1" ]]; then
    SERVER_ARGS+=" -tail-after-max-layers"
  fi
  echo "starting diffusiongemmaserver on ${HOST}:${PORT}" >&2
  # shellcheck disable=SC2086
  go run ./cmd/diffusiongemmaserver -model "${MODEL}" -listen "${HOST}:${PORT}" -max-new "${MAX_TOKENS}" -canvas "${CANVAS}" -denoise-steps "${DENOISE_STEPS}" ${SERVER_ARGS} >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 120); do
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

ids_json="[$(printf '%s' "${PROMPT_IDS}" | sed 's/[[:space:]]//g')]"
rm -f "${OUT}"
for i in $(seq 1 "${REQUESTS}"); do
  body=$(cat <<JSON
{"model":"diffusiongemma-k3","prompt_ids":${ids_json},"max_tokens":${MAX_TOKENS},"canvas_length":${CANVAS},"denoise_steps":${DENOISE_STEPS},"seed":${SEED},"return_diffusion_steps":$(if [[ "${RETURN_STEPS}" == "1" ]]; then echo true; else echo false; fi),"stream":$(if [[ "${STREAM}" == "1" ]]; then echo true; else echo false; fi)}
JSON
)
  tmp=$(mktemp)
  start_ns=$(date +%s%N)
  if [[ "${STREAM}" == "1" ]]; then
    curl -fsS -N -H 'Content-Type: application/json' -d "${body}" "${URL}/v1/chat/completions" >"${tmp}"
    end_ns=$(date +%s%N)
    step_count=$(grep -c '^event: diffusion_step' "${tmp}" || true)
    done_seen=$(grep -c '^data: \[DONE\]' "${tmp}" || true)
    jq -cn --argjson request "$i" --argjson wall_ms "$(((end_ns-start_ns)/1000000))" --argjson step_count "${step_count}" --argjson done_seen "${done_seen}" '{request:$request,wall_ms:$wall_ms,stream:true,diffusion_step_events:$step_count,done_seen:$done_seen}' | tee -a "${OUT}"
  else
    curl -fsS -H 'Content-Type: application/json' -d "${body}" "${URL}/v1/chat/completions" >"${tmp}"
    end_ns=$(date +%s%N)
    jq -c --argjson request "$i" --argjson wall_ms "$(((end_ns-start_ns)/1000000))" '{request:$request,wall_ms:$wall_ms,server_latency_ms:.usage.latency_ms,prompt_tokens:.usage.prompt_tokens,completion_tokens:.usage.completion_tokens,generated_token_ids:.generated_token_ids,diffusion_steps:(.diffusion_steps|length)}' "${tmp}" | tee -a "${OUT}"
  fi
  rm -f "${tmp}"
done

echo "wrote ${OUT}" >&2
if [[ -n "${SERVER_PID}" ]]; then
  echo "server log: ${SERVER_LOG}" >&2
fi
