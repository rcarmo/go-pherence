#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-/home/me/models/gguf-misc/qwen3-0.6b-q4_k_m.gguf}"
BIN="${BIN:-./ime2run}"
TOKENS="${TOKENS:-32}"

run_case() {
  local label="$1" prompt="$2" envspec="$3"
  env IME2_NO_CHAT_TEMPLATE=1 ${envspec} "$BIN" -model "$MODEL" -tokens "$TOKENS" -prompt "$prompt" 2>&1 \
    | awk -v label="$label" '/^Output:/{sub(/^Output: /, ""); print label "\t" $0}'
}

check_prompt() {
  local prompt="$1"
  echo "checking prompt: $prompt" >&2
  local base
  base=$(run_case default "$prompt" "")
  local base_out=${base#*$'\t'}
  local modes=(
    "q8_round_off:IME2_Q8_ROUND=0"
    "down_add_off:IME2_INT8_DOWN_ADD=0"
    # B-wave paths require 626-line kernel; skip for now
    # "q4_bwave_safe:IME2_TCM_B_WAVE=1"
    # "int8_down_bwave:IME2_INT8_TCM_B_WAVE=1"
    # "combined_bwave:IME2_TCM_B_WAVE=1 IME2_INT8_TCM_B_WAVE=1"
  )
  for item in "${modes[@]}"; do
    local label=${item%%:*}
    local envspec=${item#*:}
    local line out
    line=$(run_case "$label" "$prompt" "$envspec")
    out=${line#*$'\t'}
    if [[ "$out" != "$base_out" ]]; then
      echo "MISMATCH for $label" >&2
      echo "default: $base_out" >&2
      echo "$label: $out" >&2
      return 1
    fi
    echo "ok $label" >&2
  done
}

check_prompt "The capital of Portugal is"
check_prompt "Q: Write one word: blue. A:"

# Diagnostic-only: native i8i8 routes are direct-port parity experiments and
# are not default-equivalent yet. Print their output without making this safe
# check fail.
echo "diagnostic native_i8i8_down:" >&2
run_case native_i8i8_down "The capital of Portugal is" "IME2_NATIVE_I8I8_DOWN=1" >&2 || true

echo "diagnostic native_i8i8_lm:" >&2
run_case native_i8i8_lm "Q: Write one word: blue. A:" "IME2_NATIVE_I8I8_LM=1" >&2 || true

echo "correctness checks passed" >&2
