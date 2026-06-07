#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-/home/me/models/gguf-misc/qwen3-0.6b-q4_k_m.gguf}
PROMPT=${PROMPT:-Once upon a time}
TOKENS=${TOKENS:-8}
C_THREADS=${C_THREADS:-6}
GO_THREADS=${GO_THREADS:-6}
GO_SCALAR_THREADS=${GO_SCALAR_THREADS:-5}

cd "$(dirname "$0")/.."

if [[ -z "${HOME:-}" || ! -w "${HOME:-/nonexistent}" ]]; then export HOME=/home/me; fi
if [[ -z "${TMPDIR:-}" || ! -d "${TMPDIR:-/nonexistent}" || ! -w "${TMPDIR:-/nonexistent}" ]]; then export TMPDIR=/home/me/tmp; fi
export GOTMPDIR=${GOTMPDIR:-/home/me/tmp}
export GOCACHE=${GOCACHE:-/home/me/.cache/go-build}
export GOMODCACHE=${GOMODCACHE:-/home/me/go/pkg/mod}
mkdir -p "$TMPDIR" "$GOTMPDIR" "$GOCACHE" "$GOMODCACHE"

CGO_ENABLED=1 go build -tags llamacpp -o k3llama ./cmd/k3/k3llama
go build -o ime2run ./cmd/k3/ime2run

c_out=$(mktemp)
go_out=$(mktemp)
trap 'rm -f "$c_out" "$go_out"' EXIT

./k3llama \
  -model "$MODEL" \
  -prompt "$PROMPT" \
  -tokens "$TOKENS" \
  -threads "$C_THREADS" \
  -ignore-eos \
  -trace-ids > "$c_out"

./ime2run \
  -model "$MODEL" \
  -prompt "$PROMPT" \
  -tokens "$TOKENS" \
  -threads "$GO_THREADS" \
  -scalar-threads "$GO_SCALAR_THREADS" \
  -trace-ids > "$go_out" 2>&1

c_prompt=$(grep '^prompt ids:' "$c_out" | sed 's/^prompt ids: *//')
go_prompt=$(grep '^prompt ids:' "$go_out" | sed 's/^prompt ids: *//')
c_gen=$(grep '^gen ids:' "$c_out" | sed 's/^gen ids: *//')
go_gen=$(grep '^gen ids:' "$go_out" | sed 's/^gen ids: *//')

printf 'model: %s\n' "$MODEL"
printf 'prompt: %s\n' "$PROMPT"
printf 'tokens: %s\n\n' "$TOKENS"

printf 'C timing:\n'
grep -E '^(load|prefill|decode):' "$c_out" || true
printf '\nGo timing:\n'
grep -E '^(Loaded in|Prefill:|Decode:)' "$go_out" || true

printf '\nC prompt ids:  %s\n' "$c_prompt"
printf 'Go prompt ids: %s\n' "$go_prompt"
printf 'C gen ids:     %s\n' "$c_gen"
printf 'Go gen ids:    %s\n' "$go_gen"

if [[ "$c_prompt" != "$go_prompt" ]]; then
  printf '\nFAIL: prompt token IDs differ\n'
  exit 2
fi
if [[ "$c_gen" != "$go_gen" ]]; then
  printf '\nFAIL: generated token IDs differ\n'
  exit 1
fi
printf '\nPASS: token IDs match\n'
