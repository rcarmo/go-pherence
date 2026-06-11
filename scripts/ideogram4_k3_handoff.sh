#!/usr/bin/env bash
set -euo pipefail
out=${IDEOGRAM4_K3_HANDOFF_DIR:-/workspace/tmp/ideogram4/k3-handoff}
mkdir -p "$out"

cp -f bin/ideogram4gen-k3 "$out/" 2>/dev/null || true
cp -f bin/ideogram4vaeprobe-k3 "$out/" 2>/dev/null || true
./scripts/ideogram4_k3_coverage.py > "$out/coverage.json"

cat > "$out/README.md" <<'MD'
# Ideogram4 K3 handoff

This handoff bundle contains the current riscv64 Ideogram4 binaries for SpaceMIT
K3 hardware smoke testing.

## Binaries

```text
ideogram4gen-k3       Full text-to-image CLI
ideogram4vaeprobe-k3  VAE-only decode probe
```

## Required model

Provide a local Ideogram4 Diffusers directory containing:

```text
tokenizer/
text_encoder/
transformer/
unconditional_transformer/
vae/
```

## Environment

```bash
export IME2_TCM_ACT=1
```

## Full pipeline smoke

```bash
./ideogram4gen-k3 \
  -k3 -k3-threads 8 -k3-prewarm \
  -model /path/to/ideogram4-model \
  -prompt "$(cat prompts/ideogram4/cat.json)" \
  -width 256 -height 256 -steps 4 \
  -guidance 7.0 -mu 0.0 -std 1.75 \
  -seed 2026060803 \
  -timing
```

If `prompts/ideogram4/cat.json` is not present on the device, copy it from the
repository or pass the same structured JSON caption manually.

## VAE-only smoke

```bash
./ideogram4vaeprobe-k3 \
  -k3 -k3-threads 8 -k3-prewarm \
  -model /path/to/ideogram4-model \
  -width 256 -height 256
```

## Current K3 coverage status

See `coverage.json` for the machine-readable current coverage matrix. Summary:

Implemented K3-specific bridges:

- FP8 `FP8Linear.Apply/ApplyBatch`: K3 RVV fp16 bridge under
  `GO_PHERENCE_IDEOGRAM4_K3=1` / `-k3`.
- FP8 resident prewarm: decoded fp16 rows and N32-packed fp16 weights cached per
  `FP8Linear` for the 24GB profile.
- VAE Conv2D im2col GEMM: K3 RVV fp16 bridge under `-k3`.

Remaining hardware/kernel work:

- Replace FP8→fp16 bridge with fused FP8→int8/IME2 + TCM staging.
- Add K3/RVV kernels for RMSNorm, LayerNorm, RoPE/MRoPE, attention/softmax,
  SiLU/SwiGLU, CFG/update, VAE GroupNorm/Upsample/RGB, and VAE attention.
- Validate A100 worker placement (`/proc/set_ai_thread`, cores 8–15) and TCM.

## Expected first test outcome

The current binaries should start and exercise the K3-gated bridge paths, but
full performance and full SIMD coverage are not complete until the remaining
kernel matrix is implemented and tested on hardware.
MD

echo "wrote $out"
