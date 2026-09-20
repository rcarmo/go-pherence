# Needle 2 archive compatibility -- 2026-09-20

Needle 2 `.cact` archives can now be loaded with an explicit architecture JSON sidecar. The format uses magic `0x05E12A82`, a 20-byte five-field header, 44-byte tensor records and 64-byte payload alignment. Unlike Needle 3's `0x05E12A84` format, it does not store enough architecture metadata to reconstruct a model safely. The loader therefore refuses to guess.

## Explicit sidecar contract

`LoadArchiveWithConfig(path, configJSON)` accepts a bounded, strict Needle 2 architecture object. The CLI equivalent is `-archive-config file.json`. Sidecars are rejected for Needle 3 archives, which remain self-describing.

The sidecar is limited to 64 KiB, rejects duplicate/unknown/trailing fields, and requires explicit generation, vocabulary/model/attention/layer/context/mHC/engram/RoPE/padding/contrastive geometry. Generation must be 2. Needle 3-only convolution, window/global/ladder/seed-head geometry is rejected. The archive's KV window must fit the declared context. A tokenizer, when present, must match sidecar vocabulary size and padding ID.

```sh
bin/needle -model checkpoints/needle2.cact \
  -archive-config checkpoints/needle2-archive-config.json \
  -text-file prompt.txt -max-new 8
```

CLI output identifies fixed archive numerics as `archive-a8-fp32kv`. The pinned exporter writes header `kv_bits=8`, but its deployment configuration maps that setting to full-precision KV; see the [Needle 2 CQ record](needle2-cq-20260920.md). Lower-bit CQ KV archives are rejected. The CLI does not allow a sidecar for source safetensors or Needle 3 archives.

## Mapping and heads

The mapper restores the exporter's nameless records in their specified order: token embeddings, per-layer attention/norm/gate and three-vector Hadamard MLP, mHC coefficients/matrices, engram tables/projections/taps, final norm, optional heads, and optional tokenizer. Needle 2 has neither trained split permutations nor the Needle 3 conditional/factorized Hadamard records.

The optional FP16 head manifest admits strictly ordered contrastive/confidence codes. Contrastive uses four probes and sidecar `contrastive_dim`; confidence uses eight probes and one output. Exported projections are transposed back to checkpoint layout. The exporter writes a zero placeholder bias for contrastive output and omits `log_temp`; the loader validates and discards the zero placeholder and does not invent a temperature. This supports inference because normalized embeddings do not consume temperature. Archive training remains rejected.

Decoded tensors and packed CQ matrices are privately owned. Public archive mapping copies caller-owned records; tests clear the source archive after mapping and require unchanged logits. Packed projections remain opt-in and share the existing immutable CQ runtime. Decoded checkpoint snapshots can be loaded for inference, but they do not regain original packed blobs or training support.

## Reproducible evidence

[`scripts/needle2-archive-reference.py`](../../scripts/needle2-archive-reference.py) extracts architecture, quantization and export code from upstream pin `741ee892c5f8c4f5c0bb467c9566ea7a1eba919b`. It creates a CPU-only synthetic archive with 280 tokenizer pieces, 51 model/head records and an explicit sidecar, then reconstructs upstream parameters from the exporter's own reader and records logits and both head outputs.

```sh
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /path/to/cpu-jax-venv/bin/python scripts/needle2-archive-reference.py \
  --upstream /path/to/needle --outdir loader/needle/testdata
```

The generated archive is 35,222 bytes. Go tests compare all 51 decoded records to upstream, plus logits, contrastive/confidence outputs, every cached prefix and eight greedy generated IDs for decoded and packed execution at existing tolerances. Admission tests cover missing/mismatched/unknown sidecar fields, wrong generation, vocabulary/padding mismatch, bad head manifest/code/bias, shape errors, trailing records, unsupported KV bits, ownership, archive-training rejection and truncated/corrupt records. A short five-second parser fuzz smoke run completed without a crash; exhaustive allocation attribution makes execution count low and is not a throughput claim.

## Gates and limits

Final native ARM64 CIX P1 runs under `GOMAXPROCS=2 nice -n 10` passed **122 model, 58 loader, 11 CLI and 565 SIMD tests/subtests**. Three transfers through the `.local` hostname failed before execution; using the host's resolved LAN address completed the same final bundle. Whole-tree race, vet/build, RISC-V cross-build and CPU-feature-disabled checks pass; exact counts and preservation state are shared with the [second allocation report](needle-allocation-pass2-20260920.md).

This establishes bounded synthetic archive compatibility, not a released Needle 2 checkpoint or task-quality result. The format still depends on a trustworthy matching sidecar, does not support lower-bit KV execution, has no quantized-head parity, and is not packed-only because decoded weights remain resident. Independent review was unavailable.
