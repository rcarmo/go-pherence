# Whisper fixed FFT400 — 12 September 2026

This checkpoint replaces the exact frontend's per-bin direct DFT execution with a fixed 20×20 Cooley–Tukey FFT for the required 400-sample Whisper window.

## Contract

- Input/output geometry, centred reflect padding, periodic Hann coefficients, Slaney filters, final-frame removal, log clamp and normalisation are unchanged.
- The transform computes only the required bins 0..200 and preserves the existing complex64 rounding boundary before float64 magnitude.
- The old direct DFT tables remain in the package as an independent test oracle.
- The fixed plan is immutable after `sync.Once` construction; each call owns one 400-complex scratch buffer. No global mutable execution state or worker is added.
- This is CPU frontend work only. It changes no model, decoder, Vulkan path, default inference policy or quality threshold.

## Verification

- Direct-DFT comparisons pass for impulse, deterministic sinusoidal mixture and signed-zero inputs.
- All pinned Transformers 4.57.1 80- and 128-band frontend fixtures pass; the 80-band fixture remains bit-exact (`max abs diff 0`) and all 128-band values remain within the existing `1e-5` gate.
- `loader/audio`, `models/whisper` and `runtime/speechjob` pass; ten shuffled frontend repetitions pass.
- Affected `go vet`, Linux/ARM64 and Windows/AMD64 `loader/audio` test-binary cross-builds, and `git diff --check` pass.

## Isolated timing

On the i5-1340P with `GOMAXPROCS=2`, five 200-iteration samples measured:

- direct DFT: 1.354–1.420 ms per spectrum;
- mixed-radix FFT400: 12.02–12.12 µs per spectrum, zero allocations in the transform;
- median isolated speedup: approximately 112.7×.

This is a single-transform microbenchmark, not a whole frontend, model or job speed claim.
