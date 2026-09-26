# Qwen3-TTS 64-frame sentence: failing qualification checkpoint

The released CPU sentence test fails waveform parity on production baseline
`216cdd55`. Tokenisation and all 1,024 codec IDs match the independent Rust
reference. Both implementations produce 64 frames and 122,880 samples without
observing EOS. The maximum waveform error is `2.7548521757125854e-6`, above the
unchanged `1.6e-6` gate. The maintenance checkpoint adds the reproducer and
fixtures, with no production decoder change.

The previously qualified [46-frame natural-EOS result](qwen3-tts-natural-eos-20260926.md)
still stands. This sentence exhausts the frame cap; complete speech, listener
quality and general prompt coverage have not been established.

## Pinned inputs and reproduction

- Oracle: `TrevorS/qwen3-tts-rs` revision
  `711ceee07cad92673f86de8997bdf54c30caa49f`, Candle 0.9.2, CPU F32.
- Checkpoint: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` revision
  `85e237c12c027371202489a0ec509ded67b5e4b5`.
- Text: “The quick brown fox jumps over the lazy dog and then walks slowly back
  to the house, where a warm dinner is waiting.”
- Ryan, English, seed 42, temperature 0.7, top-k 50, top-p 0.9, repetition
  penalty 1, minimum two semantic tokens, cap 64. EOS remains enabled.
- Six CPU workers; native `GOMAXPROCS=6`, reference `RAYON_NUM_THREADS=6`.
  One checkpoint process at a time; no GPU execution.

Copy `scripts/qwen3tts_probe_sentence_64.rs` into the pinned reference checkout
as `examples/probe_sentence_64.rs`, then run:

```sh
cargo build --release --no-default-features --features cpu --example probe_sentence_64
RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES='' \
  target/release/examples/probe_sentence_64 "$MODEL" "$ORACLE"
```

Two independent executions produced byte-identical observation, codes and
waveform files. The committed fixture test verifies source, observation and code
hashes. The released test additionally verifies all six model/tokeniser assets
and the external waveform (`491520` bytes, SHA-256
`e0ef875fbe94c1eb923b78c28fc72e5bca87af0b28985db07d3e8e641dcb02fb`).

```sh
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 \
  GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR="$MODEL" \
  GO_PHERENCE_QWEN3TTS_SENTENCE64_ORACLE_DIR="$ORACLE" \
  go test ./model/qwen3tts \
  -run '^TestCappedSentenceSeed42CPUReleasedSixtyFourFrames$' \
  -count=1 -v -timeout=1200s
```

**Expected current result: failure at the waveform gate.** The ordinary run took
234.58 seconds, with maximum RSS 5,250,816 KiB. Its later mixed-input reuse and
retained-output assertions were not reached. Released race qualification has
not been run for this failing sentence. Default offline tests skip the released
case unless both explicit environment variables are set.

## Decoder diagnosis

Fixed reference codes reproduce the production waveform error exactly. The
largest error occurs at sample 60,379, away from the frame boundary. Sampling
therefore does not account for this mismatch.

Temporary overlays compare stage outputs with an instrumented reference whose
full waveform remains byte-identical to the frozen reference. Initially the
traces selected 2,048 evenly spaced elements per stage; later experiments saved
full tensors and injected selected reference tensors into individual stages.
Injected runs only localise differences and cannot qualify end-to-end parity.

Source inspection and experiments identified several arithmetic differences:

- Candle uses im2col/GEMM for convolution, adds bias afterwards, and groups
  transposed-convolution channel sums before adding overlapping taps.
- On this host, Candle's GEMM uses balanced reduction blocks with a 512-element
  initial bound. Matching these blocks and sequential FMA makes the sampled
  codebook, pre-convolution and input-projection traces exact.
- Normalisation, GELU and SnakeBeta also differ in float32 operation order and
  elementary-function evaluation. The early `bias-snake` overlay's SnakeBeta
  text replacement did not match the source; that trial only changed bias
  ordering. The later `recip` overlay checks its replacement explicitly.

A diagnostic overlay combining blocked GEMM, source-matched normalisation and
C/libm elementary functions reached maximum waveform error
`1.321663148701191e-6` on fixed codes, with no reference injection. A Go-only
variant using the same matrix changes and reciprocal SnakeBeta still failed at
`4.392641130834818e-6`. Neither overlay is production code. No native correction,
performance improvement or full released-test pass is qualified. The C bridge
exists only under the temporary evidence directory; the repository has no new
C dependency.

## Checkpoint checks

`go test -race ./model/qwen3tts -count=10 -timeout=300s` passed in 18.844
seconds, including the offline pinned fixture. `go vet ./...`, `go build ./...`,
`make docs-check` (446 Markdown files, zero broken links) and `git diff --check`
passed. These checks did not enable released-model tests. No runtime or shared
backend code changed in this checkpoint.

## Evidence and next checks

Local evidence is under `/workspace/tmp/qwen3tts-cap64-20260926/`:

- `reference1/`, `reference2/`, `native.log`: frozen reference and failing run;
- `trace-comparison.txt`, `comparison-*.txt`, `trace-*.log`: diagnostics;
- `diagnose.ts`, `variant.ts`, `subtrace.ts`, `inject.ts`: temporary tooling;
- `trace-full-rust/`, `trace-sub-rust/`: stage tensors;
- `libmath/`, `libmath.mod`: diagnostic C bridge, outside the repository.

A maintenance request prompted this checkpoint; the hold was removed before
publication. Continue by checking actual generated overlay diffs before
execution, isolating the elementary-function discrepancy,
and producing a bounded native correction with focused regression tests. Then
rerun the original released test, its unreached reuse checks, affected released
fixtures, races and resource gates. Do not widen tolerance, force output length,
or claim intelligibility from numerical parity. GPU execution stays on hold.
