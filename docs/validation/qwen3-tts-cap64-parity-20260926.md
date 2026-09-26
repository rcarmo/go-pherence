# Qwen3-TTS CPU: 64-frame sentence parity

The approved 0.6B CustomVoice CPU checkpoint now passes the pinned independent
64-frame sentence test. All 1,024 codec codes match Rust/Candle exactly. The
122,880-sample waveform has maximum absolute error
`1.2848759070038795e-6`, below the unchanged `1.6e-6` gate. The sentence hits
its 64-frame cap without selecting EOS; completion and intelligibility are not
established. After listening to the attached Go and Rust MP3 copies, one
listener reported that they seem identical. This observation compares those
two five-second clips; it does not establish what words were spoken or whether
the sentence finished. The [failing checkpoint](qwen3-tts-cap64-diagnostic-20260926.md)
records the earlier `2.7548521757125854e-6` failure and frozen evidence.

## Inputs and correction

- CPU reference: `TrevorS/qwen3-tts-rs@711ceee07cad92673f86de8997bdf54c30caa49f`,
  Candle 0.9.2; checkpoint
  `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice@85e237c12c027371202489a0ec509ded67b5e4b5`.
  Both independent oracle runs yielded identical bytes.
- Speaker Ryan, English, seed 42, EOS enabled, cap 64. The test hashes all six
  approved model/tokeniser assets, the external Rust waveform and the checked-in
  codes, observation and oracle script. It compares text IDs and every code
  before comparing every waveform sample.
- On amd64 with native SGEMM, the CPU decoder now follows the reference's
  balanced F32/FMA matrix reductions, im2col convolution, bias-after-reduction
  order, and channel-summed transposed convolution with overlapping taps. Other
  architectures keep the prior portable implementation. This change does not
  alter the Talker, CodePredictor, sampling, GPU paths or frame cap.
- Decoder ConvNeXt GELU uses an F32 `erff` port from Rust `libm=0.2.16`, with
  its FreeBSD/Sun licence notice preserved. The temporary C bridge was not
  added. The checked-in independent Rust fixture tests 9,387 F32 input/output
  pairs bitwise. An additional temporary oracle compared 1,573,057 boundary
  and released ConvNeXt inputs bitwise with zero mismatches. The libm source
  hashes are `erff.rs` `62c30876…76196`, `expf.rs` `33b01180…f01c0`,
  `scalbn.rs` `9b0297d1…125c05`, `generic/scalbn.rs`
  `986e37f1…1ebf28`; full hashes and generated evidence are under
  `/workspace/tmp/qwen3tts-cap64-20260926/`.
- Shared GEBP now sizes unsafe views to the last element accessed and accepts
  caller-owned packing scratch. A small offset-row checkptr test caught the
  original slice-bound error; its ten race repetitions pass. The decoder
  reuses per-call packing scratch without sharing mutable data between requests.

## Released and local checks

Commands use `GOMAXPROCS=6`, `GO_PHERENCE_DISABLE_NVIDIA=1`, one checkpoint
process at a time and `TMPDIR=/tmp` for build scratch. Tests load the approved
assets from `/dev/shm/qwen3tts-0b6-customvoice`. The sentence oracle directory
is `/workspace/tmp/qwen3tts-cap64-20260926/reference1`.

| Check | Result |
|---|---|
| `TestCappedSentenceSeed42CPUReleasedSixtyFourFrames` ordinary | Pass: 64 frames, EOS false, 1,024 exact codes; max waveform error `1.2848759070038795e-6`; same-instance short mixed-input recovery and retained outputs passed. 54.55 s; peak RSS 5,327,616 KiB. |
| The same released test under `-race` | Pass in 120.56 s, with the same codes, waveform error and reuse checks. A first attempt ended without a result; preserved log has only the test start. The earlier successful retry before caller-owned packing peaked at 17,246,008 KiB RSS; final race peak was not captured. |
| `TestCappedHiSeed42CPUReleasedNaturalEOS` | Pass: EOS at step 46, 46 frames/736 exact codes, 88,320 samples; max waveform error `6.705522537231445e-7` versus its unchanged `1.6e-6` gate. No need to repeat the independent Rust oracle. |
| Earlier released 2/8/16/32-frame gates | Pass, maximum waveform errors `3.15049e-9`, `1.96742e-7`, `5.21541e-7`, `6.70552e-7` respectively, each under its existing threshold. |
| Affected package race | `go test -race ./backends/simd/runtime ./model/qwen3tts -count=10 -timeout=300s` passed, including GEBP zero-allocation/offset-view and eight concurrent decoder calls. |
| Whole-tree CPU race | `go test -race -p=2 ./... -count=1 -timeout=180s` passed: 189 package results; disabled NVIDIA. |
| Static and foreign build gates | `go vet ./...`, `go build ./...`, `make docs-check`, `git diff --check`, and ARM64/RISC-V `go test -c` for both affected packages passed. Cross-compilation is not native execution. |

The ordinary baseline failed after 234.58 s before its reuse assertions, at
5,250,816 KiB peak RSS. The corrected test takes 54.55 s *including* the
previously unreachable second request; the two times are different workloads.
Peak RSS is 76,800 KiB higher. No peak-admission improvement is claimed.

Five warm 16-frame benchmark samples, with checkpoint loading outside the timed
loop, were `12.744`, `12.754`, `12.980`, `13.028` and `13.201` s/op after packing
reuse, all at `345,173,776 B/op` and `4,422 allocs/op`. The immediately preceding
correctness-qualified decoder before packing reuse measured `12.351`–`12.848`
s/op, `462,479,248 B/op`, `12,063 allocs/op` across five samples. Packing
reuse saves 117,305,472 B and 7,641 allocations per request; the samples do
not support a latency gain from packing. The older published decoder benchmark
measured about 42.4 s/op and 329,877,008 B/4,323 allocations; current staging
adds 15,296,768 B and 99 allocations per request against that historical
allocation baseline. Different sequential runs limit timing comparisons.

The ordinary package coverage run reached 79.5% overall. New reachable
`erff`/`erfc` branches reached 100%; rare `expf`/`scalbnf` branches outside the
GELU input range were lower. The new matrix convolution methods were 92.9%
and 96.3% covered on amd64, with portable foreign paths only cross-built.

Evidence: `/workspace/tmp/qwen3tts-cap64-20260926/` contains
`pack-released-cap64.log`, `pack-released-cap64-race.log`,
`pack-regression-natural-eos.log`, `production-greedy-released.log`,
`production-short-released.log`, `pack-race10-final.log`,
`pack-whole-tree-race.log`, `pack-benchmark-sixteen.log`, vet/build/docs/cross
logs and temporary oracle traces. No GPU work ran. Arbitrary prompts, listening
quality, cancellation, broader concurrency, hours-long retention, streaming,
other checkpoint sizes and native ARM64/RISC-V numerical execution remain open.
