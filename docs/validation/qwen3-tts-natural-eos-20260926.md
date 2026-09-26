# Qwen3-TTS native CPU natural EOS

## Result

Native Go now reproduces the pinned Rust/Candle natural end-of-speech observation
for **“Hi”, Ryan, English, seed42**, under a64-frame cap:

- **46 complete acoustic frames**, followed by codec EOS at semantic step46.
- All**736 codes** exact:46 semantic codes and690 acoustic codes.
- **88,320 mono samples at24kHz**, or3.68s of audio.
- Maximum F32 waveform error **1.3485550880432129e-6**, below the unchanged
  `1.6e-6` gate used for the qualified32-frame path.
- A subsequent32-frame request on the same models returns the exact semantic
  and acoustic prefix, does not report EOS, and leaves the held46-frame output
  unchanged.

This closes the specific native natural-EOS gap. It does not establish arbitrary
prompt intelligibility, all-speaker/language quality, released-model64-frame cap
exhaustion, streaming, NVIDIA,1.7B or long-running concurrency admission.

## Runtime change

`MaxCappedCPUFrames=64` replaces the former32-frame bound in the shared bounded
CustomVoice CPU path. No sampling distribution, PCG state, minimum-two-token EOS
policy, convolution arithmetic, decoder or fixture tolerance changes.

`BoundedCPUResult.StoppedAtEOS` distinguishes an observed continuation EOS from
cap exhaustion:

- True only when a continuation selects EOS before another acoustic frame.
  No EOS token, acoustic codes, hidden row or raw-logit row is appended for it.
- False on a successful result means the cap was reached. It does not predict
  whether the next step would be EOS; the bounded loop does not sample past the cap.
- First-token EOS remains an error with a zero result, including a false flag.
- With success and `StoppedAtEOS=true`, accepted frames are fewer than the cap.

The existing request-local KV, RoPE and scratch reservations use
`prefixLen + MaxFrames - 1`, so the higher cap remains bounded and does not add
shared mutable state. Waveform decoding uses actual accepted frames, not the
plan's maximum sample budget. Existing fixed2/3/4-frame APIs remain fixed-size.
The minimum-two-token policy still permits a one-frame cap; it never exceeds the
caller cap merely to make EOS eligible.

## Independent reference and assets

Baseline Go revision: `53246130`. The previously approved checkpoint was absent
after maintenance and restored under `/dev/shm/qwen3tts-0b6-customvoice`, without
removing other artifacts:

- Model: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` revision
  `85e237c12c027371202489a0ec509ded67b5e4b5`.
- Model safetensors:1,811,626,576 bytes,
  SHA-256 `bc3c7e785eb961179c25450d1acff03f839e0002f2f3a5aeb67b5735c0fa2adb`.
- Speech tokenizer safetensors:682,293,092 bytes,
  SHA-256 `836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258`.
- Both configs, text vocab and merges are hash-checked as well; all six asset
  hashes match the existing repository reference.
- Rust oracle revision: `TrevorS/qwen3-tts-rs@711ceee07cad92673f86de8997bdf54c30caa49f`.
  Existing `scripts/qwen3tts_probe_hi_64_eos.rs` is copied as an example and built
  with `--no-default-features --features cpu`.

The regenerated codes and EOS observation match the already pinned fixture
byte-for-byte. Regenerated waveform SHA-256 is
`8011f727acf104e24e606b5aea4f3881bf55e1d5d575dd5063586a72dce33da2`;
the new opt-in native test pins it before comparison. The code fixture SHA is
`076cd7e828d71c652aae38471928e7f111c6b5334e746e9bc9a09610b23bd449`.
No Go output participated in producing the expected waveform/codes.

No model downloads beyond the already approved pinned revision, GPU kernels,
frozen evaluations or unrelated service restarts occurred. Model processes ran
sequentially with six-worker settings (`GOMAXPROCS=6` for Go; Rust Rayon/OMP6).
The Go test used Go1.26.3/Linux amd64. Rust build/source and command logs are
preserved outside the repository; waveform bytes remain outside Git.

## Correctness and resource evidence

| Gate | Outcome |
|---|---|
| Independent Rust/Candle64-cap observation | Reproduced46-frame EOS and exact pinned codes |
| Native ordinary natural-EOS plus32-prefix/reuse test | PASS208.49s |
| Same released test under race | PASS1777.24s |
| Qwen3-TTS synthetic/package race, count10 | PASS18.326s |
| Whole-tree NVIDIA-disabled CPU race,189 packages | Exit0, wall5m02.99s |
| `go vet ./...`, `go build ./...`, docs checks | Passed |
| Linux ARM64/RISC-V Qwen3-TTS test cross-builds | Passed; not native execution |

Synthetic integration cases cover cap32 exhaustion, EOS immediately after32
accepted frames with cap33, EOS after33/46/63 with cap64, and full64-frame
exhaustion. They check all semantic/acoustic/waveform/continuation lengths and
EOS flags. First-EOS zero-result and65-cap rejection are explicit. Existing
minimum-two-token, seed sampling and concurrent request-local ownership tests
remain enabled. **Full64-frame exhaustion uses tiny synthetic weights**, not a
new released-model case; only the46-frame natural-EOS path has new released
qualification.

The ordinary released test peak RSS was5,021,184KiB (4.79GiB), zero swaps.
Post-GC heap increased4,079,662,080→4,081,523,240 bytes while both46- and32-frame
results and test copies were deliberately retained. That is not a leak or
hours-long retention qualification. The race run peak was13,907,164KiB
(13.26GiB), which must not be treated as ordinary admission memory.

These durations include loading, validation,46-frame synthesis/decoding and a
second32-frame request; they are not a single-request latency benchmark or a
Rust-versus-Go speed comparison. No optimization speedup/allocation saving is
claimed in this bounded capability extension.

Ordinary package coverage is79.3%; bounded public entrypoints are100%, while the
shared `generateCappedGreedyCPU` is83.7%. Not every injected error branch is
covered, so the changed-code coverage target remains open rather than being
silently weakened. Released parity was checked separately without coverage
instrumentation.

A delegated review of the described change found no obvious blocker, highlighting
EOS-at-cap semantics and the lack of a full released64-frame exhaustion run.
Those limits are explicit here. It was not an independent filesystem audit.
An early test compile failed due to incorrect helper names and was corrected;
its log is retained. An interrupted whole-tree run had no completion status;
only the subsequent exit0 run counts as a pass. Completed released runs were
not repeated when resuming after that interruption.

## Listening sample and quality

The attached Go and Rust listening files contain the same3.68s segment as their
respective raw fixtures, converted to PCM16 WAV at24kHz without resampling or
loudness normalization. Raw F32 files are used for parity, not quantized WAVs.
Go sample peak is0.22283782 and RMS0.04577449, confirming finite, non-silent
samples, not intelligibility or good pronunciation. Prior short-sample listener
feedback included laughter on another prompt in both Go and Rust; numerical
agreement is not evidence that such quality issues are resolved. Human review
of this natural-EOS sample is still required.

## Reproduction and evidence

Evidence: `/workspace/tmp/qwen3tts-natural-eos-20260926`. Asset hashes, regenerated
reference codes/waveform, native output, test/build logs, WAV conversion and
binary/source provenance are retained. Checkpoints are under `/dev/shm`; Rust
source and build artifacts remain outside Git. The downloadable archive excludes
weights, Rust target/cache and foreign test executables.

```sh
GOMAXPROCS=6 GO_PHERENCE_DISABLE_NVIDIA=1 \
  GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR=/dev/shm/qwen3tts-0b6-customvoice \
  GO_PHERENCE_QWEN3TTS_HI64_ORACLE_DIR=/path/to/regenerated/reference \
  go test ./model/qwen3tts -run '^TestCappedHiSeed42CPUReleasedNaturalEOS$' \
  -count=1 -v -timeout=1200s
# Add -race and use -timeout=2400s for the instrumented gate.
```

No claim is made for arbitrary lengths, more than two released concurrent
requests, cancellation, hours-long retention, GPU execution or general voice
quality. Those remain separate speech-model work.
