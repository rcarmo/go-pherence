# Qwen3-TTS 64-frame output projection on NVIDIA

The released 0.6B Decoder12Hz output projection passed a second opt-in RTX 3060 stage test. Production waveform synthesis and runtime status remain CPU-only.

## Frozen reference and boundary

- Starting Go revision: `e95ff5e4c7c764dcf9aaff2aeae8f38c5bf03068`. GPU: GeForce RTX 3060, driver 610.57.04; Go 1.26.3, linux/amd64, i7-12700, `GOMAXPROCS=6`.
- Released checkpoint: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` revision `85e237c12c027371202489a0ec509ded67b5e4b5`; the test verifies SHA-256 of `speech_tokenizer/config.json` and `model.safetensors` before loading. The existing sentence64 fixture hashes the pinned Rust source, observation and 1,024 codec IDs.
- CPU oracle: `TrevorS/qwen3-tts-rs` revision `711ceee07cad92673f86de8997bdf54c30caa49f`, Candle 0.9.2. The instrumented local `src/models/codec/decoder_12hz.rs` has SHA-256 `a2edf2ad569200dc64d2bc07ae8d95d25c32cb20130435d3ca036f579d74c761`. The decoder-only probe `examples/probe_decoder_codes.rs` has SHA-256 `4ea10f1b8c96481d3d7a83b9c197a2c2972b200be8dc99693cd7307eb98391f6`. The source checkout was not changed for this test.
- The CPU oracle decoded the pinned 64×16 codes with `RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES='' QWEN_FULL_TRACE=1`. Its sampled `finalnorm` and `outputproj` traces reproduced the prior saved hashes; its waveform reproduced the frozen waveform SHA-256 `e0ef875fbe94c1eb923b78c28fc72e5bca87af0b28985db07d3e8e641dcb02fb`. The committed full time-major `[64,512]` final-normalisation tensor hashes to `a0e9e1ee8b5ff796aaf8a2228ac0fdb53666b870474b44d1b79fe630f3145713`; the full `[64,1024]` output hashes to `9324dd4c394cf844f55dd121ee384b51b0afc83bf4082401c2c24cb59d50745d`. The 2,048 evenly spaced output samples hash to `85ef81eefb6e8896d39fd320f000952904ca4c41b9edec71b24e32b2560c7df8` and match their positions in the full output exactly.

Reproduce the independent trace outside the Go tree:

```sh
cd /workspace/tmp/qwen3tts-natural-eos-20260926/oracle
CARGO_TARGET_DIR=/workspace/tmp/qwen3tts-gpu-outputproj-20260927/target \
  cargo build --offline --release --no-default-features --features cpu --example probe_decoder_codes
RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES='' QWEN_FULL_TRACE=1 \
  QWEN_DECODER_TRACE=/workspace/tmp/qwen3tts-gpu-outputproj-20260927/rust \
  /workspace/tmp/qwen3tts-gpu-outputproj-20260927/target/release/examples/probe_decoder_codes \
  /dev/shm/qwen3tts-0b6-customvoice \
  /workspace/projects/go-pherence/model/qwen3tts/testdata/customvoice_0b6_ryan_hello/sentence_64_codes.u32le
```

The Go test consumes the independently traced finalnorm tensor, loads owned released output-projection weights and adds the existing F32 bias on CPU after `SgemmHost`. The GPU computes only the matrix product. It compares all 65,536 outputs against both independent Rust and native Go CPU, and checks Rust's sampled output against the full Rust trace. The fixed diagnostic GPU maximum-absolute-error threshold is `1e-6`. It does not replace the `1.6e-6` end-to-end CPU waveform gate.

```sh
GOMAXPROCS=6 GO_PHERENCE_QWEN3TTS_GPU_TEST=1 \
  GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR=/dev/shm/qwen3tts-0b6-customvoice \
  go test ./model/qwen3tts -run '^TestDecoderOutputProjectionPinnedGPU$' \
  -count=10 -v -timeout=300s
# Repeat with go test -race and -count=3.
```

Ten ordinary and three race repetitions passed. Whole-tree CPU race (132 package results), `go vet ./...`, `go build ./...`, docs/layout checks (455 Markdown files, zero broken links) and linux/arm64 and linux/riscv64 test cross-builds passed. The default offline test skips this hardware gate when the opt-in flag is unset. Both Rust/CPU and Rust/GPU full outputs matched bitwise in this fixed slice (maximum error zero). Per call, NVIDIA counters reported one launch, 2,228,224 bytes host-to-device, 262,144 bytes device-to-host, and three matched allocations/frees totalling 2,490,368 bytes. These counters describe transfers and allocations, not peak device memory, speed, concurrent lifecycle or retained VRAM. Idle GPU memory was 33 MiB before the test. Transformer execution, GPU waveform decoding, quality and production admission remain unqualified.
