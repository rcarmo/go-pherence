# Qwen3-TTS 64-frame causal pre-convolution on NVIDIA

A 512-wide blocked NVIDIA SGEMM diagnostic matched the released Qwen3-TTS decoder pre-convolution against the full independent Rust/Candle and native Go CPU tensors, bitwise. Ordinary and compensated unblocked SGEMM both failed the fixed `1e-6` stage threshold. The production decoder and waveform path remain CPU-only.

## Inputs and boundary

- Go baseline: `e06adceafcff66410b9637aa70b12ef52339a31a`, clean before this test. GeForce RTX 3060 (driver 610.57.04), Go 1.26.3 on linux/amd64, i7-12700, `GOMAXPROCS=6`.
- Checkpoint: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` revision `85e237c12c027371202489a0ec509ded67b5e4b5`. The opt-in test hashes both `speech_tokenizer/config.json` and `model.safetensors` before loading; the existing sentence64 fixture verifies the independent probe, observation and all 1,024 codec IDs.
- Independent CPU oracle: `TrevorS/qwen3-tts-rs` revision `711ceee07cad92673f86de8997bdf54c30caa49f`, Candle 0.9.2. Local instrumented `src/models/codec/decoder_12hz.rs` SHA-256 `a2edf2ad569200dc64d2bc07ae8d95d25c32cb20130435d3ca036f579d74c761`; `examples/probe_decoder_codes.rs` SHA-256 `4ea10f1b8c96481d3d7a83b9c197a2c2972b200be8dc99693cd7307eb98391f6`. The oracle was run CPU-only with `RAYON_NUM_THREADS=6 CUDA_VISIBLE_DEVICES='' QWEN_FULL_TRACE=1` over the pinned sentence64 codes. Its sampled stage traces matched the earlier frozen traces and its waveform matched the frozen `e0ef875fbe94c1eb923b78c28fc72e5bca87af0b28985db07d3e8e641dcb02fb` SHA-256.
- Committed full F32 input is channel-major `[512,64]`, SHA-256 `4cce48ec0c48189105f28e24ab6404cadbdec7130f84a3f07c19979989298a9d`; the full channel-major `[1024,64]` pre-convolution output hashes to `75c0a7feda1be45023577e1f2675f909b75351fd5bc977978e385597a4703685`.

The test loads the decoder's owned 3-tap `[1024,512,3]` weights and bias. It builds one zero-left-padded `[64,1536]` im2col input and transposes weights to `[1536,1024]` for SGEMM. Three contiguous 512-wide products are downloaded and added in F32, in Candle's block order, then the F32 bias is added. It checks all 65,536 outputs against both independent Rust and native Go CPU. This is host-staged diagnostic arithmetic, not a device-resident convolution or a GPU waveform path.

## Results

The initial single unblocked `SgemmHost(64,1024,1536)` differed from both Rust and CPU by up to `4.29153442e-6`; the backend's separately named `SgemmCompensated` policy reduced the maximum to `2.38418579e-6`. Both exceeded `1e-6`. The blocked 512+512+512 test kept that threshold and passed 10 ordinary and 3 race repetitions with zero maximum difference against all pinned Rust and CPU values. Baseline and compensated failures are preserved under `/workspace/tmp/qwen3tts-gpu-outputproj-20260927/preconv-{diagnostic,compensated}.log`.

```sh
GOMAXPROCS=6 GO_PHERENCE_QWEN3TTS_GPU_TEST=1 \
  GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR=/dev/shm/qwen3tts-0b6-customvoice \
  go test ./model/qwen3tts -run '^TestDecoderPreConvPinnedGPU$' -count=10 -v -timeout=300s
# Repeat with go test -race and -count=3.
```

The default offline test skips this gate unless its explicit GPU flag is set. Whole-tree CPU race (132 package results), `go vet ./...`, `go build ./...`, docs/layout checks (456 Markdown files, zero broken links), and linux/arm64 and linux/riscv64 test cross-builds passed. Those foreign builds were not native runtime tests.

Per blocked call the NVIDIA counters recorded 3 kernel launches, 6,684,672 host-to-device bytes, 786,432 device-to-host bytes, and 9 matched allocations/frees totalling 7,471,104 bytes. The unblocked candidate used one launch and 6,946,816 allocated/freed bytes, but failed numerically. These are transfer and allocation volumes, not peak or retained VRAM; idle device memory returned to 33 MiB. The extra launches, transfers and host packing were not benchmarked against production CPU convolution, so there is no speed or admission claim. Neither the 1.6e-6 CPU end-to-end waveform gate nor any CPU fallback has changed. GPU execution of the rest of the decoder, cancellation, concurrent requests and sustained device memory remain unqualified.
