# Qwen3-TTS 64-frame input projection on NVIDIA

The 0.6B CustomVoice Decoder12Hz input projection passed an opt-in NVIDIA F32 diagnostic on an RTX 3060. The decoder and capped synthesis APIs still execute on CPU; no production GPU dispatch was added.

## Inputs and test boundary

- Repository baseline: `f7c253543cb7b46b9596d1460fc696a6a5ecb1cc` (clean before this test). Go 1.26.3, linux/amd64, i7-12700, `GOMAXPROCS=6`. GPU: GeForce RTX 3060 12 GiB, driver 610.57.04.
- Checkpoint: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` revision `85e237c12c027371202489a0ec509ded67b5e4b5`. The opt-in test hashes `speech_tokenizer/config.json` and `speech_tokenizer/model.safetensors` before loading. The existing 64-frame fixture pins the independent Rust probe, observation and all 1,024 codes.
- Independent oracle: `TrevorS/qwen3-tts-rs` revision `711ceee07cad92673f86de8997bdf54c30caa49f`, Candle 0.9.2, CPU F32. The diagnostic trace hooks are in the local reference checkout's `src/models/codec/decoder_12hz.rs` (SHA-256 `a2edf2ad569200dc64d2bc07ae8d95d25c32cb20130435d3ca036f579d74c761`). Each committed trace contains 2,048 evenly spaced little-endian F32 samples, at indexes `i*(n-1)/2047` for a stage with `n` elements. SHA-256: `preconv.f32le` `62d2766cb5f9bae22af5783268953e4b5d824ed54c885dc775157935a7b433cf`; `inputproj.f32le` `65ab117cc051fc6bf0817590a75d605acf86d2fe02a854944c10caf209b2979a`. The reference trace run wrote a waveform with the same SHA-256 as the frozen 64-frame waveform, `e0ef875fbe94c1eb923b78c28fc72e5bca87af0b28985db07d3e8e641dcb02fb`.
- Scope: CPU codebook projection and causal pre-convolution produce a channel-major `[1024,64]` tensor; `channelToTime` makes the `[64,1024]` SGEMM input. The owned decoder input-projection weights are transposed from `[512,1024]` to `[1024,512]`. NVIDIA `SgemmHost` computes `[64,512]`; the existing F32 bias is added on CPU. The test compares all 32,768 GPU values against native CPU and 2,048 sampled values against independent Rust traces. No waveform is synthesized on the GPU.

## Gates and measurements

```sh
GOMAXPROCS=6 GO_PHERENCE_QWEN3TTS_GPU_TEST=1 \
  GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR=/dev/shm/qwen3tts-0b6-customvoice \
  go test ./model/qwen3tts -run '^TestDecoderInputProjectionPinnedGPU$' \
  -count=10 -v -timeout=300s

# Same command with `go test -race` and `-count=3` for the released race gate.
```

Ten ordinary and three race runs passed. The ordinary whole-tree CPU race, `go vet ./...`, `go build ./...`, docs/layout checks (454 Markdown files, zero broken links) and linux/arm64 and linux/riscv64 test cross-builds passed. The unchanged opt-in CPU 64-frame end-to-end test also passed: all 1,024 codes exact and maximum waveform error `1.2848759070038795e-6` below its `1.6e-6` gate. An unset GPU flag skips the stage test; a missing released checkpoint fails the requested hardware gate. Both sampled CPU stages matched Rust bitwise (maximum absolute error zero). GPU input-projection samples differed from Rust by at most `2.98023224e-7`; the maximum difference over the full GPU/CPU output was `3.57627869e-7`. The diagnostic GPU threshold is `1e-6` for both comparisons. It does not replace or widen the existing end-to-end `1.6e-6` waveform gate.

The backend counters per call reported one kernel launch, 2,359,296 host-to-device bytes, 131,072 device-to-host bytes, and three matched allocations/frees totalling 2,490,368 bytes. This is transfer and allocation volume, not measured peak VRAM or process RSS. Idle `nvidia-smi` showed 33 MiB used before and after testing; the test does not measure in-flight VRAM. The standalone NVIDIA SGEMM live-variant test also passed before this slice.

The existing CPU decoder path, status flags and synthesis APIs are unchanged. Transformer layers, decoder convolutions/upsampling, GPU output waveform, concurrency, lifecycle under cancellation and sustained VRAM admission remain unqualified. A stage-level numerical pass alone does not justify GPU promotion or a speed claim. Keep the independent CPU 64-frame code/waveform gates and fallback intact when examining further stages.
