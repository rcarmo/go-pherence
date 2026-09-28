# Qwen3-TTS 64-frame decoder-initial convolution on NVIDIA

The released 7-tap Decoder12Hz initial convolution matched its pinned Rust/Candle F32 output and native Go CPU output bitwise in a bounded NVIDIA SGEMM diagnostic. Production synthesis, waveform decoding and runtime status remain CPU-only.

## Reference and arithmetic

- Starting Go revision `3c87468c0ef0b076b26d5dc68bade263aa885498`, clean before the test. GeForce RTX 3060 12 GiB, driver 610.57.04; Go 1.26.3 on linux/amd64, i7-12700, `GOMAXPROCS=6`.
- Checkpoint: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` revision `85e237c12c027371202489a0ec509ded67b5e4b5`. The test hashes `speech_tokenizer/config.json` and `model.safetensors` before loading. The existing sentence64 fixture hashes the independent probe, observation and all 1,024 codes.
- Independent oracle: `TrevorS/qwen3-tts-rs` revision `711ceee07cad92673f86de8997bdf54c30caa49f`, Candle 0.9.2, CPU F32. The instrumented decoder source hashes to `a2edf2ad569200dc64d2bc07ae8d95d25c32cb20130435d3ca036f579d74c761` and decoder-only probe to `4ea10f1b8c96481d3d7a83b9c197a2c2972b200be8dc99693cd7307eb98391f6`. Its CPU-only full-trace run over the frozen sentence64 codes reproduced the frozen waveform SHA-256 `e0ef875fbe94c1eb923b78c28fc72e5bca87af0b28985db07d3e8e641dcb02fb`.
- Committed full F32 channel-major pre-upsample output `[1024,256]`: SHA-256 `5884389f6891c21e173029dab67d3d138a3dd8dc4b76611cc32b0abdf854f372`. Full decoder-initial output `[1536,256]`: SHA-256 `2dd3e28511ffb314f05ab258df7ed8b69d8663b29a33f04fe0dc9ddb7c5afc85`.

The test loads the released 7-tap `[1536,1024,7]` weights and bias, checks the full Go CPU output against all 393,216 independent F32 values, then builds causal im2col for `[256,7168]`. Fourteen 512-wide GPU SGEMM outputs are downloaded and accumulated in Candle block order; the bias is added on CPU. This uses host staging per block, not a device-resident convolution.

```sh
GOMAXPROCS=6 GO_PHERENCE_QWEN3TTS_GPU_TEST=1 \
  GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR=/dev/shm/qwen3tts-0b6-customvoice \
  go test ./model/qwen3tts -run '^TestDecoderInitConvPinnedGPU$' \
  -count=10 -v -timeout=300s
# Repeat with go test -race and -count=3.
```

Ten ordinary and three race repetitions passed. Whole-tree CPU race (132 package results), `go vet ./...`, `go build ./...`, docs/layout checks (459 Markdown files, zero broken links), and linux/arm64 and linux/riscv64 test cross-builds passed. The default offline test skips when the explicit GPU flag is unset; cross-builds are not native foreign runtime tests. The maximum absolute error across all Rust/GPU and native CPU/GPU outputs was zero; the unchanged diagnostic threshold was `1e-6`. Per call the NVIDIA counters reported 14 launches, 51,380,224 host-to-device bytes, 22,020,096 device-to-host bytes and 42 matched allocations/frees totalling 73,400,320 bytes. Idle GPU memory returned to 33 MiB. These are cumulative transfer/allocation volumes, not peak VRAM or speed measurements. Full decoder GPU execution, waveform parity, concurrency, cancellation, retained VRAM and end-to-end latency remain unqualified. The production CPU waveform gate remains `1.6e-6` and has not been widened.
