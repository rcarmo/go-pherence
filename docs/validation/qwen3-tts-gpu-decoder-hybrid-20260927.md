# Qwen3-TTS two-convolution decoder GPU diagnostic

A fixed 64-frame decoder run used the NVIDIA backend for pre-convolution and decoder-initial convolution, with the rest of the decoder on CPU. Its 122,880-sample waveform differed from the independent Rust/Candle reference by at most **1.28487591e-6**, below the unchanged **1.6e-6** gate. Production synthesis still uses the CPU decoder.

## Boundary checks

The opt-in `TestDecoderTwoGPUConvsPinnedWaveform` loads the approved `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` checkpoint at revision `85e237c12c027371202489a0ec509ded67b5e4b5`. It checks the decoder weight/config hashes and the independent `TrevorS/qwen3-tts-rs` revision `711ceee07cad92673f86de8997bdf54c30caa49f` 64-frame code, stage-tensor and waveform fixtures before use. The diagnostic calls the existing CPU decoder through an internal, callback-only test seam; nil callbacks select the unchanged CPU convolutions. The test runs both GPU convolutions on the live data in one decode and never substitutes a saved Rust intermediate.

| Tensor versus pinned Rust F32 | Maximum absolute error | Different F32 bits |
|---|---:|---:|
| Live pre-convolution input, 32,768 values | 0 | 0 |
| GPU pre-convolution output, 65,536 values | 0 | 0 |
| Live decoder-initial input, 262,144 values | 3.02791595e-5 | 253,751 |
| GPU decoder-initial output, 393,216 values | 6.7949295e-6 | 366,996 |

Each GPU output also matched the existing CPU convolution **bitwise on its live input**. The second convolution's pinned-Rust difference includes upstream CPU drift. Its input and output have separate regression bounds of `4e-5` and `1e-5`; neither is a bitwise Rust gate. The first boundary retains bitwise Rust checks. The independently pinned final waveform passed the existing `1.6e-6` maximum-absolute-error gate; that does not establish bitwise end-to-end parity.

## Allocation and execution scope

The request-local test helper reserves 51 GPU buffers before decoding, uploads prepared weights, then runs three 512-wide pre-convolution and fourteen 512-wide decoder-initial SGEMMs. During the measured decode it checks **17 launches, 17 activation uploads, 17 partial downloads and zero GPU buffer allocations/frees**. After teardown, allocation and free counters each total **51 buffers / 80,871,424 bytes**. The GPU outputs are copied into owned Go slices. CUDA `MemInfo` free-memory snapshots from a passing run were 12,332,695,552 bytes before setup, 12,236,226,560 after setup, and 12,332,695,552 after teardown. Other runs returned to the same before/after value. During one repeat run, GPU-wide `nvidia-smi` polling at a requested 100 ms interval returned **289 samples**, from **33 MiB idle to 239 MiB observed high-water**. The series is `/workspace/tmp/qwen3tts-gpu-outputproj-20260927/hybrid-vram-sampled.csv`. This is an observed device-wide value, not per-request attribution or an exact in-flight peak: sampling can miss shorter spikes, and the before/after snapshots and balanced counters do not establish sustained retention. Host allocations were not measured. The test has no cancellation, multiple-request, full-request-latency or GPU Talker/CodePredictor gate.

```sh
GOMAXPROCS=6 GO_PHERENCE_QWEN3TTS_GPU_TEST=1 \
  GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR=/dev/shm/qwen3tts-0b6-customvoice \
  GO_PHERENCE_QWEN3TTS_SENTENCE64_ORACLE_DIR=/workspace/tmp/qwen3tts-cap64-20260926/reference1 \
  go test ./model/qwen3tts -run '^TestDecoderTwoGPUConvsPinnedWaveform$' -count=3 -v -timeout=300s
```

Three ordinary repeat runs and the isolated race run reproduced the same boundary and waveform results. The released CPU 64-frame sentence test also passed with maximum waveform error `1.2848759070038795e-6`. Whole-tree CPU race (`GO_PHERENCE_DISABLE_NVIDIA=1 go test -race ./...`), vet, build, docs/layout and linux/arm64 plus linux/riscv64 test cross-builds passed. An unrestricted whole-tree GPU-visible race run failed only four unrelated local DiffusionGemma FP8 tests because checkpoint index metadata points to shards removed in the earlier free-space cleanup; no DiffusionGemma weights or tests were changed. Raw logs are under `/workspace/tmp/qwen3tts-gpu-outputproj-20260927/hybrid-*.log`.

The isolated [resident pre-convolution](qwen3-tts-gpu-preconv-resident-20260927.md) and [decoder-initial convolution](qwen3-tts-gpu-decoder-initconv-resident-20260927.md) benchmarks time one stage each. This diagnostic measures correctness across two GPU stages in one decoder waveform; it is not a full-request speed measurement. Production GPU admission still needs the remaining decoder and Talker/CodePredictor stages, cancellation, concurrent ownership, actual in-flight peak and retained VRAM, and full-request latency. No service state or GPU dispatch changed.
