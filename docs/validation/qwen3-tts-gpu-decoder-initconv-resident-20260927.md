# Qwen3-TTS decoder-initial convolution resident-buffer trial

A benchmark-only, request-local NVIDIA buffer trial ran the released 256-position decoder-initial convolution faster than the existing CPU stage on this host. It matched all 393,216 pinned Rust/Candle F32 outputs bitwise after warm-up and after the timed calls. Production synthesis and waveform decoding remain CPU-only.

## Inputs and timing boundary

- Go baseline `80d48809c7f3a88413be5d294c721794ad2f2dea`, linux/amd64, Go 1.26.3, i7-12700, `GOMAXPROCS=6`. NVIDIA GeForce RTX 3060, driver 610.57.04; idle VRAM 33 MiB.
- Approved `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` revision `85e237c12c027371202489a0ec509ded67b5e4b5`. The opt-in benchmark hashes the decoder config/model and independently traced full `upsample1` input (`5884389f6891c21e173029dab67d3d138a3dd8dc4b76611cc32b0abdf854f372`) and `decoderinit` output (`2dd3e28511ffb314f05ab258df7ed8b69d8663b29a33f04fe0dc9ddb7c5afc85`) before loading. The CPU oracle is `TrevorS/qwen3-tts-rs` revision `711ceee07cad92673f86de8997bdf54c30caa49f`, Candle 0.9.2; its trace generation and frozen waveform are documented in the [stage parity report](qwen3-tts-gpu-decoder-initconv-20260927.md).
- Unit of work: input `[1024,256]`, released 7-tap weights `[1536,1024,7]`, output `[1536,256]`. CPU calls the existing `decoderConv1D.forward` and owns the output. GPU setup packs/uploads 14 immutable 512-wide F32 weight blocks and reserves 14 activation/weight/output device-buffer triples. Timed GPU calls do host im2col, 14 activation uploads, 14 SGEMMs, 14 partial downloads, blocked F32 accumulation and CPU bias/layout conversion, then copy an owned output. Checkpoint loading, GPU buffer allocation, immutable weight preparation/uploads and kernel loading are outside stage timing.

```sh
GOMAXPROCS=6 GO_PHERENCE_QWEN3TTS_GPU_TEST=1 \
  GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR=/dev/shm/qwen3tts-0b6-customvoice \
  go test ./model/qwen3tts -run '^$' \
  -bench '^BenchmarkDecoderInitConvResidentProbe$' \
  -benchmem -benchtime=30x -count=10 -v -timeout=300s
```

Raw samples: `/workspace/tmp/qwen3tts-gpu-outputproj-20260927/initconv-resident-30x10.log`. Across ten 30-call samples, CPU stage median was **56,424,381 ns/op**, 4,947,968 B/op and 4 allocs/op. The GPU resident owned-output stage median was **18,707,643 ns/op**, 1,579,516 B/op and 351 allocs/op: **3.02× faster** by stage median, while allocating many more small Go objects. Samples ran as a CPU block followed by a GPU block, not interleaved; the comparison covers one host and shape. It is not an end-to-end synthesis throughput or general GPU speed claim.

One observed setup took **38.97 ms** to allocate the buffers and pack/upload weights, excluding checkpoint and kernel load. The backend counters across setup/teardown showed **73,400,320 bytes** allocated and freed, including **44,040,192 bytes** of weight uploads. Each timed GPU call checks 14 launches, **7,340,032 activation-upload bytes**, **22,020,096 partial-download bytes**, and zero per-call device allocation/free. Ten benchmark instances passed full bitwise reference checks, and device frees matched allocations at teardown. Whole-tree CPU race (132 package results), `go vet ./...`, `go build ./...`, docs/layout checks (460 Markdown files, zero broken links), and linux/arm64 and linux/riscv64 test cross-builds passed. The default offline benchmark skips when the explicit GPU flag is unset; cross-builds are not native foreign runtime checks. Idle GPU memory returned to 33 MiB; in-flight peak VRAM, concurrent requests, retained device memory over long reuse, cancellation and failure recovery were not measured.

The existing [host-staged diagnostic](qwen3-tts-gpu-decoder-initconv-20260927.md) proves one-stage numerical parity only. This trial improves that stage's warm timing by hoisting fixed weights and device buffers, but leaves CPU-to-GPU-to-CPU transfers and a large request-local workspace. Production GPU promotion requires the rest of the decoder, Talker/CodePredictor, pinned end-to-end codes and waveform under unchanged gates, owned concurrent sessions, cancellation, VRAM admission and full-request latency. No dispatch or service state changed here.
