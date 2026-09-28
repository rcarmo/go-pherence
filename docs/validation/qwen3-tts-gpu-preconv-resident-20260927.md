# Qwen3-TTS pre-convolution resident-buffer trial

A request-local resident-buffer diagnostic ran the released 64-frame decoder pre-convolution faster than the existing CPU stage on this host. It remains a benchmark-only trial: production synthesis still uses CPU, and no GPU waveform or full-request performance gate has passed.

## Workload and scope

- Go baseline `cabd28740ea85f1312eae56ba1f4b65bad81a7e5` on linux/amd64, Go 1.26.3, Intel i7-12700, `GOMAXPROCS=6`. NVIDIA GeForce RTX 3060, driver 610.57.04; idle VRAM 33 MiB.
- Approved `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` revision `85e237c12c027371202489a0ec509ded67b5e4b5`. The opt-in benchmark verifies decoder config and weight SHA-256 and the full pinned Rust 64-frame input/output trace hashes before loading. Warm-up and final output compare all 65,536 F32 values bitwise with the independent Rust output `75c0a7feda1be45023577e1f2675f909b75351fd5bc977978e385597a4703685`.
- Unit: one `[512,64]` input, `[1024,512,3]` weight, `[1024,64]` output pre-convolution. The existing CPU `decoderConv1D.forward` returns an owned output. The GPU trial preloads three 512-wide immutable F32 weight blocks, reserves three activation/output device pairs and request-local host scratch, then times im2col, 3 activation uploads, 3 SGEMM launches, 3 partial downloads, F32 blocked accumulation, bias/layout conversion. `GPUResidentOwned` also copies the final output into an owned slice. Weight packing, device allocations/uploads and checkpoint loading are outside those per-call timings. No GPU output tensor is returned by a production API.

```sh
GOMAXPROCS=6 GO_PHERENCE_QWEN3TTS_GPU_TEST=1 \
  GO_PHERENCE_QWEN3TTS_0B6_CUSTOMVOICE_DIR=/dev/shm/qwen3tts-0b6-customvoice \
  go test ./model/qwen3tts -run '^$' \
  -bench '^BenchmarkDecoderPreConvResidentProbe$' \
  -benchmem -benchtime=100x -count=10 -v -timeout=300s
```

The ten 100-call samples are in `/workspace/tmp/qwen3tts-gpu-outputproj-20260927/resident-owned-100x10.log`. Median CPU stage: **2,260,377 ns/op**, 1,277,953 B/op and 4 allocs/op. Median `GPUResident`: **830,355 ns/op**, 1,425–1,429 B/op and 75 allocs/op. Median `GPUResidentOwned`: **872,706 ns/op**, 263,593 B/op and 76 allocs/op, about **2.59× faster** than CPU by stage median. Samples ran in CPU, GPU scratch, then GPU owned blocks rather than interleaved; this is one host/shape, not an end-to-end speed or general latency claim. The earlier [fully host-staged path](qwen3-tts-gpu-preconv-benchmark-20260927.md) was 4.40× slower than CPU. Moving immutable packing and GPU allocation outside the operation accounts for much of the difference; the timed GPU call still moves activations and partial outputs across PCIe.

One measured setup took **7.415 ms** for nine device buffers, weight packing and three weight uploads, excluding model load and kernel initialization. The backend counters recorded 7,471,104 bytes allocated and freed, with 6,291,456 bytes of F32 weight uploads. In the `GPUResident` sub-benchmark, the test checks per-call 3 launches, 393,216 activation-upload bytes, 786,432 partial-download bytes, and no per-call device allocation/free. Ten benchmark instances passed full bitwise reference checks after warm calls, and device frees matched allocations at teardown. Whole-tree CPU race (132 package results), `go vet ./...`, `go build ./...`, docs/layout checks (458 Markdown files, zero broken links), and linux/arm64 plus linux/riscv64 test cross-builds passed. The offline default benchmark skips without the explicit GPU flag; cross-builds are not native runtime qualification. Idle `nvidia-smi` returned to 33 MiB; in-flight/retained peak device memory and concurrent requests were not measured. CPU and GPU memory ownership differ; the benchmark's request-local scratch and one-shape reservation are not production lifecycle, cancellation or error-recovery qualification.

The production path remains CPU. Before any GPU promotion, the other decoder stages, Talker/CodePredictor, released end-to-end codes and waveform under their unchanged gates, request cancellation, concurrent ownership, peak/retained VRAM, and full-request latency need independent qualification. No service change, GPU reset or other model GPU execution was made in this trial.
