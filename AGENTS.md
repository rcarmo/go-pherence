# AGENTS.md — go-pherence

Ground rules and conventions for AI agents working on this repository.

## Repository structure

```
go-pherence/
├── backends/           # Hardware-specific compute backends
│   ├── ggmlcompute/    # GGML graph compute
│   ├── ggmlgraph/      # GGML graph builder
│   ├── ggmlquant/      # GGML quantization types
│   ├── k3/             # SpacemiT K3 SoC backend docs/config
│   ├── nvidia/          # NVIDIA CUDA/PTX kernels
│   ├── simd/           # Cross-platform SIMD runtime + kernels
│   │   ├── kernels/    # Reference scalar + SIMD kernels
│   │   └── runtime/    # Platform-dispatched SIMD surface (dot, saxpy, softmax, etc.)
│   └── spacemit/       # SpacemiT K3-specific backends
│       ├── ime2/       # IME2 A100 kernel wrappers + Q80x32 packing
│       ├── inference/  # SpacemiT inference abstractions
│       ├── aicpu/   # K3 AI core engine
│       │   └── aipool/ # A100 worker pool + Q80x32 GEMM dispatch
│       ├── rvv/        # RVV vector kernels (FP16, SiLU, FastExp, dot, GEMM)
│       └── tcm/        # TCM (Tightly Coupled Memory) access
├── cmd/                # CLI entry points
│   ├── audio/          # Whisper and audio tools
│   ├── image/          # Ideogram4, VAE probes, image tools
│   ├── k3/             # K3-specific benchmarks
│   ├── llm/            # LLM speculative decoding tools
│   └── checkpoints/         # Model inspection tools
├── docs/               # Architecture docs, supported models, GPU options
├── gpu/                # GPU management utilities
├── half/               # FP16/BF16 conversion
├── internal/           # Checked arithmetic, internal utilities
├── loader/             # Model loaders (GGUF, safetensors, audio, config, tokenizer)
├── model/              # All model source + shared decoder runtime
│   ├── diffusiongemma/ # Block-diffusion text + bounded vision work
│   ├── ideogram4/      # Ideogram v4 image generation
│   ├── hunyuan3d/      # Hunyuan 3D
│   ├── qwen/           # Qwen native models
│   ├── bert/           # BERT/GTE encoders
│   ├── whisper/        # Whisper speech recognition
│   ├── speaker/        # Speaker diarization + Community-1
│   └── omnivoice/      # Native speech synthesis
├── checkpoints/        # Downloaded weights/config/tokenizers (git-ignored)
├── prompts/            # Prompt templates
├── research/           # Research prototypes (NPU whisper, etc.)
└── testdata/           # Test fixtures
```

## Tooling

### Required

- **Go 1.24+** — the module uses recent Go features.
- **`go vet`** — run before every commit. Do not commit code that fails vet.
- **`go test`** — run affected packages before committing. `make host-test` scans the repository with GPU defaults disabled; when touching `backends/`, also run affected backend tests. Respect GOOS/GOARCH build constraints, never force foreign code into host checks.
- **`gofmt -w`** — format all modified `.go` files before committing.
- **`make model-layout-check`** — run after editing source paths, checkpoint defaults, generators, templates or build scaffolding. It is included in `make docs-check` and runs first in `make host-check`; the source/path guard also runs under ordinary `go test ./...`. Do not disable the guard to accommodate new `models/` paths.

### Recommended

- **`gopls`** — install with `go install golang.org/x/tools/gopls@latest`. Use it for:
  - Finding all references to a symbol before renaming or moving it.
  - Checking which packages import a function before changing its signature.
  - Navigating build-tagged files (`_riscv64.go`, `_other.go`).
  - Mechanical refactors like extracting interfaces or inlining helpers.
- **`go build ./...` / `make host-build`** — verify the host tree compiles after cross-cutting changes. Use `make spacemit-cross-compile` or explicit `GOOS`/`GOARCH` builds for other targets; use `go test -c`, never `go test`, for foreign test binaries. Cross-compilation is not runtime validation.

### Remote K3 development

When working on the Milk-V/K3 board via SSH:
- Set `HOME=/home/me TMPDIR=/tmp GOCACHE=/home/me/.cache/go-build GOMODCACHE=/home/me/go/pkg/mod` explicitly — SSH-backed tool environments may not inherit the correct home directory.
- Use `nohup` for long-running generation commands; the board has 31 GB RAM and no swap, so aggressive memory use can trigger the OOM killer or reboot.
- After installing packages, use `--user --break-system-packages` for pip since `python3-venv` may not be available and `sudo` may require a TTY.

## Ground rules

### Before editing

1. **Read the file first.** Never edit blind. Understand what is there before changing it.
2. **Check build tags.** Many files have `_riscv64.go` / `_other.go` pairs. If you edit one, check the other.
3. **Search for callers.** Before changing a function signature, grep or use `gopls references` to find all call sites.
4. **Check for tests.** If a `_test.go` exists for the file you are editing, run it after changes.

### Making changes

5. **Prefer editing over rewriting.** Do not rewrite whole files unless you understand every function in them. I have lost working code to blind rewrites.
6. **Do not remove functions you didn't add** unless you have verified zero callers with `gopls` or `grep -R`.
7. **Preserve build-tag stubs.** Every `_riscv64.go` function must have a matching stub in `_other.go` (or vice versa). If you add a new function to one, add it to both.
8. **Keep imports correct.** When adding imports to build-tagged files, verify the import is available on all target platforms. Use `gofmt -w` to sort imports.
9. **Preserve numerical contracts.** Do not widen existing tolerances to make an optimization pass. Floating-point reduction order, rounding, saturation, quantization thresholds and STE gradients are part of the contract. If a deliberate approximation cannot meet it, keep the reference/default path intact and propose a separately named opt-in mode with measured error, tests and explicit approval. A comment or build tag alone does not justify weakening parity.

### Committing

10. **Run `gofmt -w` on all modified files.**
11. **Run `go test` on affected packages.** At minimum: the package you edited plus any package that imports it.
12. **Run `go vet` on affected packages.**
13. **Run `go build ./...`** to verify the full tree compiles.
14. **Write clear commit messages.** First line is a short summary (`package: what changed`). Body explains why, not just what.
15. **Do not commit temporary/debug code** (print statements, hardcoded paths) unless guarded by an env flag like `GO_PHERENCE_IDEOGRAM4_TIMING`.

### Architecture conventions

16. **Model-specific code stays in `model/`.** `checkpoints/` is git-ignored data only. Do not put model-specific logic in `backends/` or revive `models/` source packages.
17. **Reusable kernels go in `backends/`.** If a kernel (SiLU, FastExp, Q8 packing) is useful to multiple models, put it in the appropriate backend package.
18. **Build tags for platform-specific code.** Use architecture constraints for ISA kernels and `linux && riscv64` for K3 OS/device execution. Keep portable packing/reference tests runnable. Do not manufacture scalar emulation or omit whole backend folders just to make repository checks pass; retain genuine scalar fallbacks and explicit unsupported stubs.
19. **The `half` package** owns FP16/BF16 conversion. Do not duplicate it.
20. **The `loader` package** owns model file I/O (GGUF, safetensors, audio, tokenizers). Do not add model loading logic directly in `model/`.

### K3/SpacemiT-specific

21. **A100 cores require `/proc/set_ai_thread` registration.** Never use plain `taskset` or `runtime.LockOSThread` alone for A100 access. Use the `aipool` package.
22. **TCM is 3 MB of on-chip SRAM** (8 × 384 KB blocks). It is useful for weight/activation staging but not for large buffers. The `tcm` package manages it.
23. **K3 mode disables all NVIDIA paths.** If you add a new GPU-gated path, check `gpuDisabledByK3()` or use the existing `gpuXxxEnabled()` predicates.
24. **31 GB RAM, no swap.** Be conscious of peak memory. Release large temporaries before allocating new ones. Consider `runtime.GC()` / `debug.FreeOSMemory()` only at measured, coarse phase boundaries where needed for admission; never inside token/layer/training-step loops or as a substitute for fixing allocations. Account for the pause in end-to-end measurements.
25. **Row-scale Q8** is the preferred A100 quantization contract for quality-sensitive paths (matches the native int8 quantization used by Whisper and Ideogram).

## Performance and allocation policy

Numerical parity is necessary, but it is not evidence that an implementation is efficient. Treat allocation reduction, peak/retained memory, CPU time and native execution as separate acceptance dimensions. One profiling pass does not close performance work: profile again after each material change, since removing one cost exposes the next.

Keep model weights immutable, I/O in loaders, and reusable kernels in backends. Optimize measured costs rather than pursuing a blanket rewrite. Do not trade away ownership, bounded memory, cancellation, transactional failure, numerical fidelity or supported fallbacks to improve a benchmark.

### Measurement contract

Before editing a hot path, record a reproducible baseline: Git revision and working-tree patch, Go/toolchain version, GOOS/GOARCH, CPU/ISA, thread settings, model/checkpoint pin, numerics, tensor shapes, sequence/batch lengths, and cold versus warm state. Preserve baseline binaries outside frozen evaluation paths. Use identical fixtures/options for before/after comparisons.

* Cover both tiny deterministic fixtures and a locally available representative model. For each supported model generation, include loading/preparation, full-prefix inference, prefill, one-token decode, multiple-token decode, reset/reuse, heads, loss/backward, full-trunk training and adapter training where implemented. Include decoded/packed and supported quantization modes separately. Explicitly list unavailable or deferred workloads.
* Use `b.ReportAllocs()` or `-benchmem`, setup outside the timed region and a realistic warm-up. Report `ns/op`, `B/op` and `allocs/op`; define whether an operation is a token, prompt, batch or training step. Measure initialization separately, and add an end-to-end benchmark so moving work out of the timed loop cannot masquerade as a saving.
* Run at least five independent benchmark samples (prefer ten for small/noisy changes) and compare distributions with `benchstat`. Keep the host/thread configuration comparable. Separate concurrent/busy-host measurements from controlled runs. Do not accept a single best run as a baseline.
* Capture CPU profiles, `alloc_space`, `alloc_objects`, `inuse_space` and `inuse_objects` profiles for the relevant phases. Inspect both flat and cumulative callers; drill into source/assembly for dominant costs. Measure retained heap after repeated reuse and peak process RSS separately. Logical budgets, cumulative allocation volume, live heap and RSS are different numbers.
* Use `-memprofilerate=1` for a bounded allocation-attribution run when sampling would miss small churn. Run timing benchmarks and CPU profiles separately at normal profiling settings: exhaustive allocation profiling can dominate execution time. Never use those distorted times as speed evidence.
* Use escape analysis (`go test -gcflags='-m=2'` on selected packages), execution traces, GC statistics, mutex/block profiles or platform counters when profiles suggest escapes, scheduler delays, synchronization, bandwidth or GC pressure. These diagnostics have overhead; enable them selectively and label the run. Avoid dumping unbounded compiler/profile output into reports.
* After changes, repeat the same profiles and inspect the new dominant callers. Account for allocations displaced into caches, initialization, model copies, retained capacity or another package. An improved `allocs/op` with larger memory retention or worse latency needs an explicit tradeoff, not a success claim.
* Preserve commands, benchmark samples and profiles under a bounded workspace evidence directory; summarize results and unresolved costs in `docs/validation/`. Attach requested reports/profile bundles for review. Do not commit huge profile binaries or temporary benchmark/debug code. Missing independent review is a limitation, not implied approval.

Typical commands (replace the package, benchmark and output paths with the actual workload):

```sh
GO_PHERENCE_DISABLE_NVIDIA=1 go test ./model/needle -run '^$' \
  -bench '^BenchmarkNeedle' -benchmem -count=10 > before.txt
# Repeat on the changed tree with identical options, then:
benchstat before.txt after.txt

# Separate attribution runs; select a bounded representative benchmark.
GO_PHERENCE_DISABLE_NVIDIA=1 go test ./model/needle -run '^$' \
  -bench '^BenchmarkNeedleHeads$' -benchtime=20x -memprofile=alloc.pprof -memprofilerate=1
GO_PHERENCE_DISABLE_NVIDIA=1 go test ./model/needle -run '^$' \
  -bench '^BenchmarkNeedleHeads$' -benchtime=3s -cpuprofile=cpu.pprof
go tool pprof -sample_index=alloc_space -top alloc.pprof
go tool pprof -sample_index=alloc_objects -top alloc.pprof
go tool pprof -sample_index=inuse_space -top alloc.pprof
go tool pprof -sample_index=inuse_objects -top alloc.pprof
```

### Default coverage and allocation targets

These are engineering targets for new/changed code, not claims about current repository coverage or every existing model. Establish and report the actual baseline. If a target cannot be met, quantify the gap and obtain explicit agreement before treating that optimization item as complete; never silently lower a target, exclude difficult code or mark unrun checks passed.

| Area | Typical target and acceptance rule |
|---|---|
| Reusable numerical kernels and `Into`/caller-buffer APIs | **0 heap allocations, 0 B/op** per steady-state call, with scratch supplied or preallocated. Add `testing.AllocsPerRun` assertions after warm-up as well as benchmarks. |
| Inner loops | **0 avoidable per-element, per-head, per-layer or per-token allocations.** Reuse bounded session-local scratch; no string formatting, maps, closures or shape/index construction in inner loops unless a profile and documented contract justify them. |
| Warm decoder/session | Aim for **0 internal allocations per token**. An API returning owned logits may require **one output allocation**; count and label it, and preserve ownership. Target allocation count independent of layer count/sequence position, with no unbounded growth across `Reset`/reuse. These are targets, not permission to return aliased scratch. |
| Whole-request inference | Separate model preparation, cache/workspace reservation and returned outputs. Set a workload-specific measured ceiling for bytes and allocations; no unexplained regression. A lower count alone is not sufficient if bytes or peak memory increase. |
| Training/backward and loaders | No universal zero-allocation target: tape, gradients, immutable updated models and materialized weights have real costs. Budget these separately, eliminate repeated temporary churn and duplicate full tensors, and test peak admission. Aim to remove **25–50% of a selected avoidable allocation hotspot** per substantive pass where the profile supports it; this is not a mandatory whole-model speedup or a reason to stop profiling. |
| Changed portable Go code | Aim for **at least 90% statement coverage** of the changed package or a clearly scoped changed-code report; **95%+** for new numerical kernels, parsers and admission/ownership helpers. Preserve existing coverage and explain unreachable/platform-only branches. Run `go test -coverprofile`/`go tool cover -func`; report the scope, not an inflated repository-wide percentage. |
| Behavior and backend coverage | Exercise **every changed public contract and error category**, each supported dispatch/scalar path, shape/tail class and dtype/mode. Assembly is not measured by Go statement coverage: native differential tests and dispatch evidence are required. |
| Repository model-coverage inventory | Keep the existing **90% model-inventory gate** (`make test-model-coverage`) separate from Go test coverage, numerical parity, native execution and performance qualification. Passing one does not establish the others. |

Allocation budgets should normally be exact for small kernels and bounded ceilings for whole-model workloads. Preserve the measured baseline in regression tests/benchmarks; don't assert unstable timings in unit tests. A material improvement claim must survive repeated measurements without an unexplained regression in other accepted dimensions.

### SIMD and scalar acceptance gates

* Keep a genuine scalar/reference implementation and differential tests. SIMD belongs in the backend, not duplicated in model code. Preserve build-tag counterparts and explicit unsupported stubs for genuinely unavailable device operations; do not hide packages or emulate an unsupported device to make tests green.
* Prove dispatch: test the native optimized path and force the relevant feature gates off for its fallback. `GODEBUG=cpu.all=off` is a useful check, **not proof that all assembly is disabled**; verify which dispatch flags/routes the package actually uses. Exercise mixed feature combinations and minimum-ISA targets where applicable. No illegal instructions on supported lower-ISA hosts.
* Test zero/one length, vector-width-minus/plus-one, all tail/remainder classes, awkward/odd dimensions, strides, transposes, unaligned slices and large bounded shapes. Verify bounds, checked size arithmetic and overlap rules; invalid inputs must leave destinations/state unchanged where promised. Use guard/sentinel storage around outputs.
* Test signed zero, subnormals, NaNs/Infs, large finite values, zero/tiny norms, reduction cancellation, quantization ties, saturation and FP16/BF16 rounding according to the operation's documented contract. Compare forward, loss and gradients for training changes, plus full-prefix versus cached behavior for decoder changes. Do not widen established tolerances.
* Run actual native correctness tests on Intel/amd64 and ARM64 for shared CPU kernels; run native RISC-V/device tests when those paths change and hardware is authorized/available. Cross-compile all affected targets with `go build` and `go test -c`; never describe foreign compilation as execution. If native hardware is unavailable, leave its gate explicitly open rather than claiming full cross-platform qualification.
* Benchmark optimized and scalar/reference paths across representative sizes, including dispatch thresholds. Require a demonstrated advantage in the intended workload or document why a correctness/reference path is intentionally slower. Check allocation, bandwidth, packing/setup and tail costs as well as peak kernel throughput. Tiny-matrix speed does not establish released-model speed.
* Assembly is not fully instrumented by the Go race detector. Pair race tests with native bounds/alias/ownership/concurrency tests. Verify deterministic behavior or explain the permitted numerical variation; reduced allocations must not introduce shared mutable scratch or request cross-talk.

### Optimization checklist

Use this checklist for each performance pass. Mark items as verified, open, or not applicable with a reason; it is an audit checklist, not a claim that every optimization is useful everywhere.

#### Baseline and cost selection

- [ ] Pin code, model, numerics, hardware, thread settings, workload and acceptance thresholds before changing anything.
- [ ] Measure cold load/preparation, warm prefill/decode, reset/reuse, end-to-end requests and training/head paths separately for every affected model generation.
- [ ] Collect repeated benchmark samples plus CPU, allocated bytes/objects and retained-memory profiles; identify dominant allocations/copies by caller and source line.
- [ ] Check algorithmic complexity first: repeated full-prefix work, redundant transforms, repeated quantization, avoidable recomputation and quadratic temporary storage.
- [ ] Choose bounded changes with measurable hypotheses; preserve rollback points and compare each significant change before combining them.

#### Memory, lifetimes and data movement

- [ ] Hoist immutable preparation (weights, norms, RoPE tables, permutations, shapes, offsets and lookup tables) out of hot loops where it is valid; charge it to preparation/retained-memory budgets.
- [ ] Pre-size/reuse bounded session scratch, KV/convolution/engram rings, tape storage and gradients; verify capacity and overflow before allocation or mutation.
- [ ] Remove redundant copies, transposes, gathers, concatenations, padding buffers, dequantization and format conversions. Prefer consuming the correct layout rather than repeatedly rearranging it.
- [ ] Audit view versus copy semantics, closure-captured indexes, overwritten gradient inputs, returned-output ownership, immutable weights, mmap lifetime and failure rollback. Never retain caller-owned mutable buffers without an explicit contract.
- [ ] Avoid per-token/per-layer map lookups, interface boxing, reflection, string keys/formatting, tiny slices and escaping closures where profiles show churn. Check escape-analysis output after refactoring.
- [ ] Keep reusable scratch local to a decoder/session/worker; bound pool retention and clear sensitive/stale state. Do not introduce global mutable scratch or `sync.Pool` merely to conceal allocation counts.
- [ ] Check actual peak/live memory during materialization: original bytes, decoded tensors, transposed copies, packed copies, prepared weights, workspace, gradients and optimizer state can overlap.
- [ ] Verify repeated load/close/reset/error/cancellation cycles do not leak, retain maximum-sized buffers indefinitely or grow memory with sequence position/request count.

#### Arithmetic, layout and backend use

- [ ] Use existing SIMD GEMM/GEMV/vector kernels before writing model-local loops; profile tiny and large shapes before choosing batching, fusion, blocking or dispatch cutoffs.
- [ ] Remove strided/gather-heavy access and repeated packing where a persistent layout is possible. Account for cache locality, memory bandwidth, alignment, tails and temporary layout costs.
- [ ] Consider fusion only when it removes measured passes/materializations without obscuring a testable reference or changing numerical semantics.
- [ ] Check rounding/reduction order and quantization thresholds at every fused/reordered boundary; distinguish reference dequantization, hybrid packed execution and packed-only storage.
- [ ] For training, verify stop-gradient boundaries, STE behavior, all relevant gradients, adapter/full-trunk isolation, optimizer ownership and checkpoint reload after changes.
- [ ] Measure parallelism overhead, oversubscription, false sharing, contention and cancellation/join behavior. Small kernels may need fewer threads, not more.

#### Correctness, evidence and completion

- [ ] Cover valid and rejected shapes/budgets, partial failure rollback, aliases, nonfinite values, padding, concurrency and retained-output stability; add allocation regression assertions where stable.
- [ ] Run scalar/reference and native SIMD differential tests, cached/full-prefix parity, and pinned upstream output/loss/gradient checks without tolerance changes.
- [ ] Run affected-package coverage and report its scope; add missing branch/error-path tests rather than relying on a test count or inventory percentage.
- [ ] Run `gofmt`, affected tests, `go vet ./...`, `go build ./...`, docs/layout checks and relevant cross-builds. For cross-cutting CPU changes, require a confirmed successful whole-tree `GO_PHERENCE_DISABLE_NVIDIA=1 go test -race -p=2 ./... -count=1 -timeout=180s` (or a documented equivalent). A shell timeout or partial log is not a passed gate.
- [ ] Repeat the baseline benchmark/profile matrix, compare distributions, account for initialization/retained-memory shifts and record remaining hotspots. Separate microbenchmark, native-execution, released-model and task-quality evidence.
- [ ] Do not probe/use GPUs, resume frozen evaluations, overwrite preserved binaries/artifacts or change unrelated services without authorization. Check GPU health before authorized use. On a busy ARM host use bounded correctness runs, typically `GOMAXPROCS=2 nice -n 10`.
- [ ] Publish a concise before/after validation record with commands, workload, bytes/allocations, timing variability, peak/retained memory, numerical gates, native platforms, coverage, tradeoffs and open targets. State whether independent review completed.
- [ ] Verify frozen hashes where applicable; commit only the intended files, push and verify remote/CI. Keep unfinished queue items and unmet performance targets open.
