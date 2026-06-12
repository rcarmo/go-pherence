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
│       ├── k3engine/   # K3 AI core engine
│       │   └── aipool/ # A100 worker pool + Q80x32 GEMM dispatch
│       ├── rvv/        # RVV vector kernels (FP16, SiLU, FastExp, dot, GEMM)
│       └── tcm/        # TCM (Tightly Coupled Memory) access
├── cmd/                # CLI entry points
│   ├── audio/          # Whisper and audio tools
│   ├── image/          # Ideogram4, VAE probes, image tools
│   ├── k3/             # K3-specific benchmarks
│   ├── llm/            # LLM speculative decoding tools
│   └── models/         # Model inspection tools
├── docs/               # Architecture docs, supported models, GPU options
├── gpu/                # GPU management utilities
├── half/               # FP16/BF16 conversion
├── internal/           # Checked arithmetic, internal utilities
├── loader/             # Model loaders (GGUF, safetensors, audio, config, tokenizer)
├── model/              # GGUF-based LLM models (llama, qwen, etc.)
│   ├── diffusiongemma/ # DiffusionGemma image generation
│   ├── ideogram4/      # Ideogram v4 image generation (see model/ideogram4/README.md)
│   ├── hunyuan3d/      # Hunyuan 3D
│   └── qwen/           # Qwen native models
├── models/             # Non-GGUF native models
│   ├── whisper/        # Whisper speech recognition
│   └── speaker/        # Speaker diarization
├── prompts/            # Prompt templates
├── research/           # Research prototypes (NPU whisper, etc.)
└── testdata/           # Test fixtures
```

## Tooling

### Required

- **Go 1.24+** — the module uses recent Go features.
- **`go vet`** — run before every commit. Do not commit code that fails vet.
- **`go test ./...`** — run affected packages before committing. When touching `backends/`, run `go test ./backends/...`.
- **`gofmt -w`** — format all modified `.go` files before committing.

### Recommended

- **`gopls`** — install with `go install golang.org/x/tools/gopls@latest`. Use it for:
  - Finding all references to a symbol before renaming or moving it.
  - Checking which packages import a function before changing its signature.
  - Navigating build-tagged files (`_riscv64.go`, `_other.go`).
  - Mechanical refactors like extracting interfaces or inlining helpers.
- **`go build ./...`** — verify the entire tree compiles after cross-cutting changes.

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
9. **Test tolerance.** If you change a numerical approximation (e.g. FastExp, polynomial SiLU), check whether existing tests have tight tolerances that will break. Widen tolerances explicitly with a comment, or gate the approximation behind a build tag.

### Committing

10. **Run `gofmt -w` on all modified files.**
11. **Run `go test` on affected packages.** At minimum: the package you edited plus any package that imports it.
12. **Run `go vet` on affected packages.**
13. **Run `go build ./...`** to verify the full tree compiles.
14. **Write clear commit messages.** First line is a short summary (`package: what changed`). Body explains why, not just what.
15. **Do not commit temporary/debug code** (print statements, hardcoded paths) unless guarded by an env flag like `GO_PHERENCE_IDEOGRAM4_TIMING`.

### Architecture conventions

16. **Model-specific code stays in `model/` or `models/`.** Do not put Ideogram-specific logic in `backends/`.
17. **Reusable kernels go in `backends/`.** If a kernel (SiLU, FastExp, Q8 packing) is useful to multiple models, put it in the appropriate backend package.
18. **Build tags for platform-specific code.** Use `//go:build riscv64` / `//go:build !riscv64`, not runtime checks, for platform-specific kernel dispatch.
19. **The `half` package** owns FP16/BF16 conversion. Do not duplicate it.
20. **The `loader` package** owns model file I/O (GGUF, safetensors, audio, tokenizers). Do not add model loading logic directly in `model/`.

### K3/SpacemiT-specific

21. **A100 cores require `/proc/set_ai_thread` registration.** Never use plain `taskset` or `runtime.LockOSThread` alone for A100 access. Use the `aipool` package.
22. **TCM is 3 MB of on-chip SRAM** (8 × 384 KB blocks). It is useful for weight/activation staging but not for large buffers. The `tcm` package manages it.
23. **K3 mode disables all NVIDIA paths.** If you add a new GPU-gated path, check `gpuDisabledByK3()` or use the existing `gpuXxxEnabled()` predicates.
24. **31 GB RAM, no swap.** Be conscious of peak memory. Release large temporaries before allocating new ones. Use `runtime.GC()` / `debug.FreeOSMemory()` at phase boundaries for large pipelines.
25. **Row-scale Q8** is the preferred A100 quantization contract for quality-sensitive paths (matches the native int8 quantization used by Whisper and Ideogram).
