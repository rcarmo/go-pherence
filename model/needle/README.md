# Needle 2 and Needle 3: native FP32 baseline

This package implements the language-model trunk of both Needle generations in Go, with **Needle 3 as the primary target**. It includes full-sequence inference, bounded greedy token-ID decoding, reverse-mode gradients, full-trunk training, LoRA and AdamW. Python/JAX is used only to generate reference fixtures; it is not part of the Go runtime.

This is a tested **FP32 baseline**, not yet a replacement for the upstream Needle application. The quantized `.cact` runtime, CQ/A8 straight-through fine-tuning, SentencePiece-driven tool calling, auxiliary heads, ladder slicing and production-sized performance qualification remain unfinished. Do not confuse successful small-model parity with production readiness.

## Architecture and source pins

- Needle 3: [`fc5bae0f9b6138828fe7589f6b531fb9a26968de`](https://github.com/cactus-compute/needle/tree/fc5bae0f9b6138828fe7589f6b531fb9a26968de). Separate Q/K and V widths, grouped-query gated attention, causal QKV convolution taps, local/global attention, engram gathers and dilated taps, twenty-step Sinkhorn lane mixing, conditional learned Monarch/Hadamard factors and tied output embeddings.
- Needle 2: [`741ee892c5f8c4f5c0bb467c9566ea7a1eba919b`](https://github.com/cactus-compute/needle/tree/741ee892c5f8c4f5c0bb467c9566ea7a1eba919b). Its own attention geometry and two fixed Walsh transforms; no substitution of Needle 3's MLP.

The Go implementation adapts the mathematical architecture under upstream's [Apache 2.0 licence](LICENSE.upstream). These are local modifications, not upstream code endorsed by Cactus. The existing, unrelated `memento-go/internal/needle` implementation was not copied or modified.

## Boundaries

- `loader/needle` reads/writes non-executable safetensors, including F32/F16/BF16 input. No pickle deserialization. Saving writes F32, with bounded shapes/headers, finite checks and an atomic same-directory rename.
- `model/needle` owns architecture, tape, LoRA and optimizer behavior. Loaded weights are copied and private. Public checkpoint/config access returns copies. Auxiliary tensors are retained on checkpoint save but are not executed or trained by the trunk.
- `backends/simd/runtime.MatMul` validates dimensions, overflow, lengths and destination aliasing, then dispatches to existing native SGEMM/Sdot/Saxpy kernels. Both forward and reverse matrix products use it. Intel AVX2/FMA and ARM64 NEON are selected through runtime capability checks; other targets have portable fallback.
- `cmd/needle` exposes the baseline without pretending to have a tokenizer or quantized deployment engine.

Dense products use SIMD. Scalar control, gather/index, transcendental, lane-normalization and some elementwise/backward operations remain; this is **not yet full hot-path SIMD coverage**. The tape is intentionally simple and allocation-heavy. `Options.MaxWorkBytes` bounds logical tensor/index/closure accounting, not total process RSS, resident model bytes or optimizer state.

## API

```go
m, err := needle.Load("checkpoints/needle3.safetensors")
// Handle err before use.
logits, err := m.Forward(ids, needle.Options{MaxWorkBytes: 512 << 20})
loss, gradients, err := m.LossGrad(ids, targetMask, needle.Options{})
optimizer := needle.NewAdamW()
next, loss, err := m.TrainStep(optimizer, ids, targetMask, 0.001, needle.Options{})
adapter, err := m.NewAdapter(8, 16, 42)
adapter, loss, err = m.TrainAdapterStep(adapter, optimizerForAdapter, ids, targetMask, 0.001, needle.Options{})
```

Use separate optimizer instances for different parameter sets. Optimizers are session-owned, not concurrent-safe. The immutable model supports concurrent forward calls. `mask[i]` weights next-token target `ids[i+1]`; a nil mask supervises every next-token position. Invalid or entirely empty masks fail. LoRA targets the upstream five attention projections; initialization is deterministic Go PCG/normal, **not bit-identical JAX PRNG initialization**. Upstream-shaped A/B tensors can be supplied explicitly to `Adapter`.

`TrainStep` and `TrainAdapterStep` use unquantized FP32 gradients. This is not upstream's CQ W4/A8 STE objective. `WarmupCosine` is available as a schedule helper; the initial CLI uses a fixed learning rate and repeats one supplied sequence, not upstream's dataset batching/validation workflow. `Generate` recomputes the full prefix and checks cancellation between forwards; it does not claim KV caching or mid-kernel preemption.

## CLI

```sh
go build -o bin/needle ./cmd/needle
# Input contains tokenizer IDs from the matching generation, not arbitrary text.
printf '{"tokens":[2,7,4,9,3]}' > /tmp/needle-ids.json
bin/needle -model checkpoints/needle3.safetensors -input /tmp/needle-ids.json -max-new 8
bin/needle -mode train -model checkpoints/needle3.safetensors \
  -input /tmp/needle-ids.json -steps 10 -lr 0.001 \
  -lora-rank 8 -lora-alpha 16 -out checkpoints/needle3-tuned.safetensors
```

An output path that already exists is rejected by the CLI. The saved training output is a **merged full checkpoint**, not an upstream adapter archive. Do not use these example token IDs as a quality evaluation. A current full-width model may exceed the default reference-tape budget; increasing it is not a substitute for the pending streaming/memory work.

## Validation

The fixtures are small, deterministic, perturbed upstream models, not hand-written expected values. Needle 3 checks logits, mean next-token loss and all **42** gradient tensors; Needle 2 checks all **29** trunk-gradient tensors. Upstream's unused Needle 2 MTP training leaves are excluded, as its loader does.

Tests also cover full-model loss reduction for both versions, Needle 3 LoRA loss reduction and a finite-difference gradient, atomic optimizer rejection, checkpoint round-trips, auxiliary-tensor preservation, owned state, concurrent inference, token/work bounds and the train/save/load/generate CLI flow.

On 2026-09-20, a Go-trained Needle 3 fixture checkpoint was reloaded by Python safetensors and the upstream JAX model; both runtimes generated `[3, 3]` from the same prefix. Three FP32 training steps reduced loss from 2.76507 to 2.69774. Whole-tree NVIDIA-disabled race tests passed for 114 packages (54 had no tests); vet/build and the 355-document gate also passed.

```sh
GO_PHERENCE_DISABLE_NVIDIA=1 go test -race ./model/needle ./loader/needle ./cmd/needle ./backends/simd/runtime
go test ./backends/simd/runtime -run '^$' -bench '^BenchmarkMatMul' -benchmem
go test ./model/needle -run '^$' -bench BenchmarkNeedle3Forward -benchmem
```

Reference regeneration is offline apart from installing Python dependencies. Pin JAX/jaxlib 0.11.2, Flax 0.12.9, NumPy 2.5.3 and Optax 0.2.8. Scripts force CPU execution and never download weights:

```sh
python scripts/needle-reference.py --upstream /path/to/pinned/needle \
  --output model/needle/testdata/needle3.json
python scripts/needle2-reference.py --upstream /path/to/needle/git/repository \
  --output model/needle/testdata/needle2.json
```

Measured on the Intel i7-12700 host: dense `[32,768] × [768,576]` forward multiplication about **0.399 ms**, zero allocations; transposed variants about **0.864–1.262 ms**. The tiny five-token, width-eight Needle 3 fixture takes about **0.172 ms**, 298 KB and 5065 allocations per forward. These are short local microbenchmarks, not real-model token/s or comparisons against upstream.

ARM64 test binaries and the CLI cross-compile, as does the RISC-V CLI; **no native ARM execution has been performed for this slice**. Full native Intel/ARM model throughput, scalar-versus-SIMD parity/performance, quantized formats/objectives, tokenizer, auxiliary heads and deployment behavior remain open. Frozen evaluation artifacts and GPU services are unrelated and untouched.
