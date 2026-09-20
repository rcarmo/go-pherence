# Needle 2 and Needle 3: native inference/training baseline

This package implements the language-model trunk of both Needle generations in Go, with **Needle 3 as the primary target**. It includes full-sequence inference, bounded greedy token-ID decoding, reverse-mode gradients, full-trunk training, LoRA and AdamW. Python/JAX is used only to generate reference fixtures; it is not part of the Go runtime.

This is a tested **FP32 baseline**, not yet a replacement for the upstream Needle application. Needle 3 additionally has an explicit dequantized CQ-W4/A8/KV8 straight-through reference mode with upstream logits/loss/all-gradient parity. Needle 3 also supports broadcast AB scales (including STE gradients), embedding/confidence/router head inference, and nested depth-rung slicing. Needle 3 `.cact` loading, archive BPE tokenization and text-file inference are implemented through a decoded-weight compatibility path. A direct packed CQ projection path is available as an opt-in hybrid (decoded tensors remain retained); it is slower in the current tests. Source Needle 3 models also support frozen-trunk supervised head updates and admissible trained half-width slicing, including [engram-bearing width parity](../../docs/validation/needle-engram-width-20260920.md) on Intel and native ARM. Source CQ preparation uses dense, pre-scaled Hadamard products to match upstream rounding; packed archive inference retains its fast butterfly. Packed-only loading, Needle 2 archives/quantized training, broader tool-use qualification, dataset-level training/calibration and production-sized performance qualification remain unfinished. Do not confuse successful small-model parity with production readiness.

Bounded schema-constrained tool-call generation is available through `GenerateTools` and CLI `-mode tools`; it generates data only and never executes tools. See the [tool-call guide](../../docs/guides/needle-tool-calls.md) and [validation record](../../docs/validation/needle-tool-calls-20260920.md).

## Architecture and source pins

- Needle 3: [`fc5bae0f9b6138828fe7589f6b531fb9a26968de`](https://github.com/cactus-compute/needle/tree/fc5bae0f9b6138828fe7589f6b531fb9a26968de). Separate Q/K and V widths, grouped-query gated attention, causal QKV convolution taps, local/global attention, engram gathers and dilated taps, twenty-step Sinkhorn lane mixing, conditional learned Monarch/Hadamard factors and tied output embeddings.
- Needle 2: [`741ee892c5f8c4f5c0bb467c9566ea7a1eba919b`](https://github.com/cactus-compute/needle/tree/741ee892c5f8c4f5c0bb467c9566ea7a1eba919b). Its own attention geometry and two fixed Walsh transforms; no substitution of Needle 3's MLP.

The Go implementation adapts the mathematical architecture under upstream's [Apache 2.0 licence](LICENSE.upstream). These are local modifications, not upstream code endorsed by Cactus. The existing, unrelated `memento-go/internal/needle` implementation was not copied or modified.

## Boundaries

- `loader/needle` reads/writes non-executable safetensors, including F32/F16/BF16 input. No pickle deserialization. Saving writes F32, with bounded shapes/headers, finite checks and an atomic same-directory rename. It also parses Needle 3 `.cact` records (FP16/FP32/CQ/raw) and the embedded BPE tokenizer.
- `model/needle` owns architecture, tape, LoRA and optimizer behavior. Loaded weights are copied and private. Public checkpoint/config access returns copies. Auxiliary tensors are retained on checkpoint save and can be executed with `Head`; they are not trained by the language-model loss.
- `backends/simd/runtime.MatMul` validates dimensions, overflow, lengths and destination aliasing, then dispatches to existing native SGEMM/Sdot/Saxpy kernels. Both forward and reverse matrix products use it. Intel AVX2/FMA and ARM64 NEON are selected through runtime capability checks; other targets have portable fallback.
- `cmd/needle` exposes token-ID and archive text-file inference, head inference and source-checkpoint training. The archive path retains decoded tensors and owned CQ payloads. `-packed` uses direct CQ projections for attention, engram projections and the tied output matrix; default execution uses decoded matrices.

Dense products use SIMD. Scalar control, gather/index, transcendental, lane-normalization and some elementwise/backward operations remain; this is **not yet full hot-path SIMD coverage**. The training tape remains allocation-heavy; cached inference uses bounded per-step arenas, direct layouts and an in-place Sinkhorn path. The [allocation/pprof audit](../../docs/validation/needle-allocation-pprof-audit-20260920.md) records measured costs, optimisations and remaining hotspots. `Options.MaxWorkBytes` bounds logical tensor/index/closure accounting, not total process RSS, resident model bytes or optimizer state.

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

Use separate optimizer instances for different parameter sets. Optimizers are session-owned, not concurrent-safe. The immutable model supports concurrent forward calls. For original archive models, `Options{Packed:true}` selects hybrid direct CQ projections; plain source or re-saved decoded checkpoints reject that option. `mask[i]` weights next-token target `ids[i+1]`; a nil mask supervises every next-token position. Invalid or entirely empty masks fail. LoRA targets the upstream five attention projections; initialization is deterministic Go PCG/normal, **not bit-identical JAX PRNG initialization**. Upstream-shaped A/B tensors can be supplied explicitly to `Adapter`.

`TrainStep` and `TrainAdapterStep` default to unquantized FP32. Set `Options.Quant` to `&needle.Quantization{WeightBits: 4, ActivationBits: 8, KVBits: 8}` for Needle 3 CQ-W4/A8/KV8 STE numerics. A separate upstream fixture checks logits, loss and all 42 gradient tensors in that mode; LoRA training also reduces the quantized objective. CQ uses normalized Walsh rotation, upstream codebooks, ties-even FP16 norm rounding and straight-through gradients. It dequantizes to FP32 for computation: this is **not packed quantized inference**. AB-scaled checkpoints apply upstream's `a * CQ(b * W)` in reduction-axis-last orientation, with paired scalar/broadcast scale validation and gradients through the sandwich. `WarmupCosine` is available as a schedule helper; the initial CLI uses a fixed learning rate and repeats one supplied sequence, not upstream's dataset batching/validation workflow. `Generate` remains the full-prefix reference. `GenerateCached` ingests each token once using the incremental decoder described below; neither path claims mid-kernel preemption.

## Incremental decoding

`m.NewDecoder(needle.DecoderOptions{Capacity: 1024, MaxCacheBytes: 512 << 20, Execution: opts})` creates a session-owned decoder. `Step(ctx, token)` returns owned next-token logits; `Position`, `CacheBytes` and `Reset` expose bounded state. Calls are serialized. Failed or cancelled steps do not commit any KV, convolution or engram history. `Reset` clears prompt/cache contents while retaining capacity and immutable prepared weights. The model remains immutable and can serve independent concurrent sessions.

The decoder caches normalized/rotated K/V, raw QKV convolution history and engram value taps, using absolute RoPE positions. Local attention uses fixed rings; global attention stops at the admitted capacity. Capacity cannot exceed model context or the archive KV window. **No unlimited streaming or eviction of global context is claimed.** CQ/AB weights are prepared once per decoder, while archives reuse already decoded weights. K/V storage is FP32, including dequantized A8 values—not packed int8.

Cache admission accounts for fixed rings, IDs, prepared CQ weights, metadata and conservative preparation scratch before allocations. `CacheBytes` reports logical retained state excluding the shared model, not process RSS or per-step workspace. Per-step workspace is independently bounded by `Execution.MaxWorkBytes`. The CLI defaults to cached inference, with `-cache-mib` controlling admission and `-cached=false` preserving the full-prefix oracle. `GenerateCached` sizes its temporary decoder to the requested generation rather than allocating maximum context.

Every cached token is checked against a fresh full-prefix forward for Needle 2/3 FP32, Needle 3 CQ/AB, decoded archives and depth slices over twelve tokens (crossing local-attention and convolution ring boundaries). Tests cover reset/reuse, returned-logit mutation, option ownership, invalid tokens, capacity/budget rejection and mid-step cancellation rollback. The tests also pass with AVX2/FMA disabled. An independent cache-review delegation timed out and supplied no findings; it is not review coverage.

## Auxiliary heads and depth rungs

`m.Head(ids, needle.Embedding, opts)` returns the normalized embedding. `needle.Confidence` returns one raw logit and `needle.Router` returns three raw logits. These are **not calibrated probabilities**; this API deliberately does not apply stored router thresholds or advertise confidence after fine-tuning. It implements upstream's padding-aware token probes, RMS gains and query pooling over input/per-layer hidden cells. All-padding input fails; leading/interior padding otherwise follows upstream, including its finite all-masked attention-row behavior. Head tensor dimensions must agree with the config.

`m.SliceDepth(depth)` selects Needle 3's nested bisection rung, remaps global/engram sites and head rows, and preserves the parent's order for subsequent slices. It returns an independent model. Invalid depths/orders or unknown stacked geometry fail. Reduced-depth AB-scaled models currently fail explicitly: upstream's slice leaves those scales unchanged, so supporting their remapping requires a separately defined contract. `SliceWidth` supports a trained half-width source rung subject to strict attention/Hadamard/engram geometry checks; it rejects deployment archives and AB scales. Runtime exit-depth sampling is not implemented. The [head-training and width guide](../../docs/guides/needle-head-training.md) has objectives, target JSON, API/CLI examples and limits.

## Needle 3 archives and tokenizer

`needle.LoadArchive(path)` returns a model and, when present, its embedded `loader/needle.Tokenizer`. The archive reader validates the 196-byte header, 44-byte positional directory, 64-byte alignment, record overlap/extents, shapes, codebooks and finite values. It reads binary, 2/3/4-bit CQ and ternary crumbs at group size 128. Bounds are 512 MiB for file and decoded data, 32 MiB for an individual raw attachment, plus a conservative 1 GiB logical materialization budget including optional heads and copies. These are admission limits, not an RSS guarantee.

Weights are decoded once, transposed/restacked into model layout, and **not requantized**. Original CQ payloads are also retained for optional direct products, and count against archive admission. Fresh file loading transfers private buffers rather than making a second full model copy. The executable path currently requires KV8 and fixes A8/KV8 numerics. It rejects training/LoRA on decoded archives and prefixes longer than the archive's KV window. Incremental decoding works within that bound; global-window eviction is not implemented. Saved decoded checkpoints retain a Go-specific archive-numerics marker so reloading cannot silently turn off A8 or enable training. Archive depth slices keep this marker. KV2/3/4 execution fails explicitly despite the reader understanding their metadata.

The tokenizer matches upstream `RefTokenizer`'s exported BPE contract: score-prioritized merges with leftmost tie-breaking, longest user-marker matching, dummy prefix, whitespace escape and byte fallback. Encoding is limited to 1 MiB of valid UTF-8; decoding validates IDs and caps output. It does not implement arbitrary SentencePiece normalization or schema-constrained generation. Untrusted user text containing user-defined chat markers is interpreted as such, like upstream; callers must construct/escape prompts deliberately.

## CLI

```sh
go build -o bin/needle ./cmd/needle
# Cached inference is the default; -cached=false uses full-prefix recomputation.
# Optional Needle3 quantization-aware reference mode:
#   -numerics needle3-cq4-a8-kv8
# Input contains tokenizer IDs from the matching generation, not arbitrary text.
printf '{"tokens":[2,7,4,9,3]}' > /tmp/needle-ids.json
bin/needle -model checkpoints/needle3.safetensors -input /tmp/needle-ids.json -max-new 8
bin/needle -mode train -model checkpoints/needle3.safetensors \
  -input /tmp/needle-ids.json -steps 10 -lr 0.001 \
  -lora-rank 8 -lora-alpha 16 -out checkpoints/needle3-tuned.safetensors
# Requires loaded head weights; returns values and calibrated:false.
bin/needle -mode embedding -model checkpoints/needle3.safetensors \
  -input /tmp/needle-ids.json -layers 8
# -mode confidence or -mode router returns raw logits.
# Archive numerics are fixed; omit -numerics. Text mode prepends archive BOS.
# Optional -packed uses direct CQ projections; slower in current smoke tests.
bin/needle -model checkpoints/needle3.cact -text-file prompt.txt -max-new 8
bin/needle -model checkpoints/needle3.cact -text-file prompt.txt -mode embedding
```

An output path that already exists is rejected by the CLI. The saved training output is a **merged full checkpoint**, not an upstream adapter archive. Numerics mode is chosen explicitly when loading it; a trained F32 checkpoint does not become a packed archive. Language-model fine-tuning does not retrain the preserved confidence/router heads. `-mode train-head` can update one selected head with a frozen trunk, but does not recalibrate it; old calibration must not be presented as qualified for changed weights. Do not use these example token IDs as a quality evaluation. A current full-width model may exceed the default reference-tape budget; increasing it is not a substitute for the pending streaming/memory work.

## Validation

The fixtures are small, deterministic, perturbed upstream models, not hand-written expected values. Needle 3 checks logits, mean next-token loss and all **42** gradient tensors; Needle 2 checks all **29** trunk-gradient tensors. Upstream's unused Needle 2 MTP training leaves are excluded, as its loader does.

The four-layer extended fixture additionally checks all three head outputs with interior padding in FP32 and CQ, AB-scaled logits/loss and every nonzero upstream gradient (including broadcast scale gradients), and exact tensor/output parity for direct 3-/2-layer and nested 3→2 slices. Admission and regression tests cover invalid AB broadcasts/pairs, missing heads, padding, nil models, tiny-value A8 quantization, and depth/geometry failures. A review caught and corrected a spurious A8 minimum-scale floor; upstream only special-cases exactly-zero rows.

The archive fixture comes from the pinned exporter: all 154 decoded records match upstream, and an independent reconstruction into the pinned JAX model matches mapped Go logits and all head outputs. This is **not a comparison against the native C++ engine**. Ten tokenizer cases cover merges, overlapping markers, Unicode/byte fallback and whitespace; five additional cases check Python-compatible malformed-byte replacement. Short five-second fuzz runs covered both parsers (63,979 archive and 128,562 tokenizer executions), with malformed/truncated/overflow/overlap admission tests and a 32-bit wrapped-special-ID regression. These are bounded smoke runs, not exhaustive fuzzing or security clearance.

Tests also cover full-model loss reduction for both versions, Needle 3 LoRA loss reduction and a finite-difference gradient, atomic optimizer rejection, checkpoint round-trips, auxiliary-tensor preservation, owned state, concurrent inference, token/work bounds and the train/save/load/generate CLI flow.

On 2026-09-20, a Go-trained Needle 3 fixture checkpoint was reloaded by Python safetensors and the upstream JAX model; both runtimes generated `[3, 3]` from the same prefix. Three FP32 training steps reduced loss from 2.76507 to 2.69774. Three CQ-W4/A8/KV8 LoRA steps reduced loss from 2.76678 to 2.68823; the saved merged checkpoint generated `[3, 5]` in both Go and upstream quantized-reference inference. Whole-tree NVIDIA-disabled race tests passed for 114 packages (54 had no tests); vet/build and the 355-document gate also passed.

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
python scripts/needle-reference.py --upstream /path/to/pinned/needle \
  --quantized --output model/needle/testdata/needle3-cq.json
python scripts/needle-extended-reference.py --upstream /path/to/pinned/needle \
  --output model/needle/testdata/needle3-extended.json
python scripts/needle-archive-reference.py --upstream /path/to/pinned/needle \
  --output-dir loader/needle/testdata
python scripts/needle-head-width-reference.py --upstream /path/to/pinned/needle \
  --output model/needle/testdata/needle3-head-width.json
```

`testdata/cq-codebooks.json` records upstream `_cq_codebook_np(bits, 128)` for 1, 1.58, 2, 4 and 8 bits from the same Needle 3 pin. Four-bit end-to-end parity is tested; the other tables are present but not separately end-to-end qualified.

Measured on the Intel i7-12700 host: dense `[32,768] × [768,576]` forward multiplication about **0.399 ms**, zero allocations; transposed variants about **0.864–1.262 ms**. The tiny five-token, width-eight Needle 3 fixture takes about **0.172 ms**, 298 KB and 5065 allocations per forward. A subsequent twelve-token comparison on the same tiny fixture measured cached decoding at **0.577–0.598 ms** versus **2.408–2.426 ms** for twelve full-prefix forwards, about 4× faster. Allocations fell from 4.39 MB / 72,221 allocations to 0.923 MB / 21,445 allocations for that sequence (prepared decoder reused). These are short local microbenchmarks, not real-model token/s or comparisons against upstream.

Native ARM64 fixture qualification now passes on the CIX P1 CD8160: Needle model, loader, CLI and SIMD runtime tests all pass after correcting NEON SGEMM reduction, BF16 narrowing and RMSNorm tail bugs. See the [native ARM report](../../docs/validation/needle-native-arm64-20260920.md) for failed-first results, final scope and reproduction details. The Intel tests also pass with AVX2/FMA disabled; RISC-V remains compile-only.

The released 20-layer Needle 3 archive now passes bounded text smoke tests on Intel and native ARM, with identical twelve-token continuations across decoded and direct-CQ paths. This is not task-quality or native-C++ parity validation; the [packed/real-model report](../../docs/validation/needle-packed-real-model-20260920.md) records the model pin/hash, exact output, cold timing and RSS. Packed remained slower and remains opt-in.

Full-size warmed Intel/ARM throughput, packed-only model storage, vectorised unpacking/Hadamard work, global KV eviction, reduced-depth AB remapping, Needle 2 archives/quantized objectives/heads, schema-constrained tool calling, dataset-level head training/calibration and production deployment behaviour remain open. Frozen evaluation artifacts and GPU services are unrelated and untouched.

Needle 2 additionally supports its distinct FP32 contrastive/confidence head pooling and frozen-trunk MSE/BCE updates; see the [head guide](../../docs/guides/needle-head-training.md). Contrastive temperature is preserved, not trained; no paired contrastive objective or calibration is claimed.
