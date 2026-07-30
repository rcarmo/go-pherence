# TurboFieldfare adoption results

This page tracks the baseline measurement surface added for the first TurboFieldfare-inspired MoE adoption slice.

Current scope is deliberately narrow:

- deterministic `ExpertPool` route replay infrastructure
- replay tests and slot-size benchmarks
- synthetic selected-expert compute benchmarks
- no LFU eviction policy changes
- no new CUDA routed-expert kernels

## Hardware snapshot

| Item | Value |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700 exposed as 6 KVM vCPUs |
| GPU | NVIDIA GeForce RTX 3060, driver 580.173.02, 12 GiB |
| Go | 1.26.3 |
| OS | Linux 6.8.0-136-generic |

Raw scaffold metadata lives in [`../benchmarks/turbo-fieldfare-adoption/metadata.json`](../benchmarks/turbo-fieldfare-adoption/metadata.json).

## Asset status

| Asset | Path | Status |
|---|---|---|
| Qwen3-30B-A3B MLX4 | `models/qwen3-30b-a3b-mlx4/config.json` | unavailable on this host |

Because the local Qwen3-30B asset is unavailable, this slice only commits synthetic replay and selected-expert compute benchmarks. Full route-set captures against a real 30B checkpoint remain pending.

## Added baseline benchmarks

### 1. ExpertPool route replay

Command:

```bash
go test ./backends/nvidia/runtime -run '^$' -bench 'BenchmarkExpertPoolReplaySyntheticGlobalKeys' -benchmem
```

What it measures:

- deterministic replay of global pool keys across layers
- hit/miss/eviction counts under the production LRU policy
- per-layer hit distributions for selected slot sizes

The synthetic route spans 48 layers, 32 decode steps and top-4 selections. Because the global pool key includes the layer, all three slot counts show only `1.758%` hits; evictions fall from `6020` at 16 slots to `5972` at 64. This is a useful baseline result in its own right: a global pool much smaller than one complete route set cannot exploit the within-layer repetition embedded in this fixture. LFU needs per-layer distributions or quotas to avoid simply preserving keys from whichever layers dominate whole-run counts.

Raw output: [`baseline-e7e482bc/expert-pool-replay.txt`](../benchmarks/turbo-fieldfare-adoption/baseline-e7e482bc/expert-pool-replay.txt).

### 2. Selected-expert compute

Command:

```bash
go test ./model -run '^$' -bench 'BenchmarkMoESelectedExpertComputeSynthetic512x1024Top4' -benchmem
```

What it measures:

- when live CUDA is available: warm-pool generic GPU MoE vs. a direct per-expert GPU chain
- otherwise: synthetic CPU generic MoE vs. a direct per-expert CPU chain

On the RTX 3060, the warm generic path measures `186--205µs` and 438 allocations for a synthetic hidden-512/intermediate-1024/top-4 call. The direct per-expert chain measures `166--169µs` and 373 allocations. This is not a candidate speedup -- both execute the current GEMV sequence -- but it bounds the overhead outside constituent expert kernels and supplies the oracle for a persistent routed-MoE implementation.

The initial long benchmark exposed a post-test CUDA teardown crash; bounded `20x` runs with explicit synchronization pass. That lifecycle issue must stay part of the persistent-kernel gate.

Raw output: [`baseline-e7e482bc/selected-expert-compute.txt`](../benchmarks/turbo-fieldfare-adoption/baseline-e7e482bc/selected-expert-compute.txt).

### 3. Gemma4 KV planning

The available Gemma4 E4B QAT GGUF reports 42 layers, KV width 1024 and a 131,072-token maximum context. Linear F32 K/V storage at that maximum is `45,097,156,608` bytes. A structural estimate with six full-attention layers and 36 sliding layers, a 1,024-token window and 128-token prefill allowance is `6,782,189,568` bytes (`15.04%` of linear storage).

The estimate is deliberately labelled rather than presented as a measured runtime result: the generic GGUF loader does not currently expose the full layer mask from this checkpoint and cannot run it because tied `output.weight` resolution is incomplete. The F32 ring implementation must derive its layer types from the actual model loader before this estimate becomes an acceptance value.

Raw metadata and estimates:

* [`gemma4-e4b-f16-kv.json`](../benchmarks/turbo-fieldfare-adoption/baseline-e7e482bc/gemma4-e4b-f16-kv.json)
* [`gemma4-e4b-turbo-kv.json`](../benchmarks/turbo-fieldfare-adoption/baseline-e7e482bc/gemma4-e4b-turbo-kv.json)
* [`gemma4-e4b-f32-ring-estimate.json`](../benchmarks/turbo-fieldfare-adoption/baseline-e7e482bc/gemma4-e4b-f32-ring-estimate.json)

### 4. Sampling readiness

Generic LLM generation is greedy-only. There is no generic Top-P/Top-K/temperature baseline to record; the server rejects non-zero temperature. Model-specific DiffusionGemma device sampling exists but has different semantics. Sampling adoption therefore starts with an API/reference contract, not a performance patch.

### 5. Persistent selected-expert projection candidate

The first persistent candidate processes one projection stage for all selected MLX experts through a shared pointer/work table and an atomic output-row queue. It is deliberately narrower than a complete MoE fusion: gate, up and down remain separate mathematical stages.

Live RTX 3060 parity is exact for four experts in non-zero order, a three-route tail and a five-route sequence with a repeated expert. Performance rejects the candidate at hidden 512/output 1024:

| Work | Persistent candidate | Repeated `GemvMLXDirect` | Result |
|---|---:|---:|---:|
| 4 routes | 89--94µs | 59--60µs | 0.65x |
| 3 routes | 72--73µs | 46µs | 0.63x |
| 5 routes | 104--105µs | 70--72µs | 0.68x |

The persistent launch cuts Go-side allocations, but pointer-table uploads, claim reset and atomic row scheduling cost more than four ordinary launches at this size. It remains behind an explicit live-test flag and is not integrated into `moeForwardGPU`. A future complete routed-MoE kernel would need to reuse metadata across gate/up/down or process much larger rows to overcome this baseline.

Raw output: [`baseline-e7e482bc/persistent-projection-candidate.txt`](../benchmarks/turbo-fieldfare-adoption/baseline-e7e482bc/persistent-projection-candidate.txt).

### 6. LRU/LFU policy replay

The production pool now accepts `lru` or `lfu`; LRU remains the default. LFU uses whole-run frequency over the global `layer*experts+expert` key and recency as a tie-breaker. `GO_PHERENCE_EXPERT_CACHE_POLICY=lfu` enables it for controlled model runs.

The broad 48-layer fixture is deliberately hostile to a small global pool: all slot sizes get `1.758%` LRU hits, while global LFU falls to `0.31--0.59%`. Whole-run counts preserve early keys from many layers and make later layers churn.

A synthetic layer-locality fixture shows the other side:

| Slots | LRU hit rate | LFU hit rate |
|---:|---:|---:|
| 8 | 7.357% | 9.201% |
| 16 | 7.357% | 14.39% |
| 32 | 7.357% | 22.72% |

LFU can exploit repeated experts when locality fits the budget, but the global topology can also make it worse. It must not become the default without real per-layer route traces; a layer-aware admission/quota policy remains the likely follow-up.

Raw output: [`baseline-e7e482bc/expert-cache-policy-replay.txt`](../benchmarks/turbo-fieldfare-adoption/baseline-e7e482bc/expert-cache-policy-replay.txt).

### 7. F32 KV storage contract

`runtime/kv` now has linear and fixed-capacity ring stores with paired K/V rows, logical oldest-to-newest views, split spans at wrap, materialisation, byte accounting and checkpoint/restore. The abstraction is not wired into generation yet; it isolates addressing and rollback semantics before model code changes.

Focused tests cover malformed rows, repeated wraps, split views, reset, linear/ring restore and cross-store checkpoint compatibility. Production generation continues to use the existing slices until the mixed SWA/full integration passes its own gates.

### 8. Mixed F32 ring state integration

`LayeredF32KV` now builds per-layer stores from `{dim, layer type, sliding window}`: sliding layers receive `window + maxPrefillChunk` ring capacity and full-attention layers remain linear. The opt-in `CPUDecodeState` bridge can migrate existing prompt slices, clear their backing arrays, materialise logical suffixes for verifier seams, and checkpoint/restore or retain accepted verifier rows after a ring wrap.

Focused tests cover mixed sliding/full state, multiple wraps, split logical views, prompt seeding, accepted-prefix plus bonus retention, malformed seeds and atomic rollback. `runtime/kv` cross-builds for arm64 and riscv64. Ordinary generation and NVIDIA KV remain unchanged; the exact storage boundary is ready, but dispatch cannot be enabled until decode attention consumes logical ring views directly.

### 9. FP16 ring candidate

A standalone FP16 ring halves physical KV bytes and remains allocation-free on append, but conversion is not free. At dim 512, F32 append is roughly `220--247ns`; FP16 append is `1.69--2.01µs`. Materialising a 4,224-token dim-512 ring to F32 is roughly `3.0ms` from F32 storage and `6.8--7.0ms` from FP16.

Deterministic attention-like data measures approximately `2.43e-4` maximum K/V element error, `3.44e-5` score error and `3.72e-5` context error. These numbers are suitable for selecting model-level tolerances, not for enabling the path. The available Gemma4 E4B GGUF cannot execute because its tied `output.weight` is unresolved, so selected-logit and long-generation gates remain unavailable. FP16 stays experimental and unintegrated.

Raw output: [`baseline-e7e482bc/fp16-ring-candidate.txt`](../benchmarks/turbo-fieldfare-adoption/baseline-e7e482bc/fp16-ring-candidate.txt).

### 10. Split-KV NVIDIA decode attention

The split-KV candidate writes one stable online-softmax partial per 256-key chunk (`max`, exponential sum and weighted V), then merges chunks without recomputing QK. Live RTX 3060 parity passes against the CPU oracle for sequence lengths 31, 32, 255, 256, 257, 512, 2,048, 4,096 and 16,384, including a non-standard head-dimension tail.

| Sequence | Existing shared scores | Split-KV | Speedup |
|---:|---:|---:|---:|
| 512 | 99--103µs | 49--51µs | ~2.0x |
| 2,048 | 465--466µs | 74µs | ~6.3x |
| 4,096 | unavailable (2,048-score cap) | 104--105µs | new coverage |
| 16,384 | unavailable | 336--340µs | new coverage |

Production dispatch now selects split-KV at 512 keys and retains the previous kernel below that threshold. The candidate uses reusable partial buffers; the higher Go launch-allocation count remains visible for later runtime cleanup.

Raw output: [`baseline-e7e482bc/split-kv-attention-candidate.txt`](../benchmarks/turbo-fieldfare-adoption/baseline-e7e482bc/split-kv-attention-candidate.txt).

### 11. Memory-budgeted prefill chunks

A backend-neutral planner estimates current CPU/GPU reusable prefill buffers and chooses the largest allowed 32/64/128-token chunk within a scratch budget. Planning costs roughly `269--287ns` and emits absolute spans with non-divisible tails.

For a representative hidden-4096, Q-4096, KV-1024, intermediate-14336 shape, the conservative scratch estimate is `7.75MiB` at 32 rows, `15.5MiB` at 64 and `31MiB` at 128. The existing Qwen3.6 layer-streamed executor now accepts planner spans; synthetic mixed full/linear-attention tests match token-sequential and one-full-chunk outputs, prefix states and final KV/recurrent state exactly for 32/64/128 chunks and a 130-token tail.

`qwen36run -layer-streamed-prefill` uses a 256MiB planner budget by default. `-prefill-chunk-size` remains an explicit override, while `-prefill-scratch-mb` controls automatic selection. Focused dense prefill, GGUF batch projection, multimodal embedding, sliding-window and MTP prefill gates remain green. Only Qwen3.6 is switched to planner-driven chunks because it already has a state-preserving layer-streamed executor; the other full-prompt executors are validated against regression but remain unswitched until they gain equivalent span adapters.

Raw planner output: [`baseline-e7e482bc/prefill-chunk-planner.txt`](../benchmarks/turbo-fieldfare-adoption/baseline-e7e482bc/prefill-chunk-planner.txt).

### 12. Out-of-core expert streaming

`runtime/expertstream` validates an immutable SHA-256 manifest, requires each expert's gate/up/down payloads to be contiguous, allocates fixed 4KiB-aligned reusable host slots, and services misses with a bounded `ReadAt` worker pool. It rejects overlap, malformed quant layouts, checksum changes, truncation and zero-progress reads. Slot replacement is deterministic LRU; duplicate requests preserve caller order.

On the i7-12700, four resident 1MiB experts resolve in `399--429ns`. Alternating two four-expert sets through four slots forces four reader misses per call and measures `130--144µs` from the warm OS page cache. This is explicitly **not** a disk-cold bandwidth claim. The experimental model adapter supports exact MLX affine payloads (`uint32` packed weights plus F32 scales/biases), performs hit-first planning, batches only misses, uploads in deterministic request order through the production NVIDIA upload function, and leaves normal decode unchanged unless a source is explicitly injected.

Measured/validated gates:

| Gate | Result |
|---|---|
| aligned fixed slots and RSS bound | slot allocation is exactly `slots * alignUp(maxExpertSpan, alignment)` plus manifest/file metadata |
| short reads and post-open truncation | rejected with `ErrShortRead`; partial assignments discarded |
| corruption | size and SHA-256 mismatch rejected before slot allocation |
| bounded parallel reads | worker cap and post-load goroutine cleanup tested |
| exact MLX typed views | zero-copy synthetic packed-weight/scale/bias fixtures pass |
| live device | RTX 3060 available (12GiB); existing focused NVIDIA pool/upload gates pass |
| Qwen3-30B full-token parity/TPS | unavailable: checkpoint absent |
| checkpoint above RAM/VRAM, disk-cold and I/O/upload overlap | unavailable without a representative package; no synthetic claim substituted |

The facility remains experimental and non-default. RDADVISE/read-ahead and speculative cross-layer reads were not added because page-cache-backed microbenchmarks cannot justify them.

### 13. Generic autoregressive sampling contract

`runtime/sampling` defines deterministic greedy, temperature, Top-K and Top-P behavior independently of model generation. Candidates order by descending logit and ascending token ID; NaN and negative infinity are excluded; positive-infinity ties share all mass; malformed/all-invalid inputs return explicit errors; fixed unit draws and seeded RNG sequences are reproducible. Top-K uses a bounded heap and Top-P includes the probability-threshold crossing token.

| Vocab / mode | Time | Allocation |
|---|---:|---:|
| 32K greedy | 47--51µs | 0 B |
| 32K Top-K=40 | 69--70µs | 1.36KiB |
| 32K unrestricted Top-P | 3.9--4.0ms | 1.50MiB |
| 128K greedy | 188--197µs | 0 B |
| 128K Top-K=40 | 238--244µs | 1.36KiB |
| 128K unrestricted Top-P | 19--20ms | 6.0MiB |

The result supports Top-K-first composition as the bounded candidate. Unrestricted Top-P necessarily sorts the full surviving vocabulary in this implementation and is retained for correctness/reference use, not as a default performance path. Generic generation remains greedy until model/server call sites receive an explicit opt-in configuration and full-token gates.

## Pending real-model rows

The Qwen3-30B-A3B MLX4 asset is unavailable on this host. Cold/warm decode, real route traces, upload bytes and full-token throughput remain explicitly pending rather than inferred from synthetic work. When the checkpoint is installed, these replay and selected-expert fixtures are the fixed controls for choosing LRU/LFU/layer-aware policy and any broader persistent-kernel experiment.
