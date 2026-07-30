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

## Pending real-model rows

The Qwen3-30B-A3B MLX4 asset is unavailable on this host. Cold/warm decode, real route traces, upload bytes and full-token throughput remain explicitly pending rather than inferred from synthetic work. When the checkpoint is installed, these replay and selected-expert fixtures are the fixed controls for choosing LRU/LFU/layer-aware policy and any broader persistent-kernel experiment.
