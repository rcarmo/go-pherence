# TurboFieldfare audit

This audit reviews [`drumih/turbo-fieldfare`](https://github.com/drumih/turbo-fieldfare) at commit [`f8abc442`](https://github.com/drumih/turbo-fieldfare/tree/f8abc4422e33a8808d5a5c1032a0e97ed5aa5118). TurboFieldfare is a model-specific Swift/Metal runtime for Gemma 4 26B-A4B on Apple Silicon; go-pherence is a portable Go runtime with CPU, NVIDIA and embedded backends. The useful comparison is therefore about scheduling and memory policy, not a line-by-line kernel port.

TurboFieldfare's central constraint is unusually sharp: keep roughly 1.35GB of shared weights resident on an 8GB Mac and stream the 12.9GB routed-expert pool from NVMe. That design produced several techniques worth testing in go-pherence, but it also produced a useful collection of negative results. Its [optimisation journey](https://github.com/drumih/turbo-fieldfare/blob/f8abc4422e33a8808d5a5c1032a0e97ed5aa5118/docs/OPTIMIZATION_JOURNEY.md) is unusually candid about both.

The experiments prompted by this audit are complete and tracked in [TurboFieldfare adoption results](turbo-fieldfare-adoption-results.md). The headline dispositions are: retain split-KV long-context decode attention and bounded Qwen3.6 prefill planning; keep LRU as the cache default; reject the persistent single-projection kernel and FP16 rings on speed; and keep out-of-core streaming plus generic sampling explicit/non-default until runnable Qwen3-30B full-token gates exist.

## What already maps cleanly

Several of TurboFieldfare's strongest ideas are already present in go-pherence:

| TurboFieldfare technique | go-pherence equivalent | Current state |
|---|---|---|
| Batched/chunked prompt execution | `model/cpu_prefill.go`, `model/batch_prefill.go`, model-specific encoder batches | Dense CPU/GPU prefill exists. Generic GPU MoE prefill is still excluded. |
| Group token/expert work by expert | DiffusionGemma `expertUsers`, selected-expert arrays and grouped GGUF GPU work | Implemented for DiffusionGemma CPU/GPU paths. Generic single-token MoE has no batch to group. |
| Bounded resident expert cache | `backends/nvidia/runtime.ExpertPool`, DiffusionGemma `ExpertLRUCache` | Implemented with LRU, hit/miss/eviction counters and budget accounting. |
| Overlap next-layer data work | NVIDIA prefetch stream/events; DiffusionGemma next-layer mmap prefetch | Present, although not consistently used by every model path. |
| Narrow, shape-compatible fusions | Fused QKV, activation, residual, attention and model-island kernels | The same policy has worked here; broad fusions remain parity-gated. |
| Shape-specialised quantised matmul | MLX/GPTQ/NVFP4/FP8/GGUF batch kernels and shape dispatch | Implemented and measured in the matmul programme. |
| Avoid repeated vocabulary work | Compact/resident LM heads, GPU argmax/top-k and DiffusionGemma device sampling | Several model-specific paths already avoid downloading or rescanning full logits. |
| Benchmark the complete request | `benchmarks/matmul`, model fixtures and hardware-gated parity tests | Current policy already rejects isolated wins that regress the full model. |

This overlap matters: the highest-value outcome is not another kernel rewrite. It is applying TurboFieldfare's scheduling discipline to the places where go-pherence still has a policy gap.

## Ranked opportunities

### 1. Replace generic per-expert GEMV chains with persistent routed-MoE work

TurboFieldfare's largest decode-side kernel gain came from giving the GPU many independent routed-expert rows to claim from one persistent dispatch. Its routed-compute phase fell from roughly 239ms to 60ms and decode improved from 2.188 to 3.313 tok/s. A superficially tidy cooperative kernel was slower because it removed parallel work.

go-pherence's generic [`moeForwardGPU`](../model/moe_gpu.go) still runs gate, up, activation and down as a sequence of GEMVs for each selected expert. DiffusionGemma has grouped expert metadata and scatter kernels, but that machinery is model-specific and does not cover ordinary Qwen/Gemma MoE decode. This is the largest direct compute gap exposed by the comparison.

**Implementation sketch**

* Flatten selected experts into a compact work table containing pool slot, route rank, output row and weight.
* Dispatch persistent thread blocks that claim gate/up/down row work atomically or from bounded tiles.
* Reuse the input vector and packed expert metadata across the selected set; keep gate/up fusion narrow and shape-specific.
* Accumulate into per-route scratch, then reduce routes in deterministic rank order so cache/scheduling order cannot change output.
* Use the existing per-expert GEMV chain as the parity oracle and fallback.

**Targets:** new kernels under `backends/nvidia/ptx`, runtime work-table buffers beside the existing expert scatter paths, and `model/moe_gpu.go`.

**Validation:** exact or tolerance-bounded expert outputs at realistic non-zero offsets; top-k tails; cold/warm pools; route-order invariance; kernel/upload/sync counters; Qwen3-30B full-token throughput.

**Expected benefit:** high when routed expert compute is material; TurboFieldfare measured a 51% decode gain from this structural change, but NVIDIA geometry and go-pherence's packed layout require independent measurement.

**Risk/cost:** high kernel cost, medium numerical risk and medium scheduling risk.

### 2. Add LFU eviction to NVIDIA expert pools

TurboFieldfare's bounded cache moved from LRU to LFU and reduced routed-expert I/O from 72.6ms to 64.8ms per token in its paired benchmark. Its implementation uses whole-run frequency, with recency as the tie-breaker; a 64-token frequency window did not produce a reliable full-run improvement.

go-pherence's [`ExpertPool`](../backends/nvidia/runtime/expert_pool.go) is LRU-only and already owns fixed slots, budget accounting and hit/miss/eviction statistics. The topology differs, though: TurboFieldfare has 16 slots *per layer*, while go-pherence uses one global pool keyed by `layer*experts + expert`. LFU is therefore a useful controlled experiment, not a policy whose measured gain transfers directly.

**Implementation sketch**

* Add `ExpertCachePolicy` (`lru`, `lfu`) to `backends/nvidia/runtime`.
* Track use count by pool key; select the lowest count, then oldest use, when evicting.
* Keep LRU as an explicit control and avoid a moving-window policy initially.
* Compare global LFU with a layer-aware admission or quota variant if full-run frequency over-favours early/hot layers.
* Expose policy and slot count through the existing GPU/model options rather than an environment-only switch.

**Targets:** `backends/nvidia/runtime/expert_pool.go`, `model/moe_gpu.go`, placement/reporting and focused expert-pool tests.

**Validation:** deterministic eviction tests; exact CPU/GPU MoE output; per-layer and global hit distributions; cold/warm Qwen3-30B route-set benchmarks; miss bytes, upload time and tok/s over at least 64 generated tokens.

**Expected benefit:** medium. The mechanism is small, but the result depends on route repetition, pool pressure and global-versus-per-layer cache shape.

**Risk/cost:** low implementation cost, low numerical risk, medium policy risk.

### 3. Use layer-type-aware rings for sliding-attention KV

TurboFieldfare stores sliding-window layers in fixed FP16 rings and keeps only full-attention layers linear. For its Gemma 4 layout this saved roughly 575--591MiB against linear FP16 storage. Its attempted K4/V4 cache saved only about 82MiB at 4K, grew larger at long context because it covered every attention layer, and failed the quality gate.

go-pherence already clips attention reads to `SlidingWindow`, but ordinary CPU and NVIDIA KV buffers are F32 and continue to grow until the configured context limit. TurboQuant compresses older rows, which is a different trade-off. The low-risk first step is therefore an exact F32 ring that changes capacity and addressing only. An FP16 ring could halve those bytes again, but it is a separate precision experiment with its own logit/token gates.

**Implementation sketch**

* Introduce a KV storage interface with linear and ring implementations rather than scattering modulo arithmetic through model code.
* Implement F32 ring storage first; evaluate FP16 only after exact ring parity is established.
* Size sliding layers to `slidingWindow + maxPrefillChunk`; keep full layers linear.
* Preserve separate K and V storage even when projection weights alias, because normalization and RoPE can diverge.
* Make prefill write ranges split at ring wrap; decode attention receives logical length and physical start.
* Keep TurboQuant as a separate policy. Do not combine both until each has independent parity and memory results.

**Targets:** generic generation state in `model/llama.go`, `forward_layer.go`, CPU/GPU attention wrappers, MTP checkpoint/rollback surfaces and `runtime/kv`.

**Validation:** sequential versus F32-ring logits/tokens across multiple wraps; Gemma sliding/full mixed layers; chunked prefill crossing wrap; MTP rollback; exact byte accounting; 4K/16K/32K RSS. An FP16 follow-up also needs selected-logit and long-generation token gates.

**Expected benefit:** high memory reduction from bounded SWA capacity; an FP16 follow-up can halve the ring payload again. Speed changes should be small unless reduced memory pressure avoids paging.

**Risk/cost:** high implementation cost and medium addressing/state risk for F32 rings; FP16 adds medium numerical risk.

### 4. Prototype explicit expert streaming for out-of-core MoE

TurboFieldfare's decisive result was replacing demand-paged expert `mmap` reads with bounded parallel `pread` into preallocated, page-aligned slots. Cold expert reads fell from 9.88ms to 2.79ms, and its streaming simulation improved from 0.50 to 3.97 tok/s. The resident core stayed mapped; only routed experts used explicit reads.

go-pherence currently mmap-loads safetensors/GGUF and can prefetch pages, while GPU expert caches upload from already materialised model weights. That works when RAM can hold the checkpoint, but it does not provide TurboFieldfare's 14.3GB-on-8GB operating mode.

**Implementation sketch**

* Start with one pinned checkpoint/layout, not generic safetensors slicing at runtime.
* Repack routed experts into per-layer files with a validated manifest and contiguous gate/up/down regions.
* Allocate fixed aligned host slots; use `ReadAt`/`pread` semantics to fill misses concurrently.
* Separate cache planning from I/O execution so hits can run before miss-dependent work.
* Upload from slots to the existing `ExpertPool`; do not retain duplicate decoded copies.
* Record read bytes/time, hit ratio, upload time and overlap wait separately.

**Targets:** a new loader/expert-stream package, model installation/repack tooling, `model/moe_gpu.go`, placement budgets and expert-cache reporting.

**Validation:** manifest hashes and offset guards; realistic non-zero tensor offsets; exact expert outputs; cold-cache repeated runs; memory/RSS ceiling; corruption and short-read tests.

**Expected benefit:** potentially transformative for MoE checkpoints larger than RAM/VRAM; little value for fully resident models.

**Risk/cost:** high cost and operational complexity; low numerical risk if bytes/layout are unchanged; high I/O-policy risk.

### 5. Run expert-cache hits before misses and overlap safe resident work

TurboFieldfare's coarse schedule runs the always-resident shared MLP while expert misses are read, then processes cache hits before miss-backed experts. Shared-MLP overlap improved 4.404 to 4.736 tok/s; hit-first scheduling measured 5.169 versus 4.518 tok/s. Launching each expert as soon as its read completed was slower and changed output.

go-pherence currently uploads a selected expert synchronously on a miss in `moeForwardGPU`, then executes selected experts in route order. It defers some CPU-fallback uploads for CUDA context safety, but does not make hit-first execution an explicit schedule.

**Implementation sketch**

* Build a selected-expert plan using `Peek` before mutating cache statistics.
* Partition cached and missing experts while preserving a deterministic final accumulation order.
* Start bounded miss uploads, execute cached experts, then wait and execute newly resident experts.
* Overlap only independent resident work (for example a shared expert branch). Avoid per-expert completion callbacks.

**Targets:** `model/moe_gpu.go`, `backends/nvidia/runtime/expert_pool.go`, potentially DiffusionGemma's FP8 expert cache.

**Validation:** exact weighted output order; CUDA lifetime/race tests; cold/warm route sets; kernel/upload/sync counters; whole-token timing.

**Expected benefit:** medium on mixed hit/miss route sets, negligible when everything hits or misses.

**Risk/cost:** medium cost, medium synchronization risk, medium numerical risk if accumulation order changes.

### 6. Split long-context decode attention across KV chunks

TurboFieldfare divides decode KV across threadgroups, computes online-softmax partials and merges them in a second pass. It reports about 3.3x for sliding-window attention, 4.1x for full attention at 4K, and roughly 28% lower attention GPU time at long context.

go-pherence already has model-specific full-attention kernels and rejected a three-pass no-score Whisper kernel because recomputation made it 5--8x slower. That rejection does not invalidate split-KV decode: TurboFieldfare stores partial max/sum/output and merges them, avoiding QK recomputation.

**Implementation sketch**

* Prototype only for causal decode (`queryLen=1`) and long contexts.
* One kernel writes per-chunk `{max, sum, weighted V}` partials; a small merge kernel applies stable online-softmax combination.
* Specialise GQA so query heads sharing a KV head reuse K/V loads where the backend permits it.
* Dispatch only above a measured sequence threshold; short contexts remain on the current kernel.

**Targets:** NVIDIA attention PTX/runtime first, then CPU/RVV if profiling justifies it.

**Validation:** CPU scalar oracle across 1, 31, 32, 33, 127 and multi-chunk tails; sliding/full masks; softcap; long-context token parity; 512/2K/4K/16K timings.

**Expected benefit:** high for long-context decode, near zero for short contexts.

**Risk/cost:** medium-high kernel cost and medium numerical risk from reduction order.

### 7. Chunk prefill by memory budget rather than only by capability

TurboFieldfare's 128-token chunk reduced a 1,017-token prefill from 92.89s to 52.35s. go-pherence already batches prompt projections, which captures the main arithmetic benefit, but it normally treats the full prompt as one batch and allocates B-scaled scratch.

A bounded chunk planner would help long prompts, memory-constrained GPUs and future MoE prefill. It is less urgent for ordinary short prompts because go-pherence's existing batched prefill already avoids token replay.

**Implementation sketch**

* Derive chunk size from scratch/KV bytes and backend limits; expose 32/64/128 controls for measurement.
* Carry KV and absolute positions across chunks; return only the final prompt activation.
* Reuse one chunk workspace rather than allocating prompt-sized temporaries.
* Add MoE grouping within each chunk when generic MoE prefill becomes available.

**Targets:** `model/cpu_prefill.go`, `model/batch_prefill.go`, GPU model workspaces and placement estimates.

**Validation:** chunked versus full-batch logits/KV; non-divisible tails; sliding-window boundaries; multimodal embeddings; peak RSS/VRAM and TTFT.

**Expected benefit:** high memory reduction, workload-dependent TTFT improvement.

**Risk/cost:** medium cost, medium state-boundary risk.

### 8. Design generic sampling around one bounded vocabulary pass

TurboFieldfare recovered sampled decode by removing repeated vocabulary scans, eventually using a staged Top-P plus Top-64 path. Generic go-pherence LLM generation is currently greedy-only; temperature requests are rejected by the server. This is therefore design guidance for a missing feature, not an optimisation of an existing generic sampler.

The transferable rule is not to copy its Gumbel implementation. When generic sampling is added, count full-vocabulary passes, transfers and allocations from the outset. Keep Top-P/Top-K ordering explicit and fuse softcap, temperature or candidate selection only when one bounded pass preserves the requested distribution.

**Targets:** future generic generation sampling contract, LM-head output boundaries and NVIDIA compact-head paths. Existing DiffusionGemma device sampling is a model-specific reference, not a drop-in implementation.

**Validation:** fixed-seed reference fixtures; Top-P then Top-K ordering; temperature zero equivalence to greedy; ties/NaNs; vocab tails; statistical distribution checks; kernel plus full-token timing.

**Expected benefit:** prevents a known performance trap when sampled generation is implemented; no current generic decode speedup.

**Risk/cost:** medium semantic risk and medium implementation cost.

## Techniques not to copy directly

* **Metal TensorOps and MPP kernels** are Apple-specific. Their broader lesson -- stage a small dequantised tile into the hardware matrix primitive -- is already reflected in go-pherence's quantised batch kernels and should be applied per backend, not abstracted as an Apple-shaped API.
* **`F_RDADVISE` policies** were unstable and remain off by default in TurboFieldfare. go-pherence should keep mmap prefetch advisory and measured, not promote read-ahead hints without cold/warm long-run evidence.
* **Speculative cross-layer expert reads** were slower and harmed prefill; adjacent layers shared too few experts. Route history is useful for eviction, not prediction.
* **Packed K4/V4 KV** failed both memory and quality gates in the source project. go-pherence's TurboQuant has a different rotation/protection design and its own fixtures, so TurboFieldfare neither validates nor invalidates it; the two policies must remain separately measured.
* **Large monolithic fusions** reduced throughput. Keep combining operations only when shapes, lifetimes and synchronization boundaries align.
* **More cache slots** are not automatically better. TurboFieldfare's 32-slot run collapsed on an 8GB host because memory pressure outweighed hit-rate gains. Slot counts must be budgeted against the complete process.
* **Allocation-count wins** do not imply latency wins. TurboFieldfare reduced one prefill path from 21,217 allocations to two and made it about 9% slower.

## Recommended sequence

1. Prototype a persistent routed-MoE kernel against the existing per-expert GEMV oracle.
2. Add LRU/LFU controls and measure the global Qwen MoE pool, including per-layer hit distributions.
3. Prototype exact F32 SWA KV rings behind a storage interface, beginning with CPU Gemma fixtures before touching GPU state; treat FP16 as a separate follow-up.
4. Add a split-KV NVIDIA decode-attention experiment with a strict sequence threshold.
5. Design a pinned out-of-core expert format and cold-I/O benchmark before integrating it into generation.
6. Add bounded prefill chunks when long-prompt scratch or MoE prefill becomes the measured constraint.
7. Use the single-pass lesson when generic sampled generation is implemented.

Each experiment needs a paired baseline, cold and warm cases where I/O is involved, exact token/output checks, memory accounting and a default-off control until the full model improves. TurboFieldfare's most reusable technique is that discipline.

## Source references

* [System design](https://github.com/drumih/turbo-fieldfare/blob/f8abc4422e33a8808d5a5c1032a0e97ed5aa5118/docs/SYSTEM_DESIGN.md)
* [Benchmarks](https://github.com/drumih/turbo-fieldfare/blob/f8abc4422e33a8808d5a5c1032a0e97ed5aa5118/docs/BENCHMARKS.md)
* [Optimisation journey](https://github.com/drumih/turbo-fieldfare/blob/f8abc4422e33a8808d5a5c1032a0e97ed5aa5118/docs/OPTIMIZATION_JOURNEY.md)
* [Runtime controls](https://github.com/drumih/turbo-fieldfare/blob/f8abc4422e33a8808d5a5c1032a0e97ed5aa5118/docs/RUNTIME_CONTROLS.md)
* [`PreadExpertStreamer`](https://github.com/drumih/turbo-fieldfare/blob/f8abc4422e33a8808d5a5c1032a0e97ed5aa5118/Sources/TurboFieldfare/Infrastructure/Streaming/PreadExpertStreamer.swift)
* [`KVCacheManager`](https://github.com/drumih/turbo-fieldfare/blob/f8abc4422e33a8808d5a5c1032a0e97ed5aa5118/Sources/TurboFieldfare/Runtime/KVCache/KVCacheManager.swift)
* [Decode attention](https://github.com/drumih/turbo-fieldfare/blob/f8abc4422e33a8808d5a5c1032a0e97ed5aa5118/Sources/TurboFieldfare/Kernels/Attention/Attention.swift)
* [Prefill runtime configuration](https://github.com/drumih/turbo-fieldfare/blob/f8abc4422e33a8808d5a5c1032a0e97ed5aa5118/Sources/TurboFieldfare/Runtime/Prefill/PrefillRuntimeConfig.swift)
