# KVBoost application plan for go-pherence

KVBoost reference: <https://pythongiant.github.io/KVBoost/> and <https://github.com/pythongiant/kvboost>.

KVBoost combines four ideas:

1. **Chunk-level prompt hashing** — split prompts into chunks, hash each chunk, and look up previously computed K/V blocks.
2. **KV cache reuse** — skip transformer work for matching prompt chunks; only novel suffix tokens run through the model.
3. **Faster attention for novel tokens** — KVBoost uses FlashAttention-2; in go-pherence this maps to our NVIDIA/Vulkan attention kernels and batching work.
4. **KV page/offload** — evict long-context KV blocks to CPU RAM and move them back asynchronously when needed.

It also advertises AWQ layer streaming. In go-pherence the closest equivalent is our existing mmap/on-the-fly MLX loading, hybrid `--gpu-layers`, Qwen/Gemma GPU weight caches, and placement policy.

## What applies directly

### 1. Prefix/chunk KV reuse

This is the highest-value idea for our interactive/chat workloads. We repeatedly see prompts with stable prefixes:

- system prompt and chat template;
- tool/developer instructions;
- long context documents;
- repeated benchmark prompt prefixes;
- MTP verifier prompts where the prompt prefix is unchanged and only draft suffix changes.

Instead of re-prefilling all tokens, cache K/V for token chunks and resume from the longest cached prefix.

Proposed package/API:

```text
runtime/kv/reuse.go
```

Core types:

```go
type ChunkKey struct {
    ModelID     string
    Backend     string // cpu, nvidia, vulkan
    DType       string // f32, bf16, turboquant, etc.
    LayerLayout string // hash of layer count, head dims, KV sharing, sliding/full pattern
    TokenHash   uint64 // rolling hash over chunk tokens plus previous hash
    ChunkSize   int
    EndPos      int
}

type ChunkEntry struct {
    Key       ChunkKey
    Tokens    []int
    SeqLen    int
    KV        KVSnapshot
    Hidden    []float32 // optional final activation/pre-norm hidden
    LastUse   uint64
    Bytes     int64
}
```

The chunk key must include a *prefix-chain hash*, not just the chunk tokens, so identical chunks at different prefix positions do not alias incorrectly.

### 2. MTP verifier reuse

MTP naturally benefits from KV staging and rollback. We already have:

- `runtime/kv` staging/rollback helpers;
- `AcceptMTPDraft` and accepted-prefix semantics;
- Gemma4 prompt context capture;
- Qwen native-MTP smoke and Qwen forward state.

KVBoost-like reuse should be inserted before the verifier/prompt prefill:

1. tokenize/wrap prompt;
2. find longest cached prefix chunk chain;
3. restore K/V + final activation/state for cached prefix;
4. run only the novel suffix through prefill;
5. store new chunk endpoints;
6. run MTP drafter/verifier from that state.

For Gemma4 E4B this can reduce long-prompt real-prompt smoke from seconds to near suffix-only time in repeated conversations. For 31B/Qwen it is more important because full prompt prefill is expensive.

### 3. CPU KV page/offload

We already have CPU-side KV caches and TurboQuant compression. A KVBoost-inspired page layer should be added as a separate storage tier:

```text
GPU KV resident -> CPU pinned/heap KV -> compressed CPU KV -> mmap/disk optional later
```

Initial scope should be CPU heap only, not async DMA:

- keep current per-layer GPU KV for resident layers;
- snapshot completed chunks to CPU `KVSnapshot`;
- for long contexts, keep only recent GPU KV window and restore older pages to CPU attention fallback when needed;
- later add asynchronous prefetch around NVIDIA streams.

### 4. Layer/weight residency policy

KVBoost’s AWQ streaming maps to our Qwen/Gemma layer residency problem:

- Gemma4 31B: compact LM head + selected transformer layers;
- Qwen3.6 27B: MLX direct cache can hold only a subset of packed weights;
- MoE: expert cache already behaves like a streaming residency tier.

The next improvement is a placement policy that chooses weights by decode/prefill hotness instead of first-fit:

- keep embeddings/LM head if useful;
- keep early layers for prompt prefill if prefix reuse misses;
- keep all weights for repeated layers if the cache is reused across decode steps;
- avoid evicting a resident prefix for one-off later-layer uploads.

The current Qwen MLX cache already moved to no-evict admission for exactly this reason.

## What does not map directly

- FlashAttention-2 is CUDA-specific; go-pherence has hand-written PTX kernels and should improve its own attention kernels rather than importing FlashAttention.
- KVBoost’s HuggingFace monkey-patching approach is irrelevant; go-pherence owns the decode loop.
- AWQ-specific streaming is not directly applicable; our primary packed format is MLX affine 4-bit plus GPTQ/NVFP4 paths.

## Implementation phases

### Phase 1 — CPU/GPU-neutral KV snapshot contract

Add a compact snapshot type under `runtime/kv`:

```go
type LayerKVSnapshot struct {
    K []float32
    V []float32
    SeqLen int
    KVDim int
}

type Snapshot struct {
    Layers []LayerKVSnapshot
    SeqLen int
}
```

For Gemma4 variable KV width, snapshot each source layer with its actual `LayerKVDim`.

Acceptance criteria:

- snapshot/restore roundtrip tests;
- overflow/shape validation;
- support empty/shared-KV layers;
- no GPU dependency.

### Phase 2 — token chunk hash and in-memory reuse cache

Add chunk hash utilities:

```go
func HashTokenChunk(prev uint64, tokens []int) uint64
func ChunkTokens(tokens []int, chunkSize int) [][]int
```

Add an LRU bounded by bytes and entries.

Acceptance criteria:

- deterministic hash tests;
- prefix-chain alias tests;
- LRU eviction tests;
- model/layout mismatch rejection.

### Phase 3 — Gemma4 prompt context reuse

Integrate reuse with:

- `LlamaModel.BuildMTPPromptContext`;
- `GPUModel.BuildMTPPromptContext`;
- `llmgen -mtp-real-prompt`.

New flags:

```text
-mtp-kv-reuse
-mtp-kv-chunk-size 64
-mtp-kv-cache-mb 1024
```

Initial implementation can be CPU snapshot only. If a GPU prompt context is built, copy GPU-resident K/V back to CPU snapshot as we already do for MTP context.

Benchmarks:

- Gemma4 E4B repeated 207-token prompt: first run stores chunks, second run should prefill only suffix and report reused token count.
- Gemma4 31B short prompt: repeated prompt should avoid the full prefill cost.

### Phase 4 — Qwen native-MTP reuse

Qwen state is not just full-attention KV; it also has linear-attention recurrent state. Extend snapshot to include:

```go
type Qwen35StateSnapshot struct {
    FullK [][]float32
    FullV [][]float32
    Linear []Qwen35LinearAttentionState
    Pos int
    Hidden []float32
}
```

This is essential: Qwen3.6 has many linear-attention layers, so KVBoost-style reuse for Qwen must snapshot recurrent linear states too.

Acceptance criteria:

- clone/restore equivalence tests;
- one-token continuation parity after restore;
- MTP acceptance path still validates.

### Phase 5 — page/offload policy

After in-memory reuse works, add tiered pages:

- resident current-window GPU K/V;
- CPU snapshot pages;
- optional TurboQuant compressed pages.

Initial goal is correctness and memory control, not async DMA.

## Recommended first implementation slice

Start with Gemma4 E4B because it is fast and fully GPU-resident:

1. add `runtime/kv.Snapshot` and LRU chunk cache;
2. add `BuildMTPPromptContextWithReuse` for CPU/GPU Gemma4;
3. expose `llmgen -mtp-kv-reuse` for `-mtp-real-prompt`;
4. benchmark the same long prompt twice:
   - first run: normal prefill, cache fill;
   - second run: high reused-token count, near-zero prefix prefill;
5. port the same snapshot concept to 31B stress path;
6. extend Qwen snapshots to include linear recurrent state.

## Expected impact

For repeated/chat-style prompts:

- TTFT/prompt prefill can improve by the proportion of reused prefix tokens.
- Gemma4 E4B repeated 207-token prompt should go from ~9.25s prefill to suffix-only time.
- Gemma4 31B and Qwen3.6 benefit more because prefill is much more expensive.

For single-turn unique prompts, KVBoost-style reuse does not help much; Qwen/Gemma GPU weight placement and attention kernels remain the main bottlenecks.
