# Community-1 hybrid Vulkan embedding owner

`community1.NewVulkanEmbedding` defines the first usable explicit placement boundary around the resident WeSpeaker ResNet34 trunk. Fixed-shape CNN work stays on Vulkan; mask-dependent statistics pooling and the relatively small final embedding projection use the existing checked CPU path. No service or model default selects it.

The constructor validates fixed frame geometry and the complete projection before Vulkan allocation, copies projection weights/bias, then creates the resident trunk owner. Copies share one serialized lifetime. `Forward` rejects malformed masks before expensive trunk work, runs one trunk upload/plans/download, applies exact existing `StatsPool` semantics, and projects each admitted row through checked `simd.GemvRows`. It returns the same raw unnormalised embeddings and support metadata as the CPU API.

The placement is intentional rather than a claim that every neural operation belongs on GPU: masks vary by diarization call, pooling has source-specific float32 reduction/indexing semantics, and the final matrix is small relative to the CNN. Whole-job measurements may later justify a different split.

## Model-free verification

- Constructor preflight rejects nil/cancelled context, nil source, invalid frames and nonfinite projection values before Vulkan access.
- Forward admission rejects fixed-frame mismatches and malformed/nonfinite masks before calling the trunk.
- Shared-copy ownership tests cover permanent close state, cleared host projection, idempotent close and context-bounded serialization.
- The combined Community Vulkan gate passes operator ABI/schedules, block/trunk layout and ownership, the 19-shader inventory, full mock-only Vulkan, ten shuffled repetitions, vet, arm64 cross-build and documentation links.

No Vulkan device, trained checkpoint, audio, embedding comparison, service, deployment, push or performance run was used. Native trunk output, hybrid embedding parity, device-loss recovery, corpus quality and CPU/full-Vulkan/hybrid timing remain unqualified.
