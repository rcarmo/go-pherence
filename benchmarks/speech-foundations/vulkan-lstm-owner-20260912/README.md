# Community-1 resident Vulkan LSTM owner

`community1.NewVulkanLSTM` composes the one-direction sequence primitive into a fixed-frame, multi-layer, optionally bidirectional unprojected LSTM owner. It is explicit opt-in; no segmentation model, service or default selects it.

## Ownership and scheduling

Construction validates the complete CPU-owned `LSTM`, copies every input/hidden weight and both PyTorch biases, allocates one resident arena for weights/state/layer outputs and creates one private plan per layer. Each plan has one stage per direction. Forward and reverse stages share their immutable layer input and write disjoint slices of the same frame-major output; each owns separate hidden/cell state.

`Forward` validates exact input and optional initial-state geometry, uploads input once, explicitly uploads caller state or zeros every state tensor on every call, executes layers serially, then downloads final sequence and terminal state. This preserves reset semantics and the reverse terminal state at frame zero. Calls are serialized. Close prevents new work, tears down plans/arena/kernel in reverse order and retains failed resources for retry after explicit drain.

## Model-free verification

- All six pinned synthetic LSTM fixture geometries produce exact weight/state/output metadata, one plan per layer and one stage per direction.
- Tests verify forward/reverse offsets, read ordering, final output shape, aligned arena accounting and source-copy independence.
- Nil/cancelled context, nil source, invalid frames, nonfinite weights and nil plan constructors fail before Vulkan access.
- Shared-copy lifecycle tests verify permanent stop after partial close, retry ownership, idempotence and context-bounded serialization.
- The combined Community Vulkan gate passes 21-shader contracts, all recurrent/CNN focused tests and the complete mock-only Vulkan suite; ten shuffled repetitions, vet, arm64 cross-build, static shader validation and documentation links pass separately.

No Vulkan device, trained checkpoint, segmentation head, audio, service, deployment, push or performance run was used. Native recurrent arithmetic and transcendental parity, complete SincNet→LSTM→head integration, device-loss behavior and CPU/full-Vulkan/hybrid whole-job comparisons remain open.
