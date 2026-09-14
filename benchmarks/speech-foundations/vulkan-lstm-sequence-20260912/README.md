# Vulkan one-direction LSTM sequence foundation

`VkLSTMSequenceF32` extends the model-free recurrent coverage to one complete forward or reverse unprojected PyTorch-IFGO sequence. It owns no model policy: callers supply resident input, input/hidden weights, separate input/hidden biases, mutable initial hidden/cell state and a destination slice selected by output width/offset.

## Contract

One 256-lane workgroup owns the recurrence, so hidden size is 1–256 and no inter-workgroup state synchronization is required. Frames are 1–4096 and input width 1–512. The shader preserves separate input and hidden projection accumulators through `(inputProjection+biasIH)+(hiddenProjection+biasHH)`, matching the existing CPU composition rather than merging reduction order. Forward and reverse directions write their original frame indices; disjoint output offsets permit concatenated bidirectional output.

All input/weight/bias buffers must be disjoint from mutable hidden/cell and output buffers. Hidden, cell and output are mutually disjoint. Exact shape/storage, offset and fixed one-workgroup dispatch are checked before recording. Stable sigmoid/tanh formulas avoid exponential overflow for finite gate values. No host content scan or fallback occurs.

## Offline verification

- Five sequence geometries cover frames1–9, input widths1–60 and hidden widths1–128, both directions, with 878 finite output comparisons and terminal-state/frame-position checks.
- Tests cover eight descriptors, six-word push ABI, forward/reverse flags, output offsets, nil/rank/shape/storage and alias rejection, cancellation, plan retention through drain and device limits.
- All 21 embedded shader contracts pass. The shader uses local size `256×1×1`, 1024 shared bytes, eight storage bindings and 24 push bytes. Stored/embedded SPIR-V SHA-256 is `9f71b2b6ecd71e9cb4d0df3cc5a1d1d49256dd3e266edaa2a5eb22e108127cd6`.

No Vulkan device, checkpoint, multi-layer owner, trained inference, service, deployment, push or performance run was used. Static tests do not establish native transcendental parity. A complete owner must still upload immutable layers, initialize/copy states, compose both directions and layers, handle drain/close, and compare against CPU/Torch before any placement decision.
