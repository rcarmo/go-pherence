# Community-1 resident Vulkan ResNet34 trunk owner

`community1.NewVulkanResNetTrunk` extends the fixed-shape BasicBlock ownership model to the complete WeSpeaker stem and 16-block ResNet34 trunk. It is explicit opt-in. Pooling and embedding projection remain on the host, and no existing model, service or default selects it.

## Graph and ownership

Construction validates the complete CPU-owned model and fixed Fbank frame count before querying Vulkan. It copies the stem and every block's convolution weights, prepares/copies all inference BatchNorm coefficients and uploads them into one aligned arena. The scratch layout uses one CHW input and three activation buffers at each of four spatial resolutions; buffers are reused only after their previous values are no longer read.

The owner builds 17 private plans: one two-stage stem and one six- or eight-stage plan for each BasicBlock. There are 104 dispatch stages in the supported `[3,4,6,3]` topology, including the three stride/channel projection shortcuts. `ForwardFrames` validates/transposes frame-major Fbank once, uploads once, executes the resident plans serially and downloads the final `[8*base,melBins/8,ceil(frames/8)]` CHW tensor once. Calls are serialized.

Close prevents new calls and tears down plans, arena and operators in reverse ownership order. Failed resources remain owned for a retry after explicit drain. There is no implicit device initialization, fallback, drain, restart or service wiring.

## Model-free verification

- Every pinned synthetic ResNet fixture yields 17 weight groups, 13 scratch tensors, 17 plans, 104 stages and exactly three projection blocks.
- Tests verify read-after-definition, exact final shape, aligned arena accounting and source-copy independence.
- Nil/cancelled context, nil/invalid source, zero frames, nonfinite stem data and nil plan constructors fail before device access.
- Shared-copy lifecycle tests verify permanent stop after partial close, retry ownership, idempotence and context-bounded serialization.
- The final focused gate covers both resident owners plus all checked Community Vulkan operators. The complete mock-only Vulkan suite, ten shuffled operator/owner repeats, vet, arm64 cross-build, 19-shader validation and documentation links pass separately.

No Vulkan device, trained checkpoint, audio, pooling/projection, service, deployment, push or performance run was used. Native block/trunk parity, full embedding equivalence, device-loss behavior and CPU/full-Vulkan/hybrid whole-job selection remain open.
