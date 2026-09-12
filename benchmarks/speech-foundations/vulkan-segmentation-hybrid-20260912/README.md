# Community-1 hybrid Vulkan segmentation feature owner

`community1.NewVulkanSegmentationFeatures` defines an explicit fixed-frame feature-stage boundary: the complete multi-layer/bidirectional LSTM runs through the resident Vulkan owner, while the copied powerset segmentation head stays on the existing checked CPU path. It consumes frame-major 60-wide SincNet features, not PCM, and no service/model default selects it.

Construction validates the checkpoint's recurrent/head geometry and copies the complete head before constructing Vulkan resources. `ForwardFeatures` checks exact fixed-frame input and head mode, runs the resident recurrent owner from zero state, then executes the unchanged CPU hidden layers, classifier and log-softmax. Calls and close are serialized through a shared state; close permanently blocks new calls and propagates unresolved recurrent cleanup without implicit drain or fallback.

## Model-free verification

- A complete pinned synthetic segmentation checkpoint reaches an injected recurrent constructor only after all source/head validation.
- Nil/cancelled context, nil/incomplete source, invalid geometry, nonfinite head values and nil recurrent constructors fail before Vulkan access.
- Forward admission rejects frame/length/mode/nonfinite input before recurrent work.
- Shared-copy lifecycle tests verify permanent close, state clearing, idempotence and context-bounded serialization.
- The combined Community Vulkan gate covers this owner, the full LSTM owner, CNN/hybrid owners, all checked operators and the 21-shader inventory. Full mock-only Vulkan, shuffled repeats, vet, arm64 cross-build, static validation and documentation links pass separately.

No Vulkan device, trained checkpoint, SincNet integration, audio, service, deployment, push or performance run was used. Native recurrent parity, complete PCM segmentation parity, strict intermediate repair, device loss and CPU/full-Vulkan/hybrid whole-job comparison remain open.
