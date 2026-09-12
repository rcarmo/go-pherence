# Community-1 hybrid Vulkan PCM segmentation composition

`community1.NewVulkanSegmentationPCM` joins the existing checked lowered-filter CPU SincNet frontend to the fixed-frame Vulkan-LSTM/CPU-head owner. It is an explicit experimental composition for one exact mono16k PCM window. Existing CPU paths, services and defaults are unchanged.

Construction validates checkpoint ownership, exact sample/grid geometry and lowered filters before constructing the Vulkan feature owner. The resulting object fixes both sample count and recurrent frame count. `ForwardPCM` accepts only explicit ordered-FMA SincNet modes and checked CPU head modes, runs the whole-window frontend, confirms the grid, then invokes resident recurrent inference and the copied CPU head. Close propagates recurrent-owner cleanup and clears the frontend/feature references only after success.

## Model-free verification

- A complete pinned synthetic checkpoint and lowered-filter set reaches an injected feature-owner constructor only after frontend/grid validation, with the exact derived recurrent frame count.
- Nil/cancelled context, nil checkpoint, invalid filters/sample count, nil constructor and a nil successful constructor result fail closed.
- Forward rejects wrong PCM extent or modes and propagates nested stopped-owner failure; close clears both references and remains idempotent.
- The combined Community Vulkan gate covers PCM composition, recurrent/head hybrid, LSTM/trunk owners, checked operators and all 21 shader contracts. Full mock-only Vulkan, shuffled repeats, vet, arm64 cross-build, static validation and documentation links pass separately.

No Vulkan device, trained checkpoint inference, public/private audio, service, deployment, push or performance run was used. The existing strict SincNet intermediate failures remain. Native recurrent parity, complete trained PCM output, diarization corpus quality, recovery and CPU/full-Vulkan/hybrid whole-job comparison remain open.
