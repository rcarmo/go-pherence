# F32 register-tile promotion — 13 September 2026

`whisper.NewVulkanEncoder` now uses the previously qualified 32×32 F32 register-tiled linear kernel. Vulkan remains explicit at the caller and service-profile levels. F32 remains the serving precision; Q8 constructors remain research-only after the natural-podcast parity failures.

The promotion changes only the private linear-kernel selection inside the resident encoder. The prior 16×16 F32 path remains available to the retained comparison test. The existing qualification established:

- full Turbo projection speedups of 2.055–2.151× with bit-exact outputs;
- a 14.244→8.298s isolated complete-encoder improvement (`1.716×`);
- bit-exact full encoder output;
- exact retained tokens and timestamps on four public speech fixtures;
- unchanged swap counters in the isolated confirmation;
- passing operator, plan, shader, lifecycle and rollback tests.

This change does not promote Q8, enable Vulkan automatically, alter backend identity fields, deploy a service or claim the planned ≥8× whole-ASR target. Existing durable profiles that explicitly select Vulkan receive the faster F32 implementation after the binary/backend hash changes.
