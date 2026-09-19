# Backends

The [safety audit](../validation/repository-safety-audit-20260919.md) records current driver/loader fixes and remaining lifecycle limits; [CPU profiles](../performance/allocation-simd-audit-20260919.md) separate measured allocation improvements from proposed SIMD work. Hardware availability and a passing host build are different gates.

Backend support depends on the operation, model and host. Start with [selection and fallback](backend-selection.md). K3/SpacemiT execution requires its target hardware; portable packing tests and cross-builds do not validate IME instructions.

* [ARM CPU ISA feasibility for Gemma inference](arm-cpu-isa-feasibility.md)
* [Backend selection](backend-selection.md)
* [CIX P1 and Orange Pi 6 Plus CPU acceleration](cix-p1-orange-pi-6-plus.md)
* [GPU Compute Options](gpu-options.md)
* [NVFP4 / FP4 Support Track](nvfp4.md)
* [NVIDIA quantized runtime boundaries](nvidia-quant-boundaries.md)
* [SpacemiT IME2 Reverse Engineering](spacemit-ime2.md)
* [TurboQuant KV Cache Compression](turboquant.md)
* [Vulkan dispatch inventory](vulkan-dispatch-inventory.md)
* [Weight Budget Manager — Design](weight-budget.md)

[Documentation index](../README.md)
