# Validation

Latest continuation: [fourth-pass sizing, slot and diagnostic review](repository-safety-fourth-pass-20260919.md). Selected source coverage is 75/162 packages; 87 remain inventory/test-only. Earlier totals below are historical snapshots, not exhaustive clearance.

Further continuation: [cache, graph and native-boundary review](repository-safety-third-pass-20260919.md). Selected source sections now cover 59/162 packages, with 103 inventory/test-only. Native GGML/ORT risks are explicit open findings; host stubs do not validate their CGo implementations.

Latest continuation: [follow-on audit](repository-safety-followon-20260919.md). Selected source sections now cover 40 of 162 packages; 122 remain inventory/test-only. New tests cover HTTP admission, worker lifecycle, mapping close, finite comparisons, CPU fallback attention/convolution and graph buffer reuse. Earlier counts below describe the first pass.

Current results: [repository safety audit](repository-safety-audit-20260919.md), [all-package coverage](repository-safety-audit-coverage-20260919.md) and [GPU crash investigation](qwen3-gpu-crash-audit-20260919.md). The final host race sweep passed 90 packages; 71 have no tests. Only selected boundaries in 26 packages were source-inspected; 135 have inventory/test coverage only. GPU/K3 execution and issue #16 ownership findings remain open. See also the [CPU allocation/SIMD profile](../performance/allocation-simd-audit-20260919.md).

Start with [Validation gates](validation-gates.md). Host tests, hardware tests and compile-only checks answer different questions; none should be reported as a substitute for another.

* [Backend parity matrix](backend-parity-matrix.md)
* [BF16 parity expectations](bf16-parity.md)
* [Malformed-input coverage tracker](malformed-input-coverage.md)
* [Validation gates](validation-gates.md)
* [Validation and hardening status](validation-hardening.md)
* [Vulkan validation plan](vulkan-validation-plan.md)

* [Documentation and host-check audit, 2026-09-18](documentation-audit-20260918.md)

* [Source and checkpoint layout validation, 2026-09-19](model-layout-20260919.md)

* [Model layout follow-up audit, 2026-09-19](model-layout-audit-20260919.md)

* [Issues #3–#11 validation, 2026-09-19](issues-3-11-20260919.md)

[Documentation index](../README.md)
