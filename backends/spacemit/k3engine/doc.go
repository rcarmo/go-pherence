// Package k3engine is the pure-Go (no-cgo) transformer inference engine for the
// SpaceMIT K3 SoC (MilkV Jupiter 2). It composes the low-level kernel packages
// — ime2 (IME int8 GEMM), rvv (RVV SIMD), tcm (scratchpad) — into a
// Q4_K/Q6_K/INT8 quantized decode loop with RoPE/SiLU math, driven by the
// aipool worker pool and config feature flags. Run() is the entry point used by
// cmd/ime2run.
//
// Sub-packages:
//   - aipool:   the engine's TCM-aware worker pool
//   - config:   IME2_* environment feature flags
//   - q4kcshim: optional cgo shim linking llama.cpp for kernel validation
//
// Not to be confused with backends/k3, which is the high-level ORT/Vulkan
// compute-backend dispatch layer for the same SoC.
package k3engine
