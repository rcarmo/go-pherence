package kv

import simd "github.com/rcarmo/go-pherence/backends/simd/runtime"

// TurboQuantCapabilities reports the native CPU features relevant to the
// TurboQuant KV cache path. It is owned by runtime/kv so command/server
// readiness surfaces do not duplicate SIMD-runtime interpretation details.
type TurboQuantCapabilities struct {
	Arch     string `json:"simd_arch"`
	Rotation bool   `json:"simd_rotation"`
	Vec      bool   `json:"simd_vec"`
	AVX2     bool   `json:"simd_avx2"`
	NEON     bool   `json:"simd_neon"`
	RVV      bool   `json:"simd_rvv"`
}

// RuntimeTurboQuantCapabilities returns the active native capability surface for
// TurboQuant rotations. Rotation is true when per-head dot products can route
// through the checked SIMD facade (AVX2/FMA, NEON, or RVV where available).
func RuntimeTurboQuantCapabilities() TurboQuantCapabilities {
	caps := simd.RuntimeCapabilities()
	return TurboQuantCapabilities{
		Arch:     caps.Arch,
		Rotation: caps.HasDot,
		Vec:      caps.HasVec,
		AVX2:     caps.HasAVX2,
		NEON:     caps.HasNEON,
		RVV:      caps.HasRVV,
	}
}
