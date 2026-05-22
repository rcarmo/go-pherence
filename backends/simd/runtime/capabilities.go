package simd

import (
	"runtime"

	"golang.org/x/sys/cpu"
)

// Capabilities describes the CPU features and SIMD entrypoints this binary can use.
// It is intentionally conservative: Go fallback paths remain valid when a feature
// is unavailable at runtime, even on an architecture with assembly files present.
type Capabilities struct {
	Arch          string
	HasAVX2       bool
	HasFMA        bool
	HasNEON       bool
	HasRVV        bool
	HasVec        bool
	HasDot        bool
	HasSGEMM      bool
	HasBF16       bool
	HasPack       bool
	HasRoPE       bool
	HasActivation bool
}

// HasDotAsm reports whether Sdot/Saxpy may use architecture-specific assembly.
var HasDotAsm = RuntimeCapabilities().HasDot

// HasSgemmAsm reports whether SIMD-accelerated SGEMM kernels are available
// on this runtime (amd64 AVX2+FMA, arm64 NEON). Callers must check it before
// invoking SgemmNN/SgemmNT because fallback architectures intentionally panic.
var HasSgemmAsm = RuntimeCapabilities().HasSGEMM

// RuntimeCapabilities returns the active SIMD capability set.
func RuntimeCapabilities() Capabilities {
	c := Capabilities{Arch: runtime.GOARCH}
	switch runtime.GOARCH {
	case "amd64":
		c.HasAVX2 = cpu.X86.HasAVX2
		c.HasFMA = cpu.X86.HasFMA
		c.HasVec = c.HasAVX2 && c.HasFMA
		c.HasDot = c.HasVec
		c.HasSGEMM = c.HasVec && hasSgemmAsm
		c.HasBF16 = c.HasVec
		c.HasPack = hasAvxPack
		c.HasRoPE = hasRoPEAsm && c.HasVec
		c.HasActivation = hasActivationAsm && c.HasVec
	case "arm64":
		c.HasNEON = cpu.ARM64.HasASIMD
		c.HasVec = c.HasNEON
		c.HasDot = c.HasNEON
		c.HasSGEMM = c.HasNEON && hasSgemmAsm
		c.HasBF16 = c.HasNEON
		c.HasPack = hasNeonPack
		c.HasRoPE = hasRoPEAsm && c.HasNEON
		c.HasActivation = hasActivationAsm && c.HasNEON
	case "riscv64":
		c.HasRVV = cpu.RISCV64.HasV
		c.HasDot = c.HasRVV
		// RVV hardware may be present, but HasVec/HasSGEMM stay false
		// until broader riscv64 vector kernels are enabled and parity-tested.
	}
	return c
}
