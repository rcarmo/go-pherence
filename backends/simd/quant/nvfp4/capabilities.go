package nvfp4

import (
	"runtime"

	"golang.org/x/sys/cpu"
)

// Capabilities describes NVFP4 CPU execution features. It is conservative:
// scalar remains the only active path until architecture kernels land and pass
// parity tests.
type Capabilities struct {
	Arch       string
	HasAVX2    bool
	HasFMA     bool
	HasNEON    bool
	HasRVV     bool
	HasDecode  bool
	HasDequant bool
	HasGemv    bool
}

const (
	hasDecodeAsm  = false
	hasDequantAsm = false
	hasGemvAsm    = false
)

// RuntimeCapabilities returns the active NVFP4 CPU capability set.
func RuntimeCapabilities() Capabilities {
	c := Capabilities{Arch: runtime.GOARCH}
	switch runtime.GOARCH {
	case "amd64":
		c.HasAVX2 = cpu.X86.HasAVX2
		c.HasFMA = cpu.X86.HasFMA
		c.HasDecode = hasDecodeAsm && c.HasAVX2 && c.HasFMA
		c.HasDequant = hasDequantAsm && c.HasAVX2 && c.HasFMA
		c.HasGemv = hasGemvAsm && c.HasAVX2 && c.HasFMA
	case "arm64":
		c.HasNEON = cpu.ARM64.HasASIMD
		c.HasDecode = hasDecodeAsm && c.HasNEON
		c.HasDequant = hasDequantAsm && c.HasNEON
		c.HasGemv = hasGemvAsm && c.HasNEON
	case "riscv64":
		c.HasRVV = cpu.RISCV64.HasV
		c.HasDequant = c.HasRVV
		c.HasGemv = c.HasRVV
	}
	return c
}
