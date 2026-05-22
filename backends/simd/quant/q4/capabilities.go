package q4

import (
	"runtime"

	"golang.org/x/sys/cpu"
)

// Capabilities describes Q4/GPTQ CPU execution features. It is conservative:
// scalar remains the only active path until architecture kernels land and pass
// parity tests.
type Capabilities struct {
	Arch       string
	HasAVX2    bool
	HasFMA     bool
	HasNEON    bool
	HasRVV     bool
	HasGemvSym bool
	HasDequant bool
}

const (
	hasGemvSymAsm = false
	hasDequantAsm = false
)

// RuntimeCapabilities returns the active Q4/GPTQ capability set.
func RuntimeCapabilities() Capabilities {
	c := Capabilities{Arch: runtime.GOARCH}
	switch runtime.GOARCH {
	case "amd64":
		c.HasAVX2 = cpu.X86.HasAVX2
		c.HasFMA = cpu.X86.HasFMA
		c.HasGemvSym = hasGemvSymAsm && c.HasAVX2 && c.HasFMA
		c.HasDequant = hasDequantAsm && c.HasAVX2 && c.HasFMA
	case "arm64":
		c.HasNEON = cpu.ARM64.HasASIMD
		c.HasGemvSym = hasGemvSymAsm && c.HasNEON
		c.HasDequant = hasDequantAsm && c.HasNEON
	case "riscv64":
		c.HasRVV = cpu.RISCV64.HasV
		c.HasGemvSym = c.HasRVV
	}
	return c
}
