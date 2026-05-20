package runtime

import (
	"runtime"

	"golang.org/x/sys/cpu"
)

// Capabilities describes MLX CPU execution features. It is conservative:
// scalar remains the only active path until architecture kernels land and pass
// parity tests.
type Capabilities struct {
	Arch       string
	HasAVX2    bool
	HasFMA     bool
	HasNEON    bool
	HasGemv4   bool
	HasDequant bool
}

const (
	hasGemv4Asm   = false
	hasDequantAsm = false
)

// RuntimeCapabilities returns the active MLX CPU capability set.
func RuntimeCapabilities() Capabilities {
	c := Capabilities{Arch: runtime.GOARCH}
	switch runtime.GOARCH {
	case "amd64":
		c.HasAVX2 = cpu.X86.HasAVX2
		c.HasFMA = cpu.X86.HasFMA
		c.HasGemv4 = hasGemv4Asm && c.HasAVX2 && c.HasFMA
		c.HasDequant = hasDequantAsm && c.HasAVX2 && c.HasFMA
	case "arm64":
		c.HasNEON = cpu.ARM64.HasASIMD
		c.HasGemv4 = hasGemv4Asm && c.HasNEON
		c.HasDequant = hasDequantAsm && c.HasNEON
	}
	return c
}
