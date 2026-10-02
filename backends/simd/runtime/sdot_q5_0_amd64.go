package simd

import "golang.org/x/sys/cpu"

// HasSdotQ5_0Asm admits the fused AVX2/FMA/F16C Q5_0 dot kernel.
var HasSdotQ5_0Asm = cpu.X86.HasAVX2 && cpu.X86.HasFMA && hasF16CConvert && HasDotAsm

//go:noescape
func sdotQ5_0Asm(x []float32, raw []byte, blocks int) float32
