//go:build !riscv64

package simd

import "math"

func fastExpF32(x float32) float32 { return float32(math.Exp(float64(x))) }

// FastExpF32 is the exported form for use by model packages.
func FastExpF32(x float32) float32 { return fastExpF32(x) }
