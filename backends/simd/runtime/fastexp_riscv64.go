//go:build riscv64

package simd

import "github.com/rcarmo/go-pherence/backends/spacemit/rvv"

// fastExpF32 uses the Schraudolph integer-trick approximation on riscv64,
// which is ~9x faster than math.Exp. Accuracy is ~6% max relative error,
// sufficient for softmax and activation numerics.
func fastExpF32(x float32) float32 { return rvv.FastExp(x) }
