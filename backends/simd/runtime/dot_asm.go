//go:build amd64 || arm64 || riscv64

package simd

//go:noescape
func sdotAsm(x, y []float32) float32

//go:noescape
func saxpyAsm(alpha float32, x []float32, y []float32)
