//go:build amd64

package fp8

import "golang.org/x/sys/cpu"

var hasDotE4M3AVX2 = cpu.X86.HasAVX2 && cpu.X86.HasFMA

//go:noescape
func dotE4M3AVX2(x []float32, w []byte, lut *[256]float32) float32

func dotE4M3(x []float32, w []byte) float32 {
	if len(x) == len(w) && len(x) > 0 && hasDotE4M3AVX2 {
		return dotE4M3AVX2(x, w, &e4m3LUT)
	}
	return dotE4M3Scalar(x, w)
}
