//go:build !amd64

package fp8

const hasDotE4M3AVX2 = false

func dotE4M3(x []float32, w []byte) float32 { return dotE4M3Scalar(x, w) }
