//go:build !amd64

package simd

const HasDotU4F32SIMD = false

func dotU4F32LowAndSum(q []byte, x []float32) (float32, float32) {
	return dotU4F32LowAndSumScalar(q, x)
}

func dotU4F32HighAndSum(q []byte, x []float32) (float32, float32) {
	return dotU4F32HighAndSumScalar(q, x)
}
