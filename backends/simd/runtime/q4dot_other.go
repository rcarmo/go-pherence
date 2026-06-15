//go:build !amd64

package simd

const HasDotU4F32SIMD = false

func dotU4F32LowAndSum(q []byte, x []float32) (float32, float32) {
	return dotU4F32LowAndSumScalar(q, x)
}

func dotU4F32HighAndSum(q []byte, x []float32) (float32, float32) {
	return dotU4F32HighAndSumScalar(q, x)
}

func dotU4F32LowAndSumx4(q []byte, x []float32, stride int) (float32, float32, float32, float32, float32, float32, float32, float32) {
	return dotU4F32LowAndSumx4Scalar(q, x, stride)
}

func dotU4F32HighAndSumx4(q []byte, x []float32, stride int) (float32, float32, float32, float32, float32, float32, float32, float32) {
	return dotU4F32HighAndSumx4Scalar(q, x, stride)
}
