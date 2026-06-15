//go:build amd64

package simd

const HasDotU4F32SIMD = true

//go:noescape
func dotU4F32LowAndSumAsm(q []byte, x []float32) (dot float32, sum float32)

//go:noescape
func dotU4F32HighAndSumAsm(q []byte, x []float32) (dot float32, sum float32)

//go:noescape
func dotU4F32LowAndSumx4Asm(q []byte, x []float32, stride int) (dot0 float32, sum0 float32, dot1 float32, sum1 float32, dot2 float32, sum2 float32, dot3 float32, sum3 float32)

//go:noescape
func dotU4F32HighAndSumx4Asm(q []byte, x []float32, stride int) (dot0 float32, sum0 float32, dot1 float32, sum1 float32, dot2 float32, sum2 float32, dot3 float32, sum3 float32)

func dotU4F32LowAndSum(q []byte, x []float32) (float32, float32) {
	if len(q)%8 != 0 {
		return dotU4F32LowAndSumScalar(q, x)
	}
	return dotU4F32LowAndSumAsm(q, x)
}

func dotU4F32HighAndSum(q []byte, x []float32) (float32, float32) {
	if len(q)%8 != 0 {
		return dotU4F32HighAndSumScalar(q, x)
	}
	return dotU4F32HighAndSumAsm(q, x)
}

func dotU4F32LowAndSumx4(q []byte, x []float32, stride int) (float32, float32, float32, float32, float32, float32, float32, float32) {
	if len(q)%8 != 0 {
		return dotU4F32LowAndSumx4Scalar(q, x, stride)
	}
	return dotU4F32LowAndSumx4Asm(q, x, stride)
}

func dotU4F32HighAndSumx4(q []byte, x []float32, stride int) (float32, float32, float32, float32, float32, float32, float32, float32) {
	if len(q)%8 != 0 {
		return dotU4F32HighAndSumx4Scalar(q, x, stride)
	}
	return dotU4F32HighAndSumx4Asm(q, x, stride)
}
