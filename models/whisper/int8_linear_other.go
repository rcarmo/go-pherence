//go:build !riscv64

package whisper

import "time"

func nowNs() int64 { return time.Now().UnixNano() }

var useInt8 = false
var attnInt8 = false

var (
	i8QuantNs int64
	i8PackNs  int64
	i8GemmNs  int64
	i8DeqNs   int64
)

func int8Eligible(inDim, outDim int) bool { return false }

func linearForwardInt8(x, weight, bias []float32, seqLen, inDim, outDim int) []float32 {
	return linearForwardScalar(x, weight, bias, seqLen, inDim, outDim)
}
