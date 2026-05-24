// Package inference implements a pure Go transformer decode loop
// using IME2 vmadot for matrix multiplications.
package inference

import (
	"math"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
)

// Model holds all weights and state for inference.
type Model struct {
	NVocab, NEmbd, NHeads, NHeadsKV int
	NLayers, NFF, NCtx              int
	NHeadDim                        int
	RopeBase, RmsEps                float32

	// Weights (INT8 pre-packed for vmadot)
	TokEmbd []int8 // [NVocab × NEmbd] — lookup table (kept as raw Q4_K for get_rows)
	Output  []int8 // [NVocab × NEmbd] — tied to TokEmbd or separate

	// Per-layer
	Layers []Layer

	// KV cache
	KCache [][]float32 // [NLayers][NCtx * NHeadDim * NHeadsKV]
	VCache [][]float32

	// Runtime state
	NPast int
}

// Layer holds weights for one transformer layer.
type Layer struct {
	AttnNorm []float32 // [NEmbd]
	FFNNorm  []float32 // [NEmbd]

	// Attention weights (pre-packed for vmadot)
	WQ []int8 // packed [NEmbd → NHeads*NHeadDim]
	WK []int8 // packed [NEmbd → NHeadsKV*NHeadDim]
	WV []int8 // packed [NEmbd → NHeadsKV*NHeadDim]
	WO []int8 // packed [NHeads*NHeadDim → NEmbd]

	// FFN weights (pre-packed)
	FFNGate []int8 // packed [NEmbd → NFF]
	FFNUp   []int8 // packed [NEmbd → NFF]
	FFNDown []int8 // packed [NFF → NEmbd]

	// Quantization scales per weight (for dequant after vmadot)
	WQScale, WKScale, WVScale, WOScale       float32
	FFNGateScale, FFNUpScale, FFNDownScale    float32
}

// QuantizeF32ToINT8 quantizes a float32 slice to int8, returning the scale.
func QuantizeF32ToINT8(src []float32, dst []int8) float32 {
	var maxAbs float32
	for _, v := range src {
		if a := float32(math.Abs(float64(v))); a > maxAbs {
			maxAbs = a
		}
	}
	if maxAbs == 0 {
		for i := range dst {
			dst[i] = 0
		}
		return 0
	}
	scale := 127.0 / maxAbs
	for i, v := range src {
		q := v * scale
		if q > 127 {
			q = 127
		} else if q < -128 {
			q = -128
		}
		dst[i] = int8(q)
	}
	return maxAbs / 127.0 // inverse scale for dequant
}

// RMSNorm computes RMS normalization: out[i] = x[i] / rms(x) * weight[i]
func RMSNorm(x, weight, out []float32, eps float32) {
	n := len(x)
	var ss float32
	for i := 0; i < n; i++ {
		ss += x[i] * x[i]
	}
	ss = 1.0 / float32(math.Sqrt(float64(ss/float32(n)+eps)))
	for i := 0; i < n; i++ {
		out[i] = x[i] * ss * weight[i]
	}
}

// MatVecQ4K performs matrix-vector multiply: out[M] = W[M×K] · x[K]
// where W is stored as pre-packed INT8 tiles and x is quantized on the fly.
// wScale is the weight quantization scale.
func MatVecQ4K(M, K int, wPacked []int8, x []float32, out []float32, wScale float32) {
	// Quantize activation to INT8
	xI8 := make([]int8, K)
	xScale := QuantizeF32ToINT8(x, xI8)

	// Pack x into tile format (replicate for 4-row broadcast)
	// Replicate x into 4 rows for broadcast
	xBroadcast := make([]int8, 4*K)
	for r := 0; r < 4; r++ {
		copy(xBroadcast[r*K:(r+1)*K], xI8)
	}
	xPacked := ime2.PackTiles(xBroadcast, 4, K) // 4 copies of x for vmadot broadcast

	// Accumulator
	cI32 := make([]int32, M*4) // M output rows × 4 (from 4 broadcast copies)

	// Run GEMM: W[M×K] * x_broadcast[4×K]^T → C[M×4]
	// But we only need the diagonal (each row of W dot x = one output)
	// Actually: GemmINT8Packed does C[M×N] = A[M×K] * B[N×K]^T
	// With B = xPacked (4 copies), N=4, we get C[M×4] where each row
	// has the same dot product repeated 4 times.
	ime2.GemmINT8Packed(M, 4, K, wPacked, xPacked, cI32)

	// Dequantize: out[i] = C[i][0] * wScale * xScale
	combinedScale := wScale * xScale
	for i := 0; i < M; i++ {
		out[i] = float32(cI32[i*4]) * combinedScale
	}
}

// Decode performs one token decode step. Returns logits.
func (m *Model) Decode(tokenID int) []float32 {
	// TODO: implement full decode loop
	// For now, just return zeros
	return make([]float32, m.NVocab)
}

// ensure unsafe import is used
var _ = unsafe.Pointer(nil)

// MatVecQ4KParallel performs matrix-vector multiply using multi-threaded GEMM.
// Same as MatVecQ4K but uses 8 threads for the vmadot inner loop.
func MatVecQ4KParallel(M, K int, wPacked []int8, x []float32, out []float32, wScale float32, nThreads int) {
	// Quantize activation to INT8
	xI8 := make([]int8, K)
	xScale := QuantizeF32ToINT8(x, xI8)

	// Replicate x into 4 rows for broadcast
	xBroadcast := make([]int8, 4*K)
	for r := 0; r < 4; r++ {
		copy(xBroadcast[r*K:(r+1)*K], xI8)
	}
	xPacked := ime2.PackTiles(xBroadcast, 4, K)

	// Run parallel GEMM
	cI32 := make([]int32, M*4)
	ime2.GemmINT8PackedParallel(M, 4, K, wPacked, xPacked, cI32, nThreads)

	// Dequantize
	combinedScale := wScale * xScale
	for i := 0; i < M; i++ {
		out[i] = float32(cI32[i*4]) * combinedScale
	}
}

// MatVecINT8Parallel performs out[M] = wPacked[M×K] · actPacked[4×K]^T
// where actPacked is ALREADY quantized and packed (call PackActivation first).
// This avoids per-call quantization overhead.
func MatVecINT8Parallel(M, K int, wPacked []int8, actPacked []int8, out []int32, nThreads int) {
	ime2.GemmINT8PackedParallel(M, 4, K, wPacked, actPacked, out, nThreads)
}

// PackActivation quantizes F32 activation to INT8 and packs into 4-row tile format.
// Returns the packed data and the dequant scale.
func PackActivation(x []float32, K int) ([]int8, float32) {
	xI8 := make([]int8, K)
	scale := QuantizeF32ToINT8(x, xI8)
	// Replicate into 4 rows
	xBroadcast := make([]int8, 4*K)
	for r := 0; r < 4; r++ {
		copy(xBroadcast[r*K:(r+1)*K], xI8)
	}
	packed := ime2.PackTiles(xBroadcast, 4, K)
	return packed, scale
}

// MatVecINT8Pool performs matvec using a persistent worker pool (no goroutine spawn per call).
func MatVecINT8Pool(M, K int, wPacked []int8, actPacked []int8, out []int32, pool *ime2.WorkerPool) {
	ime2.GemmINT8PackedPool(M, 4, K, wPacked, actPacked, out, pool)
}

// PackActivationInto quantizes and packs activation into pre-allocated buffers.
// xI8Buf must be at least K bytes. broadcastBuf must be at least 4*K bytes.
// Returns the packed slice (from broadcastBuf) and scale.
func PackActivationInto(x []float32, K int, xI8Buf, broadcastBuf []int8) ([]int8, float32) {
	xI8 := xI8Buf[:K]
	scale := QuantizeF32ToINT8(x[:K], xI8)
	broadcast := broadcastBuf[:4*K]
	for r := 0; r < 4; r++ {
		copy(broadcast[r*K:(r+1)*K], xI8)
	}
	return ime2.PackTilesInto(broadcast, 4, K, broadcastBuf[4*K:]), scale
}
