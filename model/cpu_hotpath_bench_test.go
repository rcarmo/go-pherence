package model

import (
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/backends/mlx"
	"github.com/rcarmo/go-pherence/backends/simd/runtime"
	simdnvfp4 "github.com/rcarmo/go-pherence/backends/simd/runtime/nvfp4"
	simdq4 "github.com/rcarmo/go-pherence/backends/simd/runtime/q4"
)

func benchSeq(n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(math.Sin(float64(i) * 0.013))
	}
	return x
}

func BenchmarkCPUHotRMSNorm3584(b *testing.B) {
	x := benchSeq(3584)
	w := make([]float32, len(x))
	for i := range w {
		w[i] = 1
	}
	b.ReportAllocs()
	b.SetBytes(int64(len(x) * 4 * 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		simd.RMSNorm(x, w, 1e-6)
	}
}

func BenchmarkCPUHotGELUTanhMul8192(b *testing.B) {
	a := benchSeq(8192)
	bb := benchSeq(8192)
	dst := make([]float32, len(a))
	b.ReportAllocs()
	b.SetBytes(int64(len(a) * 4 * 3))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		simd.GELUTanhMul(dst, a, bb)
	}
}

func BenchmarkCPUHotSiLUMul8192(b *testing.B) {
	a := benchSeq(8192)
	bb := benchSeq(8192)
	dst := make([]float32, len(a))
	b.ReportAllocs()
	b.SetBytes(int64(len(a) * 4 * 3))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		simd.VecSiLUMul(dst, a, bb)
	}
}

func BenchmarkCPUHotVecScale3584(b *testing.B) {
	a := benchSeq(3584)
	dst := make([]float32, len(a))
	b.ReportAllocs()
	b.SetBytes(int64(len(a) * 4 * 2))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		simd.VecScale(dst, a, 0.70710677)
	}
}

func BenchmarkCPUHotRoPEPartialGemma4SWA(b *testing.B) {
	benchmarkCPUHotRoPEPartial(b, 8, 256, 128, 2048, 10000)
}

func BenchmarkCPUHotRoPEPartialGemma4Full(b *testing.B) {
	benchmarkCPUHotRoPEPartial(b, 8, 512, 64, 2048, 1000000)
}

func BenchmarkCPUHotRoPEQwenFull(b *testing.B) {
	numHeads := 32
	headDim := 128
	rotHalf := headDim / 2
	maxSeq := 2048
	x := benchSeq(numHeads * headDim)
	freqs := ropeBenchFreqs(maxSeq, rotHalf, headDim, 1000000)
	b.ReportAllocs()
	b.SetBytes(int64(numHeads * rotHalf * 2 * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		applyRoPE(x, freqs, i&(maxSeq-1), numHeads, headDim)
	}
}

func benchmarkCPUHotRoPEPartial(b *testing.B, numHeads, headDim, rotHalf, maxSeq int, theta float64) {
	x := benchSeq(numHeads * headDim)
	freqs := ropeBenchFreqs(maxSeq, rotHalf, headDim, theta)
	b.ReportAllocs()
	b.SetBytes(int64(numHeads * rotHalf * 2 * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		applyRoPEPartial(x, freqs, i&(maxSeq-1), numHeads, headDim, rotHalf)
	}
}

func ropeBenchFreqs(maxSeq, rotHalf, headDim int, theta float64) []float32 {
	freqs := make([]float32, maxSeq*rotHalf*2)
	for pos := 0; pos < maxSeq; pos++ {
		for i := 0; i < rotHalf; i++ {
			angle := float64(pos) / math.Pow(theta, float64(2*i)/float64(headDim))
			freqs[(pos*rotHalf+i)*2] = float32(math.Cos(angle))
			freqs[(pos*rotHalf+i)*2+1] = float32(math.Sin(angle))
		}
	}
	return freqs
}

func BenchmarkCPUHotGQAAttentionDecode512(b *testing.B) {
	numHeads := 12
	numKVHeads := 4
	headDim := 128
	seqLen := 512
	q := benchSeq(numHeads * headDim)
	k := benchSeq(seqLen * numKVHeads * headDim)
	v := benchSeq(seqLen * numKVHeads * headDim)
	out := make([]float32, numHeads*headDim)
	scores := make([]float32, seqLen)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	b.ReportAllocs()
	b.SetBytes(int64((len(q) + len(k) + len(v)) * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gqaAttentionScaleInto(out, scores, q, k, v, seqLen, numHeads, numKVHeads, headDim, scale)
	}
}

func BenchmarkCPUHotGemvMLQ1536x2048(b *testing.B) {
	inDim := 1536
	outDim := 2048
	groupSize := 64
	bits := 4
	packFactor := 32 / bits
	groups := inDim / groupSize
	qw := &mlx.QuantWeight{
		Weight:    make([]uint32, outDim*(inDim/packFactor)),
		Scales:    make([]float32, outDim*groups),
		Biases:    make([]float32, outDim*groups),
		InDim:     inDim,
		OutDim:    outDim,
		Groups:    groups,
		GroupSize: groupSize,
		Bits:      bits,
	}
	for i := range qw.Weight {
		qw.Weight[i] = 0x76543210
	}
	for i := range qw.Scales {
		qw.Scales[i] = 0.01
		qw.Biases[i] = -0.04
	}
	x := benchSeq(inDim)
	out := make([]float32, outDim)
	b.ReportAllocs()
	b.SetBytes(int64(inDim * outDim * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mlx.Gemv(out, x, qw)
	}
}

func BenchmarkCPUHotGemvQ4Sym1536x2048(b *testing.B) {
	inDim := 1536
	outDim := 2048
	groups := inDim / 128
	qweight := make([]int32, (inDim/8)*outDim)
	gIdx := make([]int32, inDim)
	scales := make([]float32, groups*outDim)
	for i := range qweight {
		qweight[i] = int32(0x76543210)
	}
	for i := range gIdx {
		gIdx[i] = int32(i / 128)
	}
	for i := range scales {
		scales[i] = 0.01
	}
	x := benchSeq(inDim)
	out := make([]float32, outDim)
	b.ReportAllocs()
	b.SetBytes(int64(inDim * outDim * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		simdq4.GemvSym(out, x, qweight, gIdx, scales, inDim, outDim)
	}
}

func BenchmarkCPUHotGemvNVFP4_1536x2048(b *testing.B) {
	inDim := 1536
	outDim := 2048
	groups := inDim / 16
	qw := &simdnvfp4.NVFP4Weight{
		Weight:       make([]byte, outDim*(inDim/2)),
		WeightScale:  make([]byte, outDim*groups),
		WeightScale2: 0.5,
		OutDim:       outDim,
		InDim:        inDim,
		Groups:       groups,
		GroupSize:    16,
	}
	for i := range qw.Weight {
		qw.Weight[i] = 0x76
	}
	for i := range qw.WeightScale {
		qw.WeightScale[i] = 0x38 // E4M3 1.0
	}
	x := benchSeq(inDim)
	out := make([]float32, outDim)
	b.ReportAllocs()
	b.SetBytes(int64(inDim * outDim * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		simdnvfp4.GemvNVFP4(out, x, qw)
	}
}
