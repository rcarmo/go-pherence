package bench

import (
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/backends/mlx"
	simdnvfp4 "github.com/rcarmo/go-pherence/backends/simd/quant/nvfp4"
	simdq4 "github.com/rcarmo/go-pherence/backends/simd/quant/q4"
	"github.com/rcarmo/go-pherence/backends/simd/runtime"
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
	qw := benchMLXWeight(inDim, outDim, 64)
	x := benchSeq(inDim)
	out := make([]float32, outDim)
	b.ReportAllocs()
	b.SetBytes(int64(inDim * outDim * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mlx.Gemv(out, x, qw)
	}
}

func BenchmarkCPUHotGemmMLXBatch8_1536x2048(b *testing.B) {
	inDim := 1536
	outDim := 2048
	batch := 8
	qw := benchMLXWeight(inDim, outDim, 64)
	x := benchSeq(batch * inDim)
	out := make([]float32, batch*outDim)
	b.ReportAllocs()
	b.SetBytes(int64(batch * inDim * outDim * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !mlx.Gemm(out, x, batch, qw) {
			b.Fatal("Gemm returned false")
		}
	}
}

func BenchmarkCPUHotDequantMLX1536x2048(b *testing.B) {
	inDim := 1536
	outDim := 2048
	qw := benchMLXWeight(inDim, outDim, 64)
	out := make([]float32, inDim*outDim)
	b.ReportAllocs()
	b.SetBytes(int64(inDim * outDim * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !mlx.DequantTo(out, qw) {
			b.Fatal("DequantTo returned false")
		}
	}
}

func BenchmarkCPUHotMoEMLXExperts512x1024Top2(b *testing.B) {
	hidden := 512
	inter := 1024
	numExperts := 8
	active := 2
	layer := &LlamaLayer{
		RouterW:     benchMLXWeight(hidden, numExperts, 64),
		ExpertGateW: make([]*mlx.QuantWeight, numExperts),
		ExpertUpW:   make([]*mlx.QuantWeight, numExperts),
		ExpertDownW: make([]*mlx.QuantWeight, numExperts),
	}
	for i := 0; i < numExperts; i++ {
		layer.ExpertGateW[i] = benchMLXWeight(hidden, inter, 64)
		layer.ExpertUpW[i] = benchMLXWeight(hidden, inter, 64)
		layer.ExpertDownW[i] = benchMLXWeight(inter, hidden, 64)
	}
	cfg := LlamaConfig{NumExperts: numExperts, NumExpertsPerTok: active, MoEIntermediate: inter, NormTopKProb: true}
	x := benchSeq(hidden)
	b.ReportAllocs()
	b.SetBytes(int64(active * (hidden*inter*2 + inter*hidden) * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := moeForward(x, layer, cfg)
		if len(out) != hidden {
			b.Fatalf("moe len=%d", len(out))
		}
	}
}

func benchMLXWeight(inDim, outDim, groupSize int) *mlx.QuantWeight {
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
	return qw
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

func BenchmarkCPUHotGemvQ4Asym1536x2048(b *testing.B) {
	inDim := 1536
	outDim := 2048
	groups := inDim / 128
	qweight := make([]int32, (inDim/8)*outDim)
	qzeros := make([]int32, groups*(outDim/8))
	gIdx := make([]int32, inDim)
	scales := make([]float32, groups*outDim)
	for i := range qweight {
		qweight[i] = int32(0x76543210)
	}
	for i := range qzeros {
		qzeros[i] = int32(0x77777777)
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
		simdq4.Gemv(out, x, qweight, qzeros, gIdx, scales, inDim, outDim, false)
	}
}

func BenchmarkCPUHotMoEQ4Experts512x1024Top2(b *testing.B) {
	inDim := 512
	outDim := 1024
	topK := 2
	experts := 4
	groups := inDim / 128
	x := benchSeq(inDim)
	out := make([]float32, outDim)
	tmp := make([]float32, outDim)
	expertQW := make([][]int32, experts)
	expertGIdx := make([][]int32, experts)
	expertScales := make([][]float32, experts)
	for e := 0; e < experts; e++ {
		qweight := make([]int32, (inDim/8)*outDim)
		gIdx := make([]int32, inDim)
		scales := make([]float32, groups*outDim)
		for i := range qweight {
			qweight[i] = int32(0x76543210 + e)
		}
		for i := range gIdx {
			gIdx[i] = int32(i / 128)
		}
		for i := range scales {
			scales[i] = 0.005 * float32((i+e)%7+1)
		}
		expertQW[e] = qweight
		expertGIdx[e] = gIdx
		expertScales[e] = scales
	}
	weights := []float32{0.6, 0.4}
	b.ReportAllocs()
	b.SetBytes(int64(topK * inDim * outDim * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := range out {
			out[j] = 0
		}
		for k := 0; k < topK; k++ {
			expert := (i + k) % experts
			if !simdq4.GemvSymTo(tmp, x, expertQW[expert], expertGIdx[expert], expertScales[expert], inDim, outDim) {
				b.Fatal("GemvSymTo returned false")
			}
			weight := weights[k]
			for j := range out {
				out[j] += weight * tmp[j]
			}
		}
	}
}

func BenchmarkCPUHotDequantQ4Asym1536x2048(b *testing.B) {
	inDim := 1536
	outDim := 2048
	groups := inDim / 128
	qweight := make([]int32, (inDim/8)*outDim)
	qzeros := make([]int32, groups*(outDim/8))
	gIdx := make([]int32, inDim)
	scales := make([]float32, groups*outDim)
	for i := range qweight {
		qweight[i] = int32(0x76543210)
	}
	for i := range qzeros {
		qzeros[i] = int32(0x77777777)
	}
	for i := range gIdx {
		gIdx[i] = int32(i / 128)
	}
	for i := range scales {
		scales[i] = 0.01
	}
	out := make([]float32, inDim*outDim)
	b.ReportAllocs()
	b.SetBytes(int64(inDim * outDim * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !simdq4.DequantTo(out, qweight, qzeros, gIdx, scales, inDim, outDim, false) {
			b.Fatal("DequantTo returned false")
		}
	}
}

func BenchmarkCPUHotDequantQ4Sym1536x2048(b *testing.B) {
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
	out := make([]float32, inDim*outDim)
	b.ReportAllocs()
	b.SetBytes(int64(inDim * outDim * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !simdq4.DequantSymTo(out, qweight, gIdx, scales, inDim, outDim) {
			b.Fatal("DequantSymTo returned false")
		}
	}
}

func BenchmarkCPUHotDequantNVFP4_1536x2048(b *testing.B) {
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
	out := make([]float32, inDim*outDim)
	b.ReportAllocs()
	b.SetBytes(int64(inDim * outDim * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !simdnvfp4.DequantNVFP4To(out, qw) {
			b.Fatal("DequantNVFP4To returned false")
		}
	}
}

func BenchmarkCPUHotGemmNVFP4Batch4_1536x2048(b *testing.B) {
	inDim := 1536
	outDim := 2048
	batch := 4
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
	x := make([]float32, batch*inDim)
	for row := 0; row < batch; row++ {
		copy(x[row*inDim:(row+1)*inDim], benchSeq(inDim))
	}
	out := make([]float32, batch*outDim)
	b.ReportAllocs()
	b.SetBytes(int64(batch * inDim * outDim * 4))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !simdnvfp4.GemmNVFP4(out, x, batch, qw) {
			b.Fatal("GemmNVFP4 returned false")
		}
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
