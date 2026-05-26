package main

import (
	"math"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
)

// parallelDecodeAI runs inference with matmuls on AI cores (8-15, VLEN=1024).
// Scalar ops (norm, attention, SiLU) run on the calling goroutine (core 0-7).
// Matmuls are dispatched to the AI worker pool.
func parallelDecodeAI(
	x []float32,
	layers []layerWeights,
	nLayers, nEmbd, nHeads, nKVHeads, headDim, nFF int,
	rmsEps, ropeBase float32,
	kCache, vCache [][]float32,
	nPast int,
	pool *AIWorkerPool,
) {
	nQEmbd := nHeads * headDim
	nKVD := nKVHeads * headDim

	xn := make([]float32, nEmbd)
	xn2 := make([]float32, nEmbd)
	scoresPool := make([]float32, 512)

	for il := 0; il < nLayers; il++ {
		l := &layers[il]
		pos := nPast

		// RMS Norm (scalar, main core)
		var ss float32
		for i := 0; i < nEmbd; i++ { ss += x[i] * x[i] }
		invRMS := float32(1.0 / math.Sqrt(float64(ss/float32(nEmbd)+rmsEps)))
		for i := 0; i < nEmbd; i++ { xn[i] = x[i] * invRMS * l.attnNorm[i] }

		// Quantize activation + pack for VLEN=1024
		actI8 := make([]int8, nEmbd)
		actScale := quantizeToI8(xn, actI8)
		KpEmbd := ((nEmbd + 31) / 32) * 32
		actI8Pad := make([]int8, KpEmbd)
		copy(actI8Pad, actI8)
		actPacked := ime2.BroadcastPack1024(actI8Pad, KpEmbd)

		// QKV matmuls on AI cores
		qF := make([]float32, nQEmbd)
		kF := make([]float32, nKVD)
		vF := make([]float32, nKVD)
		GemmAIPooled(nQEmbd, KpEmbd, l.wqPacked1024, actPacked, l.wqScale, actScale, qF, pool)
		GemmAIPooled(nKVD, KpEmbd, l.wkPacked1024, actPacked, l.wkScale, actScale, kF, pool)
		GemmAIPooled(nKVD, KpEmbd, l.wvPacked1024, actPacked, l.wvScale, actScale, vF, pool)

		// KV cache + attention (scalar, main core)
		if l.kNorm != nil {
			for kh := 0; kh < nKVHeads; kh++ {
				head := kF[kh*headDim : (kh+1)*headDim]
				var s2 float32
				for d := range head { s2 += head[d] * head[d] }
				inv := float32(1.0 / math.Sqrt(float64(s2/float32(headDim)+rmsEps)))
				for d := range head { head[d] = head[d] * inv * l.kNorm[d] }
			}
		}
		copy(kCache[il][pos*nKVD:pos*nKVD+nKVD], kF)
		copy(vCache[il][pos*nKVD:pos*nKVD+nKVD], vF)
		for kh := 0; kh < nKVHeads; kh++ {
			applyRoPE(kCache[il][pos*nKVD+kh*headDim:pos*nKVD+(kh+1)*headDim], headDim, pos, ropeBase)
		}

		repFactor := nHeads / nKVHeads
		invSqrtD := float32(1.0 / math.Sqrt(float64(headDim)))
		for h := 0; h < nHeads; h++ {
			qHead := qF[h*headDim : (h+1)*headDim]
			if l.qNorm != nil {
				var s3 float32
				for d := range qHead { s3 += qHead[d] * qHead[d] }
				inv := float32(1.0 / math.Sqrt(float64(s3/float32(headDim)+rmsEps)))
				for d := range qHead { qHead[d] = qHead[d] * inv * l.qNorm[d] }
			}
			applyRoPE(qHead, headDim, pos, ropeBase)
			kvH := h / repFactor
			scores := scoresPool[:pos+1]
			var maxScore float32 = -1e30
			for t := 0; t <= pos; t++ {
				var dot float32
				for d := 0; d < headDim; d++ { dot += qHead[d] * kCache[il][t*nKVD+kvH*headDim+d] }
				scores[t] = dot * invSqrtD
				if scores[t] > maxScore { maxScore = scores[t] }
			}
			var sumExp float32
			for i := range scores { scores[i] = float32(math.Exp(float64(scores[i] - maxScore))); sumExp += scores[i] }
			for i := range scores { scores[i] /= sumExp }
			for d := 0; d < headDim; d++ {
				var sum float32
				for t := 0; t <= pos; t++ { sum += scores[t] * vCache[il][t*nKVD+kvH*headDim+d] }
				qF[h*headDim+d] = sum
			}
		}

		// WO projection on AI cores
		woActI8 := make([]int8, nQEmbd)
		woActScale := quantizeToI8(qF[:nQEmbd], woActI8)
		KpQ := ((nQEmbd + 31) / 32) * 32
		woActPad := make([]int8, KpQ)
		copy(woActPad, woActI8)
		woActPacked := ime2.BroadcastPack1024(woActPad, KpQ)
		woOut := make([]float32, nEmbd)
		GemmAIPooled(nEmbd, KpQ, l.woPacked1024, woActPacked, l.woScale, woActScale, woOut, pool)
		for i := 0; i < nEmbd; i++ { x[i] += woOut[i] }

		// FFN norm + quantize
		ss = 0
		for i := 0; i < nEmbd; i++ { ss += x[i] * x[i] }
		invRMS = float32(1.0 / math.Sqrt(float64(ss/float32(nEmbd)+rmsEps)))
		for i := 0; i < nEmbd; i++ { xn2[i] = x[i] * invRMS * l.ffnNorm[i] }
		ffnActI8 := make([]int8, nEmbd)
		ffnActScale := quantizeToI8(xn2, ffnActI8)
		ffnActPad := make([]int8, KpEmbd)
		copy(ffnActPad, ffnActI8)
		ffnActPacked := ime2.BroadcastPack1024(ffnActPad, KpEmbd)

		// Gate + Up on AI cores
		gateF := make([]float32, nFF)
		upF := make([]float32, nFF)
		GemmAIPooled(nFF, KpEmbd, l.gatePacked1024, ffnActPacked, l.gateScale, ffnActScale, gateF, pool)
		GemmAIPooled(nFF, KpEmbd, l.upPacked1024, ffnActPacked, l.upScale, ffnActScale, upF, pool)

		// SiLU + element-wise multiply (scalar)
		hidden := make([]float32, nFF)
		for i := 0; i < nFF; i++ { hidden[i] = silu(gateF[i]) * upF[i] }

		// Down projection on AI cores
		downActI8 := make([]int8, nFF)
		downActScale := quantizeToI8(hidden, downActI8)
		KpFF := ((nFF + 31) / 32) * 32
		downActPad := make([]int8, KpFF)
		copy(downActPad, downActI8)
		downActPacked := ime2.BroadcastPack1024(downActPad, KpFF)
		downF := make([]float32, nEmbd)
		GemmAIPooled(nEmbd, KpFF, l.downPacked1024, downActPacked, l.downScale, downActScale, downF, pool)
		for i := 0; i < nEmbd; i++ { x[i] += downF[i] }
	}
}

// quantizeToI8 quantizes float32 to int8 with single global scale.
// Returns inverse scale (maxAbs/127).
func quantizeToI8(src []float32, dst []int8) float32 {
	var maxAbs float32
	for _, v := range src { a := v; if a < 0 { a = -a }; if a > maxAbs { maxAbs = a } }
	if maxAbs == 0 { return 0 }
	s := float32(127.0) / maxAbs
	for i, v := range src {
		q := v * s; if q > 127 { q = 127 } else if q < -128 { q = -128 }
		dst[i] = int8(q)
	}
	return maxAbs / 127.0
}

var _ = unsafe.Pointer(nil)
