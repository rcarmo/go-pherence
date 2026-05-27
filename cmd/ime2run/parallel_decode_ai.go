package main

import (
	"fmt"
	"math"
	"os"
	"time"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
)

// parallelDecodeAI runs inference with matmuls on AI cores (8-15, VLEN=1024).
// Scalar ops (norm, attention, SiLU) run on the calling goroutine (core 0-7).
// Matmuls are dispatched to the AI worker pool.
var aiProfileOn = os.Getenv("IME2_PROFILE") != ""
var aiUseVL32 = os.Getenv("IME2_AI_VL32") != ""
var scalarUseAI = os.Getenv("IME2_SCALAR_AI") != ""
var rvvQuantOn = os.Getenv("IME2_RVV_QUANT") != ""
var scalarAttnOn = os.Getenv("IME2_SCALAR_ATTN") != ""
var f32MatvecOn = os.Getenv("IME2_F32_MATVEC") != ""
var q4kBlockRefOn = os.Getenv("IME2_Q4K_BLOCK_REF") != ""
var q4kAIOn = os.Getenv("IME2_Q4K_A100") != ""
var q4kLlamaZPOn = os.Getenv("IME2_Q4K_LLAMA_ZP") != ""
var q4kLlamaX32On = os.Getenv("IME2_Q4K_LLAMA_X32") != ""
var q4kLlamaX32RefOn = os.Getenv("IME2_Q4K_LLAMA_X32_REF") != ""
var q4kCShimOn = os.Getenv("IME2_Q4K_CSHIM") != ""
var q4kGoAsmOn = os.Getenv("IME2_Q4K_GOASM") != ""
var q4kGoAsmSerialOn = os.Getenv("IME2_Q4K_GOASM_SERIAL") != ""
var q4kCompareOn = os.Getenv("IME2_Q4K_COMPARE") != ""
var q4kExactX32On = os.Getenv("IME2_Q4K_EXACT_X32") != ""
var q4kExactHalfOn = os.Getenv("IME2_Q4K_EXACT_HALF") != ""
var q4kExactAttnOn = os.Getenv("IME2_Q4K_EXACT_ATTN") != ""
var q4kExactQOn = os.Getenv("IME2_Q4K_EXACT_Q") != ""
var q4kExactKOn = os.Getenv("IME2_Q4K_EXACT_K") != ""
var q4kExactVOn = os.Getenv("IME2_Q4K_EXACT_V") != ""
var q4kExactFFNOn = os.Getenv("IME2_Q4K_EXACT_FFN") != ""
var q4kExactWOOn = os.Getenv("IME2_Q4K_EXACT_WO") != ""
var q4kMinCorrOn = os.Getenv("IME2_Q4K_MIN_CORR") != "" // add Q4K min correction to fast native path
var q4kScaledLoopOn = os.Getenv("IME2_Q4K_SCALED_LOOP") != "" // use vmadotQ4KScaledLoop1024 kernel

var aiProfile struct {
	tokens                                                          int
	norm, pack, qkv, attn, wo, ffn, gateUp, silu, down, layerTotal time.Duration
}

func resetAIProfile() {
	aiProfile = struct {
		tokens                                                          int
		norm, pack, qkv, attn, wo, ffn, gateUp, silu, down, layerTotal time.Duration
	}{}
}

func printAIProfile(tokens int) {
	if !aiProfileOn || tokens == 0 {
		return
	}
	den := float64(tokens)
	total := float64(aiProfile.layerTotal.Microseconds()) / 1000.0 / den
	profiled := float64((aiProfile.norm + aiProfile.pack + aiProfile.qkv + aiProfile.attn +
		aiProfile.wo + aiProfile.ffn + aiProfile.gateUp + aiProfile.silu + aiProfile.down).Microseconds()) / 1000.0 / den
	fmt.Fprintf(os.Stderr,
		"  ai_profile/token: norm=%.2fms pack=%.2fms qkv=%.2fms attn=%.2fms wo=%.2fms ffn=%.2fms gate_up=%.2fms silu=%.2fms down=%.2fms | layers=%.2fms profiled=%.2fms other=%.2fms\n",
		float64(aiProfile.norm.Microseconds())/1000.0/den,
		float64(aiProfile.pack.Microseconds())/1000.0/den,
		float64(aiProfile.qkv.Microseconds())/1000.0/den,
		float64(aiProfile.attn.Microseconds())/1000.0/den,
		float64(aiProfile.wo.Microseconds())/1000.0/den,
		float64(aiProfile.ffn.Microseconds())/1000.0/den,
		float64(aiProfile.gateUp.Microseconds())/1000.0/den,
		float64(aiProfile.silu.Microseconds())/1000.0/den,
		float64(aiProfile.down.Microseconds())/1000.0/den,
		total, profiled, total-profiled,
	)
}

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
	KpEmbd := ((nEmbd + 15) / 16) * 16
	KpQ := ((nQEmbd + 15) / 16) * 16
	KpFF := ((nFF + 15) / 16) * 16

	packActAI := func(src []int8, K int, dst []int8) {
		if aiUseVL32 {
			broadcastPack8Into(src, K, dst)
		} else {
			broadcastPack1024Into(src, K, dst)
		}
	}
	gemmAI := func(M, K int, wPacked1024, wPackedVL32, actPacked []int8, wScale, actScale float32, out []float32) {
		if aiUseVL32 {
			GemmAIPooledVL32(M, K, wPackedVL32, actPacked, wScale, actScale, out, pool)
		} else {
			GemmAIPooled(M, K, wPacked1024, actPacked, wScale, actScale, out, pool)
		}
	}
	matVecRef := func(forceExact bool, M, K int, raw, packed1024 []int8, q41 q4kQ41Packed, x32 q4kQ41x32, scales, mins []float32, f32 []float32, act []float32, out []float32) {
		if forceExact && scales != nil && mins != nil {
			matVecQ4KF32(M, K, raw, scales, mins, act, out)
			return
		}
		if q4kExactHalfOn && packed1024 != nil && mins != nil {
			q4kBlockMatVecAIPackedHalf(M, K, packed1024, scales, mins, act, out, pool)
			return
		}
		if (q4kExactX32On || q4kExactHalfOn) && x32.Valid && mins != nil {
			q4kQ41x32MatVecExactAI(x32, mins, act, out, pool)
			return
		}
		if q4kLlamaX32On && x32.Valid {
			q4kQ41x32MatVecAI(x32, act, out, pool)
			return
		}
		if scales != nil && mins != nil {
			if q4kAIOn {
				if q4kLlamaZPOn && q41.Valid {
					q4kBlockMatVecQ41(q41, act, out, pool)
				} else {
					q4kBlockMatVecAIPacked(M, K, packed1024, scales, mins, act, out, pool)
				}
				return
			}
			if q4kBlockRefOn {
				matVecQ4KF32(M, K, raw, scales, mins, act, out)
				return
			}
		}
		matVecF32Direct(M, K, f32, act, out)
	}
	runScalarWorkers := func(fn func(workerID, nWorkers int)) {
		if scalarUseAI {
			pool.Run(fn)
		} else if globalPool != nil {
			globalPool.Run(fn)
		} else {
			fn(0, 1)
		}
	}

	// Per-token scratch buffers. Keeping these outside the layer loop avoids
	// hundreds of short-lived allocations per decoded token.
	xn := make([]float32, nEmbd)
	xn2 := make([]float32, nEmbd)
	scoresPools := make([][]float32, 64)
	for i := range scoresPools {
		scoresPools[i] = make([]float32, 512)
	}
	actI8Pad := make([]int8, KpEmbd)
	actPacked := make([]int8, 8*KpEmbd)
	qF := make([]float32, nQEmbd)
	kF := make([]float32, nKVD)
	vF := make([]float32, nKVD)
	woActPad := make([]int8, KpQ)
	woActPacked := make([]int8, 8*KpQ)
	woOut := make([]float32, nEmbd)
	ffnActPad := make([]int8, KpEmbd)
	ffnActPacked := make([]int8, 8*KpEmbd)
	gateF := make([]float32, nFF)
	upF := make([]float32, nFF)
	hidden := make([]float32, nFF)
	downActPad := make([]int8, KpFF)
	downActPacked := make([]int8, 8*KpFF)
	downF := make([]float32, nEmbd)

	for il := 0; il < nLayers; il++ {
		l := &layers[il]
		pos := nPast
		var tLayer time.Time
		if aiProfileOn { tLayer = time.Now() }
		// RMS Norm (scalar, main core)
		t0 := time.Time{}
		if aiProfileOn {
			t0 = time.Now()
		}
		var ss float32
		for i := 0; i < nEmbd; i++ {
			ss += x[i] * x[i]
		}
		invRMS := float32(1.0 / math.Sqrt(float64(ss/float32(nEmbd)+rmsEps)))
		for i := 0; i < nEmbd; i++ {
			xn[i] = x[i] * invRMS * l.attnNorm[i]
		}
		if il == 0 && os.Getenv("IME2_DETAIL") != "" {
			fmt.Fprintf(os.Stderr, "il0 xn_attn[0:8]:")
			for _, v := range xn[:8] { fmt.Fprintf(os.Stderr, " %.5f", v) }
			fmt.Fprintf(os.Stderr, "\n")
		}
		if aiProfileOn {
			aiProfile.norm += time.Since(t0)
			t0 = time.Now()
		}

		// Quantize + pack activation for native GemmAI/VL32 paths.
		// Skipped when using Q4K_A100 path (does its own per-subblock quantization).
		var actScale float32
		if !q4kAIOn || aiUseVL32 {
			actScale = quantizeToI8(xn, actI8Pad[:nEmbd])
			clear(actI8Pad[nEmbd:])
			packActAI(actI8Pad, KpEmbd, actPacked)
		}
		if aiProfileOn {
			aiProfile.pack += time.Since(t0)
			t0 = time.Now()
		}

		// QKV matmuls
		if f32MatvecOn || q4kBlockRefOn || q4kAIOn || q4kLlamaX32On || q4kExactX32On || q4kExactHalfOn {
			matVecRef(q4kExactAttnOn || q4kExactQOn, nQEmbd, nEmbd, l.wqRaw, l.wqPacked1024, l.wqQ41, l.wqX32, l.wqScales, l.wqMins, l.wqF32, xn, qF)
			matVecRef(q4kExactAttnOn || q4kExactKOn, nKVD, nEmbd, l.wkRaw, l.wkPacked1024, l.wkQ41, l.wkX32, l.wkScales, l.wkMins, l.wkF32, xn, kF)
			matVecRef(q4kExactAttnOn || q4kExactVOn, nKVD, nEmbd, l.wvRaw, l.wvPacked1024, l.wvQ41, l.wvX32, l.wvScales, l.wvMins, l.wvF32, xn, vF)
			if il == 0 && os.Getenv("IME2_DETAIL") != "" {
				fmt.Fprintf(os.Stderr, "il0 qF[0:8]:")
				for _, v := range qF[:8] { fmt.Fprintf(os.Stderr, " %.5f", v) }
				fmt.Fprintf(os.Stderr, "\nil0 kF[0:8]:")
				for _, v := range kF[:8] { fmt.Fprintf(os.Stderr, " %.5f", v) }
				fmt.Fprintf(os.Stderr, "\nil0 vF[0:8]:")
				for _, v := range vF[:8] { fmt.Fprintf(os.Stderr, " %.5f", v) }
				fmt.Fprintf(os.Stderr, "\n")
			}		} else if aiUseVL32 {
			GemmAIPooledBatchVL32(pool,
				aiGemmSpec{M: nQEmbd, K: KpEmbd, wPacked: l.wqPacked, actPacked: actPacked, wScale: l.wqScale, actScale: actScale, out: qF},
				aiGemmSpec{M: nKVD, K: KpEmbd, wPacked: l.wkPacked, actPacked: actPacked, wScale: l.wkScale, actScale: actScale, out: kF},
				aiGemmSpec{M: nKVD, K: KpEmbd, wPacked: l.wvPacked, actPacked: actPacked, wScale: l.wvScale, actScale: actScale, out: vF},
			)
		} else {
			GemmAIPooledBatch(pool,
				aiGemmSpec{M: nQEmbd, K: KpEmbd, wPacked: l.wqPacked1024, actPacked: actPacked, wScale: l.wqScale, actScale: actScale, out: qF},
				aiGemmSpec{M: nKVD, K: KpEmbd, wPacked: l.wkPacked1024, actPacked: actPacked, wScale: l.wkScale, actScale: actScale, out: kF},
				aiGemmSpec{M: nKVD, K: KpEmbd, wPacked: l.wvPacked1024, actPacked: actPacked, wScale: l.wvScale, actScale: actScale, out: vF},
			)
			if q4kMinCorrOn {
				applyQ4KMinCorr(qF, l.wqMins, xn, nQEmbd, nEmbd)
				applyQ4KMinCorr(kF, l.wkMins, xn, nKVD, nEmbd)
				applyQ4KMinCorr(vF, l.wvMins, xn, nKVD, nEmbd)
			}
		}
		if aiProfileOn {
			aiProfile.qkv += time.Since(t0)
			t0 = time.Now()
		}

		// KV cache + attention (scalar, main core)
		if l.kNorm != nil {
			for kh := 0; kh < nKVHeads; kh++ {
				head := kF[kh*headDim : (kh+1)*headDim]
				var s2 float32
				if scalarAttnOn {
					for _, v := range head { s2 += v * v }
				} else {
					s2 = ime2.DotF32RVV(head, head)
				}
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
		runHead := func(h int, scores []float32) {
			qHead := qF[h*headDim : (h+1)*headDim]
			if l.qNorm != nil {
				var s3 float32
				if scalarAttnOn {
					for _, v := range qHead { s3 += v * v }
				} else {
					s3 = ime2.DotF32RVV(qHead, qHead)
				}
				inv := float32(1.0 / math.Sqrt(float64(s3/float32(headDim)+rmsEps)))
				for d := range qHead { qHead[d] = qHead[d] * inv * l.qNorm[d] }
			}
			applyRoPE(qHead, headDim, pos, ropeBase)
			kvH := h / repFactor
			scores = scores[:pos+1]
			// Q·K dot products
			var maxScore float32 = -1e30
			kBase := kvH * headDim
			for t := 0; t <= pos; t++ {
				var sc float32
				if scalarAttnOn {
					for d := 0; d < headDim; d++ { sc += qHead[d] * kCache[il][t*nKVD+kBase+d] }
				} else {
					sc = ime2.DotF32RVV(qHead, kCache[il][t*nKVD+kBase:t*nKVD+kBase+headDim])
				}
				scores[t] = sc * invSqrtD
				if scores[t] > maxScore { maxScore = scores[t] }
			}
			var sumExp float32
			for i := range scores {
				scores[i] = float32(math.Exp(float64(scores[i] - maxScore)))
				sumExp += scores[i]
			}
			for i := range scores { scores[i] /= sumExp }
			// V weighted sum: qHead[d] = sum_t(scores[t] * V[t,kvH,d])
			out := qF[h*headDim : (h+1)*headDim]
			clear(out)
			vBase := kvH * headDim
			for t := 0; t <= pos; t++ {
				if scores[t] != 0 {
					if scalarAttnOn {
						for d := 0; d < headDim; d++ { out[d] += scores[t] * vCache[il][t*nKVD+vBase+d] }
					} else {
						ime2.ScaleAccF32RVV(out, vCache[il][t*nKVD+vBase:t*nKVD+vBase+headDim], scores[t])
					}
				}
			}
		}
		runScalarWorkers(func(workerID, nWorkers int) {
			scores := scoresPools[workerID]
			for h := workerID; h < nHeads; h += nWorkers {
				runHead(h, scores)
			}
		})

		if aiProfileOn {
			aiProfile.attn += time.Since(t0)
			t0 = time.Now()
		}

		// WO projection on AI cores
		var woActScale float32
		if !q4kAIOn || aiUseVL32 {
			woActScale = quantizeToI8(qF[:nQEmbd], woActPad[:nQEmbd])
			clear(woActPad[nQEmbd:])
			packActAI(woActPad, KpQ, woActPacked)
		}
		if aiProfileOn {
			aiProfile.pack += time.Since(t0)
			t0 = time.Now()
		}
		if f32MatvecOn || q4kBlockRefOn || q4kAIOn || q4kLlamaX32On || q4kExactX32On || q4kExactHalfOn {
			matVecRef(q4kExactWOOn, nEmbd, nQEmbd, l.woRaw, l.woPacked1024, l.woQ41, l.woX32, l.woScales, l.woMins, l.woF32, qF[:nQEmbd], woOut)
			if il == 0 && os.Getenv("IME2_DETAIL") != "" {
				fmt.Fprintf(os.Stderr, "il0 woOut[0:8]:")
				for _, v := range woOut[:8] { fmt.Fprintf(os.Stderr, " %.5f", v) }
				fmt.Fprintf(os.Stderr, "\n")
			}
		} else {
			gemmAI(nEmbd, KpQ, l.woPacked1024, l.woPacked, woActPacked, l.woScale, woActScale, woOut)
			if q4kMinCorrOn {
				applyQ4KMinCorr(woOut, l.woMins, qF[:nQEmbd], nEmbd, nQEmbd)
			}
		}
		for i := 0; i < nEmbd; i++ {
			x[i] += woOut[i]
		}
		if aiProfileOn {
			aiProfile.wo += time.Since(t0)
			t0 = time.Now()
		}

		// FFN norm + quantize
		ss = 0
		for i := 0; i < nEmbd; i++ {
			ss += x[i] * x[i]
		}
		invRMS = float32(1.0 / math.Sqrt(float64(ss/float32(nEmbd)+rmsEps)))
		for i := 0; i < nEmbd; i++ {
			xn2[i] = x[i] * invRMS * l.ffnNorm[i]
		}
		var ffnActScale float32
		if !q4kAIOn || aiUseVL32 {
			ffnActScale = quantizeToI8(xn2, ffnActPad[:nEmbd])
			clear(ffnActPad[nEmbd:])
		}
		if os.Getenv("IME2_NORM_TRACE") != "" {
			var s float32; var mx float32
			for _, v := range xn2 {
				s += v*v
				if v < 0 { v = -v }
				if v > mx { mx = v }
			}
			fmt.Fprintf(os.Stderr, "layer %2d xn2_rms=%.4f xn2_max=%.4f xn2[0:6]= %.4f %.4f %.4f %.4f %.4f %.4f\n",
				il, float32(math.Sqrt(float64(s/float32(nEmbd)))), mx, xn2[0], xn2[1], xn2[2], xn2[3], xn2[4], xn2[5])
		}
		if !q4kAIOn || aiUseVL32 {
			packActAI(ffnActPad, KpEmbd, ffnActPacked)
		}
		if aiProfileOn {
			aiProfile.ffn += time.Since(t0)
			t0 = time.Now()
		}

		// Gate + Up
		if f32MatvecOn || q4kBlockRefOn || q4kAIOn || q4kLlamaX32On || q4kExactX32On || q4kExactHalfOn {
			matVecRef(q4kExactFFNOn, nFF, nEmbd, l.gateRaw, l.gatePacked1024, l.gateQ41, l.gateX32, l.gateScales, l.gateMins, l.gateF32, xn2, gateF)
			matVecRef(q4kExactFFNOn, nFF, nEmbd, l.upRaw, l.upPacked1024, l.upQ41, l.upX32, l.upScales, l.upMins, l.upF32, xn2, upF)
			if il <= 2 && os.Getenv("IME2_NORM_TRACE") != "" {
				var us float32
				for _, v := range upF { us += v * v }
				// Also print the first row of upF32 and xn2 to check sources
				var wrs float32
				for _, v := range l.upF32[:nEmbd] { wrs += v*v }
				fmt.Fprintf(os.Stderr, "layer %2d upF_rms=%.4f upW_row0_rms=%.4f upW_row0[0:4]=%.5f %.5f %.5f %.5f upF[0:4]=%.4f %.4f %.4f %.4f\n",
					il, float32(math.Sqrt(float64(us/float32(nFF)))),
					float32(math.Sqrt(float64(wrs/float32(nEmbd)))),
					l.upF32[0], l.upF32[1], l.upF32[2], l.upF32[3],
					upF[0], upF[1], upF[2], upF[3])
			}
		} else if aiUseVL32 {
			GemmAIPooledBatchVL32(pool,
				aiGemmSpec{M: nFF, K: KpEmbd, wPacked: l.gatePacked, actPacked: ffnActPacked, wScale: l.gateScale, actScale: ffnActScale, out: gateF},
				aiGemmSpec{M: nFF, K: KpEmbd, wPacked: l.upPacked, actPacked: ffnActPacked, wScale: l.upScale, actScale: ffnActScale, out: upF},
			)
		} else {
			GemmAIPooledBatch(pool,
				aiGemmSpec{M: nFF, K: KpEmbd, wPacked: l.gatePacked1024, actPacked: ffnActPacked, wScale: l.gateScale, actScale: ffnActScale, out: gateF},
				aiGemmSpec{M: nFF, K: KpEmbd, wPacked: l.upPacked1024, actPacked: ffnActPacked, wScale: l.upScale, actScale: ffnActScale, out: upF},
			)
			if q4kMinCorrOn {
				applyQ4KMinCorr(gateF, l.gateMins, xn2, nFF, nEmbd)
				applyQ4KMinCorr(upF, l.upMins, xn2, nFF, nEmbd)
			}
		}
		if aiProfileOn {
			aiProfile.gateUp += time.Since(t0)
			t0 = time.Now()
		}

		// SiLU + element-wise multiply (worker pool)
		runScalarWorkers(func(workerID, nWorkers int) {
			start := workerID * nFF / nWorkers
			end := (workerID + 1) * nFF / nWorkers
			for i := start; i < end; i++ {
				hidden[i] = silu(gateF[i]) * upF[i]
			}
		})
		if os.Getenv("IME2_NORM_TRACE") != "" {
			var sg, su float32
			for i := 0; i < nFF; i++ { sg += gateF[i]*gateF[i]; su += upF[i]*upF[i] }
			fmt.Fprintf(os.Stderr, "layer %2d gate_rms=%.4f up_rms=%.4f gF[0:3]= %.4f %.4f %.4f\n",
				il, float32(math.Sqrt(float64(sg/float32(nFF)))), float32(math.Sqrt(float64(su/float32(nFF)))), gateF[0], gateF[1], gateF[2])
		}
		if aiProfileOn {
			aiProfile.silu += time.Since(t0)
			t0 = time.Now()
		}

		// Down projection on AI cores
		var downActScale float32
		if !q4kAIOn || aiUseVL32 {
			downActScale = quantizeToI8(hidden, downActPad[:nFF])
			clear(downActPad[nFF:])
			packActAI(downActPad, KpFF, downActPacked)
		}
		if aiProfileOn {
			aiProfile.pack += time.Since(t0)
			t0 = time.Now()
		}
		if il <= 2 && os.Getenv("IME2_NORM_TRACE") != "" {
			var hs, ws float32
			for _, v := range hidden { hs += v * v }
			for i := 0; i < nEmbd; i++ {
				for k := 0; k < 4; k++ { ws += l.downF32[i*nFF+k] * l.downF32[i*nFF+k] }
			}
			fmt.Fprintf(os.Stderr, "layer %2d pre-down: hidden_rms=%.4f downW[0:4]=%.4f %.4f %.4f %.4f hidden[0:4]=%.4f %.4f %.4f %.4f\n",
				il, float32(math.Sqrt(float64(hs/float32(nFF)))),
				l.downF32[0], l.downF32[1], l.downF32[2], l.downF32[3],
				hidden[0], hidden[1], hidden[2], hidden[3])
		}
		if f32MatvecOn || q4kBlockRefOn || q4kAIOn || q4kLlamaX32On || q4kExactX32On || q4kExactHalfOn {
			matVecRef(q4kExactFFNOn, nEmbd, nFF, l.downRaw, l.downPacked1024, l.downQ41, l.downX32, l.downScales, l.downMins, l.downF32, hidden, downF)
		} else {
			gemmAI(nEmbd, KpFF, l.downPacked1024, l.downPacked, downActPacked, l.downScale, downActScale, downF)
			if q4kMinCorrOn {
				applyQ4KMinCorr(downF, l.downMins, hidden, nEmbd, nFF)
			}
		}
		for i := 0; i < nEmbd; i++ {
			x[i] += downF[i]
		}
		if os.Getenv("IME2_NORM_TRACE") != "" {
			var xss, dss float32
			for _, v := range x { xss += v * v }
			for _, v := range downF { dss += v * v }
			fmt.Fprintf(os.Stderr, "layer %2d down_rms=%.4f x_rms=%.4f x[0:4]= %.4f %.4f %.4f %.4f\n",
				il, float32(math.Sqrt(float64(dss/float32(nEmbd)))),
				float32(math.Sqrt(float64(xss/float32(nEmbd)))), x[0], x[1], x[2], x[3])
		}
		if aiProfileOn {
			aiProfile.down += time.Since(t0)
			aiProfile.layerTotal += time.Since(tLayer)
		}
	}
	if aiProfileOn {
		aiProfile.tokens++
	}
}

// quantizeToI8 quantizes float32 to int8 with single global scale.
// Returns inverse scale (maxAbs/127).
func quantizeToI8(src []float32, dst []int8) float32 {
	if rvvQuantOn && len(src) > 0 {
		maxAbs := ime2.FindMaxAbsRVV(src)
		if maxAbs == 0 {
			clear(dst[:len(src)])
			return 0
		}
		s := float32(127.0) / maxAbs
		ime2.QuantizeF32ToI8RVV(src, s, dst[:len(src)])
		return maxAbs / 127.0
	}
	var maxAbs float32
	for _, v := range src {
		a := v
		if a < 0 {
			a = -a
		}
		if a > maxAbs {
			maxAbs = a
		}
	}
	if maxAbs == 0 {
		return 0
	}
	s := float32(127.0) / maxAbs
	for i, v := range src {
		q := v * s
		if q > 127 {
			q = 127
		} else if q < -128 {
			q = -128
		}
		dst[i] = int8(q)
	}
	return maxAbs / 127.0
}

// broadcastPack1024Into packs K int8 values into the native VLEN=1024
// activation broadcast layout. Each tile is 8 copies of 16 bytes; dst must be
// at least 8*K.
func broadcastPack1024Into(src []int8, K int, dst []int8) {
	if K%16 != 0 {
		panic("ime2run: broadcastPack1024Into requires K%16==0")
	}
	for ki := 0; ki < K; ki += 16 {
		tileBase := (ki / 16) * 128
		for r := 0; r < 8; r++ {
			copy(dst[tileBase+r*16:tileBase+(r+1)*16], src[ki:ki+16])
		}
	}
}

// broadcastPack8Into packs K int8 values into the legacy 4×8 broadcast tile
// layout used by the forced-vl=32 AI-core kernel. dst must be at least 4*K.
func broadcastPack8Into(src []int8, K int, dst []int8) {
	if K%8 != 0 {
		panic("ime2run: broadcastPack8Into requires K%8==0")
	}
	for ki := 0; ki < K; ki += 8 {
		tileBase := (ki / 8) * 32
		for r := 0; r < 4; r++ {
			copy(dst[tileBase+r*8:tileBase+(r+1)*8], src[ki:ki+8])
		}
	}
}

var _ = unsafe.Pointer(nil)
