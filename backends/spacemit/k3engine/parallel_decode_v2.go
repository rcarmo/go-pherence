package k3engine

import (
	"math"

	"sync"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
	"github.com/rcarmo/go-pherence/backends/spacemit/k3engine/aipool"
	"golang.org/x/sys/unix"
)

// parallelDecodeV2 uses ONE barrier per layer (matching C library architecture).
// Each persistent worker processes its row-slice through ALL matmuls of a layer.
func parallelDecodeV2(
	x []float32,
	layers []layerWeights,
	nLayers, nEmbd, nHeads, nKVHeads, headDim, nFF int,
	rmsEps, ropeBase float32,
	kCache, vCache [][]float32,
	nPast int,
	nWorkers int,
) {
	nQEmbd := nHeads * headDim
	nKVD := nKVHeads * headDim
	KpEmbd := ((nEmbd + 7) / 8) * 8
	KpQEmbd := ((nQEmbd + 7) / 8) * 8
	KpFF := ((nFF + 7) / 8) * 8

	// Shared buffers (written by main between layers, read by workers)
	scoresPool := make([]float32, 512) // max context length
	xn := make([]float32, nEmbd)
	xn2 := make([]float32, nEmbd)
	qF := make([]float32, nQEmbd)
	kF := make([]float32, nKVD)
	vF := make([]float32, nKVD)
	woOut := make([]float32, nEmbd)
	hidden := make([]float32, nFF)
	downF := make([]float32, nEmbd)

	// Per-layer pack buffers (computed once per matmul group, shared read-only)
	pkAttn := aipool.NewPackBufs(nEmbd)
	pkWO := aipool.NewPackBufs(nQEmbd)
	pkFFN := aipool.NewPackBufs(nEmbd)
	pkDown := aipool.NewPackBufs(nFF)

	var wg sync.WaitGroup

	for il := 0; il < nLayers; il++ {
		l := &layers[il]
		pos := nPast

		// === SERIAL: RMS Norm + quantize activation for QKV ===
		var ss float32
		for i := 0; i < nEmbd; i++ {
			ss += x[i] * x[i]
		}
		invRMS := float32(1.0 / math.Sqrt(float64(ss/float32(nEmbd)+rmsEps)))
		for i := 0; i < nEmbd; i++ {
			xn[i] = x[i] * invRMS * l.attnNorm[i]
		}
		actAttn, scAttn := aipool.QuantizeAndPackInto(xn, KpEmbd, pkAttn)
		tprEmbd := KpEmbd / 8

		// === PARALLEL: QKV for all workers ===
		wg.Add(nWorkers)
		for w := 0; w < nWorkers; w++ {
			go func(wid int) {
				defer wg.Done()
				// Q slice
				qS := (wid * nQEmbd / nWorkers / 4) * 4
				qE := ((wid + 1) * nQEmbd / nWorkers / 4) * 4
				if wid == nWorkers-1 {
					qE = nQEmbd
				}
				for i := qS; i < qE; i += 4 {
					var acc [16]int32
					ime2.VmadotKLoop((*byte)(unsafe.Pointer(&l.wqPacked[(i/4)*tprEmbd*32])), (*byte)(unsafe.Pointer(&actAttn[0])), &acc[0], KpEmbd)
					for r := 0; r < 4 && i+r < nQEmbd; r++ {
						qF[i+r] = float32(acc[r*4]) * l.wqScale * scAttn
					}
				}
				// K slice
				kS := (wid * nKVD / nWorkers / 4) * 4
				kE := ((wid + 1) * nKVD / nWorkers / 4) * 4
				if wid == nWorkers-1 {
					kE = nKVD
				}
				for i := kS; i < kE; i += 4 {
					var acc [16]int32
					ime2.VmadotKLoop((*byte)(unsafe.Pointer(&l.wkPacked[(i/4)*tprEmbd*32])), (*byte)(unsafe.Pointer(&actAttn[0])), &acc[0], KpEmbd)
					for r := 0; r < 4 && i+r < nKVD; r++ {
						kF[i+r] = float32(acc[r*4]) * l.wkScale * scAttn
					}
				}
				// V slice
				for i := kS; i < kE; i += 4 {
					var acc [16]int32
					ime2.VmadotKLoop((*byte)(unsafe.Pointer(&l.wvPacked[(i/4)*tprEmbd*32])), (*byte)(unsafe.Pointer(&actAttn[0])), &acc[0], KpEmbd)
					for r := 0; r < 4 && i+r < nKVD; r++ {
						vF[i+r] = float32(acc[r*4]) * l.wvScale * scAttn
					}
				}
			}(w)
		}
		wg.Wait()

		// === SERIAL: KV cache + norm + RoPE + attention + WO quantize ===
		if l.kNorm != nil {
			for kh := 0; kh < nKVHeads; kh++ {
				head := kF[kh*headDim : (kh+1)*headDim]
				var s2 float32
				for d := range head {
					s2 += head[d] * head[d]
				}
				inv := float32(1.0 / math.Sqrt(float64(s2/float32(headDim)+rmsEps)))
				for d := range head {
					head[d] = head[d] * inv * l.kNorm[d]
				}
			}
		}
		copy(kCache[il][pos*nKVD:pos*nKVD+nKVD], kF)
		copy(vCache[il][pos*nKVD:pos*nKVD+nKVD], vF)
		for kh := 0; kh < nKVHeads; kh++ {
			applyRoPE(kCache[il][pos*nKVD+kh*headDim:pos*nKVD+(kh+1)*headDim], headDim, pos, ropeBase)
		}
		// Attention (all heads, serial for now — fast at short sequences)
		repFactor := nHeads / nKVHeads
		invSqrtD := float32(1.0 / math.Sqrt(float64(headDim)))
		for h := 0; h < nHeads; h++ {
			qHead := qF[h*headDim : (h+1)*headDim]
			// QK norm + RoPE on Q
			if l.qNorm != nil {
				var s3 float32
				for d := range qHead {
					s3 += qHead[d] * qHead[d]
				}
				inv := float32(1.0 / math.Sqrt(float64(s3/float32(headDim)+rmsEps)))
				for d := range qHead {
					qHead[d] = qHead[d] * inv * l.qNorm[d]
				}
			}
			applyRoPE(qHead, headDim, pos, ropeBase)
			kvH := h / repFactor
			// Compute scores
			var maxScore float32 = -1e30
			scores := scoresPool[:pos+1]
			for t := 0; t <= pos; t++ {
				var dot float32
				for d := 0; d < headDim; d++ {
					dot += qHead[d] * kCache[il][t*nKVD+kvH*headDim+d]
				}
				scores[t] = dot * invSqrtD
				if scores[t] > maxScore {
					maxScore = scores[t]
				}
			}
			var sumExp float32
			for i := range scores {
				scores[i] = float32(math.Exp(float64(scores[i] - maxScore)))
				sumExp += scores[i]
			}
			for i := range scores {
				scores[i] /= sumExp
			}
			for d := 0; d < headDim; d++ {
				var sum float32
				for t := 0; t <= pos; t++ {
					sum += scores[t] * vCache[il][t*nKVD+kvH*headDim+d]
				}
				qF[h*headDim+d] = sum
			}
		}
		// Quantize attn output for WO
		actWO, scWO := aipool.QuantizeAndPackInto(qF[:nQEmbd], KpQEmbd, pkWO)
		tprQ := KpQEmbd / 8

		// === PARALLEL: WO + FFN norm + gate/up/silu + down ===
		// FFN norm (serial, needed before gate/up)
		// Actually WO needs barrier before FFN. Let me split into 2 dispatches:
		// Dispatch 1: WO
		wg.Add(nWorkers)
		for w := 0; w < nWorkers; w++ {
			go func(wid int) {
				defer wg.Done()
				wS := (wid * nEmbd / nWorkers / 4) * 4
				wE := ((wid + 1) * nEmbd / nWorkers / 4) * 4
				if wid == nWorkers-1 {
					wE = nEmbd
				}
				for i := wS; i < wE; i += 4 {
					var acc [16]int32
					ime2.VmadotKLoop((*byte)(unsafe.Pointer(&l.woPacked[(i/4)*tprQ*32])), (*byte)(unsafe.Pointer(&actWO[0])), &acc[0], KpQEmbd)
					for r := 0; r < 4 && i+r < nEmbd; r++ {
						woOut[i+r] = float32(acc[r*4]) * l.woScale * scWO
					}
				}
			}(w)
		}
		wg.Wait()
		for i := 0; i < nEmbd; i++ {
			x[i] += woOut[i]
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
		actFFN, scFFN := aipool.QuantizeAndPackInto(xn2, KpEmbd, pkFFN)

		// Dispatch 2: gate+up+silu → hidden
		wg.Add(nWorkers)
		for w := 0; w < nWorkers; w++ {
			go func(wid int) {
				defer wg.Done()
				fS := (wid * nFF / nWorkers / 4) * 4
				fE := ((wid + 1) * nFF / nWorkers / 4) * 4
				if wid == nWorkers-1 {
					fE = nFF
				}
				for i := fS; i < fE; i += 4 {
					var gacc, uacc [16]int32
					ime2.VmadotKLoop((*byte)(unsafe.Pointer(&l.gatePacked[(i/4)*tprEmbd*32])), (*byte)(unsafe.Pointer(&actFFN[0])), &gacc[0], KpEmbd)
					ime2.VmadotKLoop((*byte)(unsafe.Pointer(&l.upPacked[(i/4)*tprEmbd*32])), (*byte)(unsafe.Pointer(&actFFN[0])), &uacc[0], KpEmbd)
					for r := 0; r < 4 && i+r < nFF; r++ {
						g := float32(gacc[r*4]) * l.gateScale * scFFN
						u := float32(uacc[r*4]) * l.upScale * scFFN
						hidden[i+r] = silu(g) * u
					}
				}
			}(w)
		}
		wg.Wait()

		// Down projection
		actDown, scDown := aipool.QuantizeAndPackInto(hidden, KpFF, pkDown)
		tprFF := KpFF / 8
		wg.Add(nWorkers)
		for w := 0; w < nWorkers; w++ {
			go func(wid int) {
				defer wg.Done()
				dS := (wid * nEmbd / nWorkers / 4) * 4
				dE := ((wid + 1) * nEmbd / nWorkers / 4) * 4
				if wid == nWorkers-1 {
					dE = nEmbd
				}
				for i := dS; i < dE; i += 4 {
					var acc [16]int32
					ime2.VmadotKLoop((*byte)(unsafe.Pointer(&l.downPacked[(i/4)*tprFF*32])), (*byte)(unsafe.Pointer(&actDown[0])), &acc[0], KpFF)
					for r := 0; r < 4 && i+r < nEmbd; r++ {
						downF[i+r] = float32(acc[r*4]) * l.downScale * scDown
					}
				}
			}(w)
		}
		wg.Wait()
		for i := 0; i < nEmbd; i++ {
			x[i] += downF[i]
		}
	}
}

var _ = unix.SchedSetaffinity
