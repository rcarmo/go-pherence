package k3

import (
	"math"
	"runtime"
	"sync"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
	"golang.org/x/sys/unix"
)

// parallelDecode runs one token through all layers using per-layer parallelism.
// Each of nWorkers goroutines processes a slice of output dimensions.
func parallelDecode(
	x []float32, // [nEmbd] hidden state (modified in place)
	layers []layerWeights,
	nLayers, nEmbd, nHeads, nKVHeads, headDim, nFF int,
	rmsEps, ropeBase float32,
	kCache, vCache [][]float32,
	nPast int,
	nWorkers int,
) {
	nQEmbd := nHeads * headDim
	nKVD := nKVHeads * headDim

	// Pre-allocate pack buffers (one per activation quantize point, reused across layers)
	maxK := nFF
	if nEmbd > maxK {
		maxK = nEmbd
	}
	if nQEmbd > maxK {
		maxK = nQEmbd
	}
	pkAttn := newPackBufs(maxK)
	pkWO := newPackBufs(maxK)
	pkFFN := newPackBufs(maxK)
	pkDown := newPackBufs(maxK)

	// Pre-allocate per-layer buffers (shared across layers)
	xn := make([]float32, nEmbd)
	xn2 := make([]float32, nEmbd)
	qF := make([]float32, nQEmbd)
	kF := make([]float32, nKVD)
	vF := make([]float32, nKVD)
	woOut := make([]float32, nEmbd)
	_ = make([]float32, nFF)
	_ = make([]float32, nFF)
	hidden := make([]float32, nFF)
	downF := make([]float32, nEmbd)

	var wg sync.WaitGroup

	for il := 0; il < nLayers; il++ {
		l := &layers[il]

		// === RMS Norm (parallel over elements) ===
		var ss float32
		for i := 0; i < nEmbd; i++ {
			ss += x[i] * x[i]
		}
		invRMS := float32(1.0 / math.Sqrt(float64(ss/float32(nEmbd)+rmsEps)))
		// Parallel norm multiply
		wg.Add(nWorkers)
		for w := 0; w < nWorkers; w++ {
			wStart := w * nEmbd / nWorkers
			wEnd := (w + 1) * nEmbd / nWorkers
			go func(start, end int) {
				defer wg.Done()
				for i := start; i < end; i++ {
					xn[i] = x[i] * invRMS * l.attnNorm[i]
				}
			}(wStart, wEnd)
		}
		wg.Wait()

		// === QKV Projections (vmadot, parallel over output rows) ===
		Kp := ((nEmbd + 7) / 8) * 8
		actPacked, actScale := quantizeAndPackInto(xn, Kp, pkAttn)
		tilesPerRow := Kp / 8
		wg.Add(nWorkers)
		for w := 0; w < nWorkers; w++ {
			go func(workerID int) {
				defer wg.Done()
				// Q: rows [wStart..wEnd) of nQEmbd
				qStart := (workerID * nQEmbd / nWorkers / 4) * 4
				qEnd := ((workerID + 1) * nQEmbd / nWorkers / 4) * 4
				if workerID == nWorkers-1 {
					qEnd = nQEmbd
				}
				for i := qStart; i < qEnd; i += 4 {
					var acc [16]int32
					ime2.VmadotKLoop((*byte)(unsafe.Pointer(&l.wqPacked[(i/4)*tilesPerRow*32])), (*byte)(unsafe.Pointer(&actPacked[0])), &acc[0], Kp)
					for r := 0; r < 4 && i+r < nQEmbd; r++ {
						qF[i+r] = float32(acc[r*4]) * l.wqScale * actScale
					}
				}
				kStart := (workerID * nKVD / nWorkers / 4) * 4
				kEnd := ((workerID + 1) * nKVD / nWorkers / 4) * 4
				if workerID == nWorkers-1 {
					kEnd = nKVD
				}
				for i := kStart; i < kEnd; i += 4 {
					var kacc [16]int32
					ime2.VmadotKLoop((*byte)(unsafe.Pointer(&l.wkPacked[(i/4)*tilesPerRow*32])), (*byte)(unsafe.Pointer(&actPacked[0])), &kacc[0], Kp)
					for r := 0; r < 4 && i+r < nKVD; r++ {
						kF[i+r] = float32(kacc[r*4]) * l.wkScale * actScale
					}
				}
				for i := kStart; i < kEnd; i += 4 {
					var vacc [16]int32
					ime2.VmadotKLoop((*byte)(unsafe.Pointer(&l.wvPacked[(i/4)*tilesPerRow*32])), (*byte)(unsafe.Pointer(&actPacked[0])), &vacc[0], Kp)
					for r := 0; r < 4 && i+r < nKVD; r++ {
						vF[i+r] = float32(vacc[r*4]) * l.wvScale * actScale
					}
				}
			}(w)
		}
		wg.Wait()

		// === Store K,V in cache + Apply QK norm + RoPE ===
		pos := nPast
		if l.kNorm != nil {
			for kh := 0; kh < nKVHeads; kh++ {
				head := kF[kh*headDim : (kh+1)*headDim]
				var ss2 float32
				for d := range head {
					ss2 += head[d] * head[d]
				}
				inv := float32(1.0 / math.Sqrt(float64(ss2/float32(headDim)+rmsEps)))
				for d := range head {
					head[d] = head[d] * inv * l.kNorm[d]
				}
			}
		}
		copy(kCache[il][pos*nKVD:pos*nKVD+nKVD], kF)
		copy(vCache[il][pos*nKVD:pos*nKVD+nKVD], vF)
		// RoPE on cached K
		for kh := 0; kh < nKVHeads; kh++ {
			applyRoPE(kCache[il][pos*nKVD+kh*headDim:pos*nKVD+(kh+1)*headDim], headDim, pos, ropeBase)
		}

		// === Attention (parallel over Q heads) ===
		repFactor := nHeads / nKVHeads
		invSqrtD := float32(1.0 / math.Sqrt(float64(headDim)))
		wg.Add(nWorkers)
		for w := 0; w < nWorkers; w++ {
			hStart := w * nHeads / nWorkers
			hEnd := (w + 1) * nHeads / nWorkers
			go func(hs, he int) {
				defer wg.Done()
				for h := hs; h < he; h++ {
					qHead := make([]float32, headDim) // TODO: pre-allocate
					copy(qHead, qF[h*headDim:(h+1)*headDim])
					if l.qNorm != nil {
						var ss3 float32
						for d := range qHead {
							ss3 += qHead[d] * qHead[d]
						}
						inv := float32(1.0 / math.Sqrt(float64(ss3/float32(headDim)+rmsEps)))
						for d := range qHead {
							qHead[d] = qHead[d] * inv * l.qNorm[d]
						}
					}
					applyRoPE(qHead, headDim, pos, ropeBase)
					kvH := h / repFactor
					scores := make([]float32, pos+1)
					var maxScore float32 = -1e30
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
						qF[h*headDim+d] = sum // reuse qF as attn output
					}
				}
			}(hStart, hEnd)
		}
		wg.Wait()

		// === WO Projection (vmadot, parallel) ===
		KpWO := ((nQEmbd + 7) / 8) * 8
		actWO, actScaleWO := quantizeAndPackInto(qF[:nQEmbd], KpWO, pkWO)
		wg.Add(nWorkers)
		for w := 0; w < nWorkers; w++ {
			go func(workerID int) {
				defer wg.Done()
				KpWO := ((nQEmbd + 7) / 8) * 8
				wStart := (workerID * nEmbd / nWorkers / 4) * 4
				wEnd := ((workerID + 1) * nEmbd / nWorkers / 4) * 4
				if workerID == nWorkers-1 {
					wEnd = nEmbd
				}
				tpr := KpWO / 8
				for i := wStart; i < wEnd; i += 4 {
					var acc [16]int32
					ime2.VmadotKLoop((*byte)(unsafe.Pointer(&l.woPacked[(i/4)*tpr*32])), (*byte)(unsafe.Pointer(&actWO[0])), &acc[0], KpWO)
					for r := 0; r < 4 && i+r < nEmbd; r++ {
						woOut[i+r] = float32(acc[r*4]) * l.woScale * actScaleWO
					}
				}
			}(w)
		}
		wg.Wait()
		for i := 0; i < nEmbd; i++ {
			x[i] += woOut[i]
		}

		// === FFN Norm ===
		ss = 0
		for i := 0; i < nEmbd; i++ {
			ss += x[i] * x[i]
		}
		invRMS = float32(1.0 / math.Sqrt(float64(ss/float32(nEmbd)+rmsEps)))
		for i := 0; i < nEmbd; i++ {
			xn2[i] = x[i] * invRMS * l.ffnNorm[i]
		}

		// === Gate + Up + SiLU (vmadot, parallel) ===
		KpFFN := ((nEmbd + 7) / 8) * 8
		actFFN, actScaleFFN := quantizeAndPackInto(xn2, KpFFN, pkFFN)
		wg.Add(nWorkers)
		for w := 0; w < nWorkers; w++ {
			go func(workerID int) {
				defer wg.Done()
				fStart := (workerID * nFF / nWorkers / 4) * 4
				fEnd := ((workerID + 1) * nFF / nWorkers / 4) * 4
				if workerID == nWorkers-1 {
					fEnd = nFF
				}
				tpr := KpFFN / 8
				for i := fStart; i < fEnd; i += 4 {
					var gacc, uacc [16]int32
					ime2.VmadotKLoop((*byte)(unsafe.Pointer(&l.gatePacked[(i/4)*tpr*32])), (*byte)(unsafe.Pointer(&actFFN[0])), &gacc[0], KpFFN)
					ime2.VmadotKLoop((*byte)(unsafe.Pointer(&l.upPacked[(i/4)*tpr*32])), (*byte)(unsafe.Pointer(&actFFN[0])), &uacc[0], KpFFN)
					for r := 0; r < 4 && i+r < nFF; r++ {
						g := float32(gacc[r*4]) * l.gateScale * actScaleFFN
						u := float32(uacc[r*4]) * l.upScale * actScaleFFN
						hidden[i+r] = silu(g) * u
					}
				}
			}(w)
		}
		wg.Wait()

		// === Down Projection (vmadot, parallel) ===
		KpDown := ((nFF + 7) / 8) * 8
		actDown, actScaleDown := quantizeAndPackInto(hidden, KpDown, pkDown)
		wg.Add(nWorkers)
		for w := 0; w < nWorkers; w++ {
			go func(workerID int) {
				defer wg.Done()
				dStart := (workerID * nEmbd / nWorkers / 4) * 4
				dEnd := ((workerID + 1) * nEmbd / nWorkers / 4) * 4
				if workerID == nWorkers-1 {
					dEnd = nEmbd
				}
				tpr := KpDown / 8
				for i := dStart; i < dEnd; i += 4 {
					var acc [16]int32
					ime2.VmadotKLoop((*byte)(unsafe.Pointer(&l.downPacked[(i/4)*tpr*32])), (*byte)(unsafe.Pointer(&actDown[0])), &acc[0], KpDown)
					for r := 0; r < 4 && i+r < nEmbd; r++ {
						downF[i+r] = float32(acc[r*4]) * l.downScale * actScaleDown
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

// Ensure imports are used
var _ = runtime.LockOSThread
var _ = unix.SchedSetaffinity
var _ = unsafe.Pointer(nil)
var _ = ime2.GemmINT8Packed
