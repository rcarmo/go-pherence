package mlx

import (
	"runtime"
	"sync"
)

// Gemv performs matrix-vector multiply with MLX quantized weight.
// out[outDim] = W_mlx[outDim, inDim] · x[inDim] (dequantized on-the-fly)
func Gemv(out, x []float32, qw *QuantWeight) { _ = GemvTo(out, x, qw) }

// GemvTo performs MLX quantized GEMV into caller-owned output and reports
// malformed inputs while preserving Gemv's no-allocation behavior.
func GemvTo(out, x []float32, qw *QuantWeight) bool {
	if err := ValidateQuantWeight(qw); err != nil || len(out) < qw.OutDim || len(x) < qw.InDim {
		return false
	}
	if qw.Bits == 4 && qw.GroupSize%8 == 0 {
		// Dispatch hook kept explicit so AVX2/NEON MLX4 kernels can be wired
		// without changing callers. Scalar remains active until hasGemv4Asm flips.
		gemv4Scalar(out, x, qw)
		return true
	}
	gemvScalar(out, x, qw)
	return true
}

// Gemv2To computes two MLX quantized GEMVs that share the same input vector.
// When both weights use the 4-bit packed path with identical group layout, the
// per-group x sums are computed once and reused for both outputs. Otherwise it
// falls back to two GemvTo calls after validating both weights up front.
func Gemv2To(outA, outB, x []float32, qwA, qwB *QuantWeight) bool {
	if err := ValidateQuantWeight(qwA); err != nil {
		return false
	}
	if err := ValidateQuantWeight(qwB); err != nil {
		return false
	}
	if qwA.InDim != qwB.InDim || len(x) < qwA.InDim || len(outA) < qwA.OutDim || len(outB) < qwB.OutDim {
		return false
	}
	if gemv4SharedXSumsCompatible(qwA, qwB) {
		var groupXSumsStack [gemm4StackGroups]float32
		groupXSums := groupXSumsStack[:]
		if qwA.Groups > len(groupXSumsStack) {
			groupXSums = make([]float32, qwA.Groups)
		} else {
			groupXSums = groupXSums[:qwA.Groups]
		}
		gemv4XSums(x, qwA, groupXSums)
		gemv4Rows(outA, x, qwA, groupXSums, 0, qwA.OutDim)
		gemv4Rows(outB, x, qwB, groupXSums, 0, qwB.OutDim)
		return true
	}
	return GemvTo(outA, x, qwA) && GemvTo(outB, x, qwB)
}

// GemvParallel computes the same result as GemvTo but distributes the output
// rows across goroutines. It is intended for the single-vector autoregressive
// decode projections, where each per-token GEMV otherwise runs on one core.
// Numerics are identical to GemvTo (same per-row dot). Callers that already run
// inside a goroutine pool (e.g. MoE experts) must keep using GemvTo/Gemv to
// avoid oversubscription.
func GemvParallel(out, x []float32, qw *QuantWeight) bool {
	if err := ValidateQuantWeight(qw); err != nil || len(out) < qw.OutDim || len(x) < qw.InDim {
		return false
	}
	rows := qw.OutDim
	nWorkers := runtime.GOMAXPROCS(0)
	if rows < 256 || nWorkers <= 1 {
		return GemvTo(out, x, qw)
	}
	if nWorkers > rows/64 {
		nWorkers = rows / 64
	}
	if nWorkers < 1 {
		nWorkers = 1
	}
	four := qw.Bits == 4 && qw.GroupSize%8 == 0
	var groupXSums []float32
	if four {
		groupXSums = make([]float32, qw.Groups)
		gemv4XSums(x, qw, groupXSums)
	}
	chunk := (rows + nWorkers - 1) / nWorkers
	var wg sync.WaitGroup
	for w := 0; w < nWorkers; w++ {
		r0 := w * chunk
		r1 := r0 + chunk
		if r1 > rows {
			r1 = rows
		}
		if r0 >= r1 {
			break
		}
		wg.Add(1)
		go func(r0, r1 int) {
			defer wg.Done()
			if four {
				gemv4Rows(out, x, qw, groupXSums, r0, r1)
			} else {
				gemvScalarRows(out, x, qw, r0, r1)
			}
		}(r0, r1)
	}
	wg.Wait()
	return true
}

func gemvScalar(out, x []float32, qw *QuantWeight) {
	gemvScalarRows(out, x, qw, 0, qw.OutDim)
}

// gemvScalarRows computes output rows [r0,r1) of a generic MLX quantized GEMV.
func gemvScalarRows(out, x []float32, qw *QuantWeight, r0, r1 int) {
	packFactor := 32 / qw.Bits
	mask := uint32((1 << qw.Bits) - 1)

	for row := r0; row < r1; row++ {
		packedOff := row * (qw.InDim / packFactor)
		scaleOff := row * qw.Groups
		sum := float32(0)

		for g := 0; g < qw.Groups; g++ {
			scale := qw.Scales[scaleOff+g]
			bias := qw.Biases[scaleOff+g]
			gStart := g * qw.GroupSize

			gsum := float32(0)
			xsum := float32(0) // for bias accumulation

			for e := 0; e < qw.GroupSize; e++ {
				idx := gStart + e
				packIdx := idx / packFactor
				bitPos := uint(idx%packFactor) * uint(qw.Bits)
				val := float32((qw.Weight[packedOff+packIdx] >> bitPos) & mask)
				gsum += val * x[idx]
				xsum += x[idx]
			}
			sum += gsum*scale + xsum*bias
		}
		out[row] = sum
	}
}

// gemv4XSums precomputes the per-group sums of x used by the 4-bit kernel.
func gemv4XSums(x []float32, qw *QuantWeight, dst []float32) {
	packsPerGroup := qw.GroupSize / 8
	for g := 0; g < qw.Groups; g++ {
		xBase := g * qw.GroupSize
		xsum := float32(0)
		for p := 0; p < packsPerGroup; p++ {
			xi := xBase + p*8
			xsum += x[xi] + x[xi+1] + x[xi+2] + x[xi+3] + x[xi+4] + x[xi+5] + x[xi+6] + x[xi+7]
		}
		dst[g] = xsum
	}
}

func gemv4SharedXSumsCompatible(qwA, qwB *QuantWeight) bool {
	return qwA.Bits == 4 && qwB.Bits == 4 && qwA.GroupSize%8 == 0 && qwB.GroupSize%8 == 0 && qwA.InDim == qwB.InDim && qwA.Groups == qwB.Groups && qwA.GroupSize == qwB.GroupSize
}

func gemv4Scalar(out, x []float32, qw *QuantWeight) {
	var groupXSumsStack [gemm4StackGroups]float32
	groupXSums := groupXSumsStack[:]
	if qw.Groups > len(groupXSumsStack) {
		groupXSums = make([]float32, qw.Groups)
	} else {
		groupXSums = groupXSums[:qw.Groups]
	}
	gemv4XSums(x, qw, groupXSums)
	gemv4Rows(out, x, qw, groupXSums, 0, qw.OutDim)
}

// gemv4Rows computes output rows [r0,r1) of a 4-bit MLX quantized GEMV using the
// shared per-group x sums.
func gemv4Rows(out, x []float32, qw *QuantWeight, groupXSums []float32, r0, r1 int) {
	packedPerRow := qw.InDim / 8
	packsPerGroup := qw.GroupSize / 8
	for row := r0; row < r1; row++ {
		packedOff := row * packedPerRow
		scaleOff := row * qw.Groups
		sum := float32(0)
		for g := 0; g < qw.Groups; g++ {
			scale := qw.Scales[scaleOff+g]
			bias := qw.Biases[scaleOff+g]
			packBase := packedOff + g*packsPerGroup
			xBase := g * qw.GroupSize
			gsum := float32(0)
			for p := 0; p < packsPerGroup; p++ {
				packed := qw.Weight[packBase+p]
				xi := xBase + p*8
				x0 := x[xi]
				x1 := x[xi+1]
				x2 := x[xi+2]
				x3 := x[xi+3]
				x4 := x[xi+4]
				x5 := x[xi+5]
				x6 := x[xi+6]
				x7 := x[xi+7]
				gsum += float32(packed&0xF)*x0 + float32((packed>>4)&0xF)*x1 + float32((packed>>8)&0xF)*x2 + float32((packed>>12)&0xF)*x3 +
					float32((packed>>16)&0xF)*x4 + float32((packed>>20)&0xF)*x5 + float32((packed>>24)&0xF)*x6 + float32((packed>>28)&0xF)*x7
			}
			sum += gsum*scale + groupXSums[g]*bias
		}
		out[row] = sum
	}
}
