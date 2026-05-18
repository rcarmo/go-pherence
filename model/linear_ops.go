package model

import (
	"math"
	"runtime"
	"sync"
	"unsafe"

	"github.com/rcarmo/go-pherence/backends/simd"
)

func rmsNormInPlace(x, weight []float32, eps float32) {
	simd.RMSNorm(x, weight, eps)
}

// gemv: out = x @ w where w is either:
//
//	pre-transposed [inDim, outDim] (use NN), or
//	original [outDim, inDim] (use NT via dot products)
func gemv(out, x []float32, w []float32, inDim, outDim int) {
	for i := range out {
		out[i] = 0
	}
	weightLen, ok := checkedProduct(inDim, outDim)
	if inDim <= 0 || outDim <= 0 || !ok || len(out) < outDim || len(x) < inDim || len(w) < weightLen {
		return
	}
	if len(w) >= weightLen {
		// Detect layout: if w is [inDim, outDim] (pre-transposed), use NN
		// If w is [outDim, inDim] (original), use NT (dot per output)
		// Heuristic: try NN first (pre-transposed path)
		if simd.HasSgemmAsm {
			simd.SgemmNN(1, outDim, inDim, 1.0,
				unsafe.Pointer(&x[0]), unsafe.Pointer(&w[0]), unsafe.Pointer(&out[0]),
				inDim, outDim, outDim)
		} else {
			for j := 0; j < outDim; j++ {
				sum := float32(0)
				for p := 0; p < inDim; p++ {
					sum += x[p] * w[p*outDim+j]
				}
				out[j] = sum
			}
		}
	}
}

// gemvNT: out = x @ w^T where w is [outDim, inDim] (original layout)
func gemvNT(out, x []float32, w []float32, inDim, outDim int) {
	for i := range out {
		out[i] = 0
	}
	weightLen, ok := checkedProduct(inDim, outDim)
	if inDim <= 0 || outDim <= 0 || !ok || len(out) < outDim || len(x) < inDim || len(w) < weightLen {
		return
	}
	for j := 0; j < outDim; j++ {
		sum := float32(0)
		row := w[j*inDim : (j+1)*inDim]
		if inDim >= 8 {
			sum = simd.Sdot(x, row)
		} else {
			for p := 0; p < inDim; p++ {
				sum += x[p] * row[p]
			}
		}
		out[j] = sum
	}
}

func checkedProduct(a, b int) (int, bool) {
	if a < 0 || b < 0 {
		return 0, false
	}
	maxInt := int(^uint(0) >> 1)
	if b != 0 && a > maxInt/b {
		return 0, false
	}
	return a * b, true
}

func geluTanh(x float32) float32 {
	// GELU with tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
	x3 := x * x * x
	inner := float32(0.7978845608) * (x + 0.044715*x3) // sqrt(2/pi) ≈ 0.7978845608
	return 0.5 * x * (1.0 + float32(math.Tanh(float64(inner))))
}

// gemvNTParallel is like gemvNT but parallelized across CPU cores.
func gemvNTParallel(out, x []float32, w []float32, inDim, outDim int) {
	for i := range out {
		out[i] = 0
	}
	weightLen, ok := checkedProduct(inDim, outDim)
	if inDim <= 0 || outDim <= 0 || !ok || len(out) < outDim || len(x) < inDim || len(w) < weightLen {
		return
	}
	nCPU := runtime.NumCPU()
	if nCPU > 8 {
		nCPU = 8
	} // cap at 8 for cache efficiency
	chunkSize := (outDim + nCPU - 1) / nCPU

	var wg sync.WaitGroup
	wg.Add(nCPU)
	for c := 0; c < nCPU; c++ {
		start := c * chunkSize
		end := start + chunkSize
		if end > outDim {
			end = outDim
		}
		go func(s, e int) {
			defer wg.Done()
			for j := s; j < e; j++ {
				row := w[j*inDim : (j+1)*inDim]
				if inDim >= 8 {
					out[j] = simd.Sdot(x, row)
				} else {
					sum := float32(0)
					for p := 0; p < inDim; p++ {
						sum += x[p] * row[p]
					}
					out[j] = sum
				}
			}
		}(start, end)
	}
	wg.Wait()
}
