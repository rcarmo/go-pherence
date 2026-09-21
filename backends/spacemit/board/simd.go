package board

import (
	"fmt"
	"github.com/rcarmo/go-pherence/internal/checked"
	"math"
	"runtime"
	"sync"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// SIMDBackend routes ops through the RVV-tuned CPU SIMD paths.
// It is the portable fallback. Malformed inputs return errors before slicing.
type SIMDBackend struct{}

func (SIMDBackend) Name() string { return TierCPU.String() }

// GemvF32: W is [outDim × inDim], GemvRows(out, x, w, rows=outDim, cols=inDim).
// Large matrices are split across rows so the K3 can use multiple X100 cores;
// each row still uses the RVV Sdot kernel inside simd.GemvRows.
func (SIMDBackend) GemvF32(out, x, w []float32, inDim, outDim int) error {
	n, ok := checked.MulInt(inDim, outDim)
	if !ok || inDim <= 0 || outDim <= 0 || len(out) < outDim || len(x) < inDim || len(w) < n {
		return fmt.Errorf("invalid board GEMV buffers/dimensions")
	}
	if outDim < 512 || inDim < 512 {
		simd.GemvRows(out, x, w, outDim, inDim)
		return nil
	}
	workers := runtime.GOMAXPROCS(0)
	if workers < 2 {
		simd.GemvRows(out, x, w, outDim, inDim)
		return nil
	}
	if workers > outDim {
		workers = outDim
	}
	chunk := (outDim-1)/workers + 1
	var wg sync.WaitGroup
	for start := 0; start < outDim; start += chunk {
		end := start + chunk
		if end > outDim {
			end = outDim
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			simd.GemvRows(out[start:end], x, w[start*inDim:end*inDim], end-start, inDim)
		}(start, end)
	}
	wg.Wait()
	return nil
}

func (SIMDBackend) RMSNormF32(x, w []float32, eps float32) error {
	simd.RMSNorm(x, w, eps)
	return nil
}

func (SIMDBackend) RMSNormNoScaleF32(x []float32, eps float32) error {
	simd.RMSNormNoScale(x, eps)
	return nil
}

func (SIMDBackend) SiLUMulF32(dst, gate, up []float32) error {
	simd.SiLUMul(dst, gate, up)
	return nil
}

// GELUTanhMulF32: uses GELUTanhMulTo(dst, gate, up) → dst[i] = gelu_tanh(gate[i])*up[i].
func (SIMDBackend) GELUTanhMulF32(dst, gate, up []float32) error {
	simd.GELUTanhMulTo(dst, gate, up)
	return nil
}

func (SIMDBackend) RoPEPartialF32(x, freqs []float32, pos, nHeads, headDim, rotHalf int) error {
	simd.ApplyRoPEPartial(x, freqs, pos, nHeads, headDim, rotHalf)
	return nil
}

// AttentionScoresF32 computes scaled QK^T logits via a simple dot-product loop.
// This avoids pulling in vCache (not available at this interface level).
func (SIMDBackend) AttentionScoresF32(out, q, kCache []float32, seqLen, nHeads, nKVHeads, headDim int, scale float32) error {
	qN, okQ := checked.MulInt(nHeads, headDim)
	kvWidth, okW := checked.MulInt(nKVHeads, headDim)
	kvN, okK := checked.MulInt(seqLen, kvWidth)
	outN, okO := checked.MulInt(seqLen, nHeads)
	if seqLen <= 0 || headDim <= 0 || nHeads <= 0 || nKVHeads <= 0 || nHeads%nKVHeads != 0 || !okQ || !okW || !okK || !okO || len(q) < qN || len(kCache) < kvN || len(out) < outN || math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return fmt.Errorf("invalid board attention geometry/buffers")
	}
	groupSize := nHeads / nKVHeads
	for h := 0; h < nHeads; h++ {
		kvHead := h / groupSize
		qRow := q[h*headDim : (h+1)*headDim]
		for t := 0; t < seqLen; t++ {
			kRow := kCache[(t*nKVHeads+kvHead)*headDim : (t*nKVHeads+kvHead+1)*headDim]
			sum := simd.Sdot(qRow, kRow)
			out[h*seqLen+t] = sum * scale
		}
	}
	return nil
}
