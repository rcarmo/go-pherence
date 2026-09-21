//go:build riscv64

package whisper

import (
	"math"
	"math/rand"
	"testing"
	"time"

	"github.com/rcarmo/go-pherence/backends/spacemit/ime2"
)

// quantizeRowsI8 quantizes a row-major [rows, K] f32 matrix to int8 with a
// per-row symmetric scale (scale[r] = max|row|/127). Returns int8 data + scales.
func quantizeRowsI8(x []float32, rows, K int) ([]int8, []float32) {
	q := make([]int8, rows*K)
	sc := make([]float32, rows)
	for r := 0; r < rows; r++ {
		base := r * K
		var maxAbs float32
		for k := 0; k < K; k++ {
			if a := float32(math.Abs(float64(x[base+k]))); a > maxAbs {
				maxAbs = a
			}
		}
		s := maxAbs / 127.0
		sc[r] = s
		inv := float32(0)
		if s > 0 {
			inv = 1.0 / s
		}
		for k := 0; k < K; k++ {
			v := x[base+k] * inv
			iv := int32(math.Round(float64(v)))
			if iv > 127 {
				iv = 127
			} else if iv < -127 {
				iv = -127
			}
			q[base+k] = int8(iv)
		}
	}
	return q, sc
}

// TestInt8LinearProbe checks per-row int8 IME GEMM accuracy + speed against the
// F32 path for an encoder-sized linear: out[M,N] = X[M,K] @ W[N,K]^T.
func TestInt8LinearProbe(t *testing.T) {
	M, N, K := 1500, 1280, 1280 // seqLen, outDim, inDim
	rng := rand.New(rand.NewSource(11))
	// Realistic-ish: small Gaussian-like weights, activations with some spread.
	X := make([]float32, M*K)
	W := make([]float32, N*K)
	for i := range X {
		X[i] = float32(rng.NormFloat64()) * 0.5
	}
	for i := range W {
		W[i] = float32(rng.NormFloat64()) * 0.05
	}

	// F32 reference (single-thread, direct).
	ref := make([]float32, M*N)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var s float32
			xo, wo := i*K, j*K
			for k := 0; k < K; k++ {
				s += X[xo+k] * W[wo+k]
			}
			ref[i*N+j] = s
		}
	}

	// int8 path.
	tq := time.Now()
	xi8, xs := quantizeRowsI8(X, M, K)
	wi8, ws := quantizeRowsI8(W, N, K)
	xp := ime2.PackTiles(xi8, M, K)
	wp := ime2.PackTiles(wi8, N, K)
	quantNs := time.Since(tq)

	C := make([]int32, M*N)
	tg := time.Now()
	ime2.GemmINT8Packed(M, N, K, xp, wp, C)
	gemmNs := time.Since(tg)

	C2 := make([]int32, M*N)
	tp := time.Now()
	ime2.GemmINT8PackedParallel(M, N, K, xp, wp, C2, 4)
	gemmParNs := time.Since(tp)

	// Dequantize: out[i,j] = C[i,j] * xs[i] * ws[j].
	out := make([]float32, M*N)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			out[i*N+j] = float32(C[i*N+j]) * xs[i] * ws[j]
		}
	}

	// Accuracy vs F32 reference.
	var maxRel, sumRel float64
	var refRMS float64
	n := 0
	for i := range ref {
		refRMS += float64(ref[i]) * float64(ref[i])
	}
	refRMS = math.Sqrt(refRMS / float64(len(ref)))
	for i := range ref {
		d := math.Abs(float64(out[i] - ref[i]))
		rel := d / refRMS // relative to signal RMS (cells span +/- around 0)
		if rel > maxRel {
			maxRel = rel
		}
		sumRel += rel
		n++
	}
	t.Logf("int8 accuracy: refRMS=%.4f maxErr/RMS=%.4f meanErr/RMS=%.5f", refRMS, maxRel, sumRel/float64(n))
	t.Logf("timing: quant+pack=%v gemm(1t)=%v gemm(4t)=%v", quantNs, gemmNs, gemmParNs)
	mac := float64(M) * float64(N) * float64(K)
	t.Logf("int8 gemm: 1t=%.2f GMAC/s  4t=%.2f GMAC/s", mac/gemmNs.Seconds()/1e9, mac/gemmParNs.Seconds()/1e9)
}
