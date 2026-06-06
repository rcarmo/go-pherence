package rvv

import (
	"math"
	"sync"
	"unsafe"
)

//go:noescape
func kernelM4N32U(a, bp *int8, c *int32, K, lda, ldc int64)

func fround(x float32) float32 {
	if x < 0 {
		return float32(int32(x - 0.5))
	}
	return float32(int32(x + 0.5))
}

// QuantizeDynamicU8 implements ONNX DynamicQuantizeLinear (per-tensor, uint8):
// scale=(max-min)/255 over [min(0,x..),max(0,x..)], zp=round(-min/scale).
func QuantizeDynamicU8(x []float32) (q []uint8, scale float32, zp uint8) {
	mn, mx := float32(0), float32(0)
	for _, v := range x {
		if v < mn {
			mn = v
		} else if v > mx {
			mx = v
		}
	}
	scale = (mx - mn) / 255
	if scale == 0 {
		scale = 1
	}
	zpf := fround(-mn / scale)
	if zpf < 0 {
		zpf = 0
	} else if zpf > 255 {
		zpf = 255
	}
	zp = uint8(zpf)
	q = make([]uint8, len(x))
	inv := float32(1) / scale
	var wg sync.WaitGroup
	nt := 8
	chunk := (len(x) + nt - 1) / nt
	for t := 0; t < nt; t++ {
		s, e := t*chunk, (t+1)*chunk
		if s >= len(x) {
			break
		}
		if e > len(x) {
			e = len(x)
		}
		wg.Add(1)
		go func(s, e int) {
			defer wg.Done()
			for i := s; i < e; i++ {
				r := fround(x[i]*inv) + zpf
				if r < 0 {
					r = 0
				} else if r > 255 {
					r = 255
				}
				q[i] = uint8(r)
			}
		}(s, e)
	}
	wg.Wait()
	return
}

// QuantizeWeightsSym quantizes f32 weights W[N,K] to symmetric per-channel int8:
// wScale[n]=max|W[n,:]|/127. Returns packed (PackB) weights, scales, and the
// per-channel colsum used for activation zero-point correction.
func QuantizeWeightsSym(W []float32, N, K int) (Wp []int8, wScale []float32, wColSum []int32) {
	Wq := make([]int8, N*K)
	wScale = make([]float32, N)
	wColSum = make([]int32, N)
	for n := 0; n < N; n++ {
		mx := float32(0)
		for k := 0; k < K; k++ {
			a := float32(math.Abs(float64(W[n*K+k])))
			if a > mx {
				mx = a
			}
		}
		s := mx / 127
		if s == 0 {
			s = 1
		}
		wScale[n] = s
		var cs int32
		for k := 0; k < K; k++ {
			q := int32(math.Round(float64(W[n*K+k] / s)))
			if q > 127 {
				q = 127
			}
			if q < -127 {
				q = -127
			}
			Wq[n*K+k] = int8(q)
			cs += q
		}
		wColSum[n] = cs
	}
	return PackB(Wq, N, K), wScale, wColSum
}

// gemmU8Outer: raw int32 = aq[M,K] (uint8) · Wp (packed int8), threaded.
func gemmU8Outer(aq []uint8, Wp []int8, raw []int32, M, N, K, nthreads int) {
	mblocks := M / 4
	work := func(mb0, mb1 int) {
		for mb := mb0; mb < mb1; mb++ {
			m := mb * 4
			for nt := 0; nt < N/32; nt++ {
				kernelM4N32U((*int8)(unsafe.Pointer(&aq[m*K])), &Wp[nt*K*32], &raw[m*N+nt*32],
					int64(K), int64(K), int64(N*4))
			}
		}
	}
	if nthreads <= 1 {
		work(0, mblocks)
		return
	}
	var wg sync.WaitGroup
	ch := (mblocks + nthreads - 1) / nthreads
	for t := 0; t < nthreads; t++ {
		a, b := t*ch, (t+1)*ch
		if a >= mblocks {
			break
		}
		if b > mblocks {
			b = mblocks
		}
		wg.Add(1)
		go func(a, b int) { defer wg.Done(); work(a, b) }(a, b)
	}
	wg.Wait()
}

// MatMulIntegerDequant computes Cf32[M,N] ~= Af32[M,K] · W[N,K]^T via the int8
// RVV path: dynamic uint8 quant of A, packed int8 GEMM, then per-channel dequant
// with activation zero-point correction. Wp/wScale/wColSum come from
// QuantizeWeightsSym. Requires M%4==0, N%32==0.
func MatMulIntegerDequant(Af32 []float32, Wp []int8, wScale []float32, wColSum []int32,
	C []float32, M, N, K, nthreads int) {
	aq, aScale, aZp := QuantizeDynamicU8(Af32)
	raw := make([]int32, M*N)
	gemmU8Outer(aq, Wp, raw, M, N, K, nthreads)
	zpf := int32(aZp)
	var wg sync.WaitGroup
	chunk := (M + nthreads - 1) / nthreads
	for t := 0; t < nthreads; t++ {
		m0, m1 := t*chunk, (t+1)*chunk
		if m0 >= M {
			break
		}
		if m1 > M {
			m1 = M
		}
		wg.Add(1)
		go func(m0, m1 int) {
			defer wg.Done()
			for m := m0; m < m1; m++ {
				ro := m * N
				for n := 0; n < N; n++ {
					corr := raw[ro+n] - zpf*wColSum[n]
					C[ro+n] = aScale * wScale[n] * float32(corr)
				}
			}
		}(m0, m1)
	}
	wg.Wait()
}
