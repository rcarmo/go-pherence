package model

import (
	"math"
	"testing"

	"github.com/rcarmo/go-pherence/half"
)

func ggmlFlashAttnF16KVReference(q []float32, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale float32) []float32 {
	out := make([]float32, numHeads*headDim)
	headsPerKV := numHeads / numKVHeads
	kvDim := numKVHeads * headDim
	qF16 := make([]float32, headDim)
	accH := make([]uint16, headDim)
	for head := 0; head < numHeads; head++ {
		kvHead := head / headsPerKV
		for d, v := range q[head*headDim : (head+1)*headDim] {
			qF16[d] = half.F16ToF32(half.F32ToF16(v))
			accH[d] = 0
		}
		S := float32(0)
		M := float32(math.Inf(-1))
		for t := 0; t < seqLen; t++ {
			kHead := kCache[t*kvDim+kvHead*headDim : t*kvDim+(kvHead+1)*headDim]
			var s float32
			// This intentionally remains a simple loop. ggml_vec_dot_f16's SIMD
			// reduction is close but not sufficient alone; this helper is a readable
			// scaffold for porting the whole online-softmax composition.
			for d := 0; d < headDim; d++ {
				s += qF16[d] * half.F16ToF32(half.F32ToF16(kHead[d]))
			}
			s *= scale
			Mold := M
			vs := float32(1)
			if s > M {
				M = s
				ms := float32(math.Exp(float64(Mold - M)))
				for d := 0; d < headDim; d++ {
					accH[d] = half.F32ToF16(half.F16ToF32(accH[d]) * ms)
				}
				S *= ms
			} else {
				vs = float32(math.Exp(float64(s - M)))
			}
			vHead := vCache[t*kvDim+kvHead*headDim : t*kvDim+(kvHead+1)*headDim]
			for d := 0; d < headDim; d++ {
				accH[d] = half.F32ToF16(half.F16ToF32(accH[d]) + half.F16ToF32(half.F32ToF16(vHead[d]))*vs)
			}
			S += vs
		}
		invS := float32(0)
		if S != 0 {
			invS = 1 / S
		}
		for d := 0; d < headDim; d++ {
			out[head*headDim+d] = half.F16ToF32(accH[d]) * invS
		}
	}
	return out
}

func TestGGMLFlashAttnF16KVReferenceShape(t *testing.T) {
	q := make([]float32, 2*4)
	k := make([]float32, 3*1*4)
	v := make([]float32, 3*1*4)
	for i := range q {
		q[i] = float32(i+1) * 0.01
	}
	for i := range k {
		k[i] = float32(i-3) * 0.02
	}
	for i := range v {
		v[i] = float32(i+5) * -0.015
	}
	out := ggmlFlashAttnF16KVReference(q, k, v, 3, 2, 1, 4, 1)
	if len(out) != len(q) {
		t.Fatalf("out len=%d want %d", len(out), len(q))
	}
}
