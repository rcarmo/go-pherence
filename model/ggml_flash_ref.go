package model

import (
	"math"
	"os"
	"strings"

	"github.com/rcarmo/go-pherence/half"
)

func mtpPureGoFlashEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_MTP_PURE_FLASH")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func tryPureGoFlashAttentionInto(out, q, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale float32) bool {
	if !mtpPureGoFlashEnabled() {
		return false
	}
	qDim := numHeads * headDim
	if len(out) < qDim || len(q) < qDim {
		return false
	}
	got := ggmlFlashAttnF16KVReference(q[:qDim], kCache, vCache, seqLen, numHeads, numKVHeads, headDim, scale)
	if len(got) != qDim {
		return false
	}
	copy(out[:qDim], got)
	return true
}

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
