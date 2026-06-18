package model

import (
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/rcarmo/go-pherence/half"
)

func mtpPureGoFlashEnabled() bool {
	v := strings.TrimSpace(strings.ToLower(os.Getenv("GO_PHERENCE_MTP_PURE_FLASH")))
	return v == "1" || v == "true" || v == "yes" || v == "on"
}

func mtpPureGoFlashFilter(name string) int {
	v := strings.TrimSpace(os.Getenv(name))
	if v == "" {
		return -1
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return -1
	}
	return n
}

func mtpPureGoFlashLayerFilter() int { return mtpPureGoFlashFilter("GO_PHERENCE_MTP_PURE_FLASH_LAYER") }
func mtpPureGoFlashPosFilter() int   { return mtpPureGoFlashFilter("GO_PHERENCE_MTP_PURE_FLASH_POS") }

func tryPureGoFlashAttentionInto(out, q, kCache, vCache []float32, layerIdx, pos, seqLen, numHeads, numKVHeads, headDim int, scale float32) bool {
	if !mtpPureGoFlashEnabled() {
		return false
	}
	if lf := mtpPureGoFlashLayerFilter(); lf >= 0 && lf != layerIdx {
		return false
	}
	if pf := mtpPureGoFlashPosFilter(); pf >= 0 && pf != pos {
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

func ggmlF32ToF16Even(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xff) - 127 + 15
	mant := bits & 0x7fffff
	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint(14 - exp)
		base := mant >> shift
		rem := mant & ((uint32(1) << shift) - 1)
		half := uint32(1) << (shift - 1)
		if rem > half || (rem == half && base&1 != 0) {
			base++
		}
		return sign | uint16(base)
	}
	if exp >= 31 {
		return sign | 0x7c00
	}
	base := mant >> 13
	rem := mant & 0x1fff
	if rem > 0x1000 || (rem == 0x1000 && base&1 != 0) {
		base++
		if base == 0x400 {
			base = 0
			exp++
			if exp >= 31 {
				return sign | 0x7c00
			}
		}
	}
	return sign | uint16(exp<<10) | uint16(base)
}

func ggmlF16VecScaleX86(y []uint16, v float32) {
	const step = 32
	np := len(y) & ^(step - 1)
	for i := 0; i < np; i += step {
		for j := 0; j < step; j++ {
			y[i+j] = ggmlF32ToF16Even(half.F16ToF32(y[i+j]) * v)
		}
	}
	for i := np; i < len(y); i++ {
		y[i] = half.F32ToF16(half.F16ToF32(y[i]) * v)
	}
}

func ggmlF16VecMadX86(y, x []uint16, v float32) {
	n := len(y)
	if len(x) < n {
		n = len(x)
	}
	const step = 32
	np := n & ^(step - 1)
	for i := 0; i < np; i += step {
		for j := 0; j < step; j++ {
			y[i+j] = ggmlF32ToF16Even(half.F16ToF32(y[i+j]) + half.F16ToF32(x[i+j])*v)
		}
	}
	for i := np; i < n; i++ {
		y[i] = half.F32ToF16(half.F16ToF32(y[i]) + half.F16ToF32(x[i])*v)
	}
}

func ggmlFlashAttnF16KVReference(q []float32, kCache, vCache []float32, seqLen, numHeads, numKVHeads, headDim int, scale float32) []float32 {
	out := make([]float32, numHeads*headDim)
	headsPerKV := numHeads / numKVHeads
	kvDim := numKVHeads * headDim
	qF16 := make([]float32, headDim)
	accH := make([]uint16, headDim)
	vH := make([]uint16, headDim)
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
				ggmlF16VecScaleX86(accH[:headDim], ms)
				S *= ms
			} else {
				vs = float32(math.Exp(float64(s - M)))
			}
			vHead := vCache[t*kvDim+kvHead*headDim : t*kvDim+(kvHead+1)*headDim]
			for d := 0; d < headDim; d++ {
				vH[d] = half.F32ToF16(vHead[d])
			}
			ggmlF16VecMadX86(accH[:headDim], vH[:headDim], vs)
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
