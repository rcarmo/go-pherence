//go:build riscv64

package simd

import "github.com/rcarmo/go-pherence/backends/simd/kernels"

// RVV vector assembly is enabled incrementally. HasVecAsm remains false until
// the full vector surface (including RMSNorm/BF16) has parity-tested RVV
// implementations, but the basic elementwise kernels below can use RVV today.
var hasRVVVecAsm = RuntimeCapabilities().HasRVV

//go:noescape
func vecAddAsm(dst, a, b []float32)

//go:noescape
func vecMulAsm(dst, a, b []float32)

//go:noescape
func vecScaleAddAsm(dst, a, b []float32, scale float32)

//go:noescape
func vecScaleAsm(dst, a []float32, scale float32)

//go:noescape
func rmsNormScaleAsm(x, w []float32, scale float32)

//go:noescape
func toBF16Asm(x []float32)

//go:noescape
func bf16WidenToF32Asm(dst []float32, src []uint16)

//go:noescape
func bf16NarrowFromF32Asm(dst []uint16, src []float32)

//go:noescape
func bf16VecAddAsm(dst, a, b []uint16)

//go:noescape
func bf16DotAsm(x, y []uint16) float32

//go:noescape
func bf16RMSNormScaleAsm(x, w []uint16, scale float32)

func Snrm2(x []float32) float32 {
	if len(x) > 0 && HasDotAsm {
		return float32Sqrt(sdotM4Asm(x, x))
	}
	return snrm2Go(x)
}

func VecAdd(dst, a, b []float32) {
	if len(a) > 0 && len(dst) == len(a) && len(b) == len(a) && hasRVVVecAsm {
		vecAddAsm(dst, a, b)
		return
	}
	vecAddGo(dst, a, b)
}

func VecMul(dst, a, b []float32) {
	if len(a) > 0 && len(dst) == len(a) && len(b) == len(a) && hasRVVVecAsm {
		vecMulAsm(dst, a, b)
		return
	}
	vecMulGo(dst, a, b)
}

func VecScaleAdd(dst, a, b []float32, scale float32) {
	if len(a) > 0 && len(dst) == len(a) && len(b) == len(a) && hasRVVVecAsm {
		vecScaleAddAsm(dst, a, b, scale)
		return
	}
	vecScaleAddGo(dst, a, b, scale)
}

func VecScale(dst, a []float32, scale float32) {
	if len(a) > 0 && len(dst) == len(a) && hasRVVVecAsm {
		vecScaleAsm(dst, a, scale)
		return
	}
	vecScaleGo(dst, a, scale)
}

func VecSiLUMul(dst, a, b []float32) {
	if len(a) > 0 && len(dst) == len(a) && len(b) == len(a) && hasRVVVecAsm {
		kernels.SiLU(dst, a)
		vecMulAsm(dst, dst, b)
		return
	}
	vecSiLUMulGo(dst, a, b)
}
func RMSNorm(x, w []float32, eps float32) { rmsNormGo(x, w, eps) }
func RMSNormNoScale(x []float32, eps float32) {
	if len(x) > 0 && HasDotAsm && hasRVVVecAsm {
		ss := sdotM4Asm(x, x)
		scale := float32(1.0 / float32Sqrt(ss/float32(len(x))+eps))
		vecScaleAsm(x, x, scale)
		return
	}
	rmsNormNoScaleGo(x, eps)
}
func GELUTanhMul(dst, a, b []float32) {
	if len(a) > 0 && len(dst) == len(a) && len(b) == len(a) && hasRVVVecAsm {
		kernels.GELUTanh(dst, a)
		vecMulAsm(dst, dst, b)
		return
	}
	geluTanhMulGo(dst, a, b)
}
func RMSNormBF16(x, w []float32, eps float32) {
	RMSNorm(x, w, eps)
	ToBF16(x)
}
func ToBF16(x []float32) {
	if len(x) > 0 && hasRVVVecAsm {
		toBF16Asm(x)
		return
	}
	toBF16Go(x)
}

func init() { HasVecAsm = RuntimeCapabilities().HasVec }

func BF16DotAsm(x, y []uint16) float32              { return BF16Dot(x, y) }
func BF16RMSNormAsm(x, w []uint16, eps float32)     { BF16RMSNorm(x, w, eps) }
func BF16VecAddAsm(dst, a, b []uint16)              { BF16VecAdd(dst, a, b) }
func BF16WidenToF32(dst []float32, src []uint16)    { bf16WidenToF32Go(dst, src) }
func BF16NarrowFromF32(dst []uint16, src []float32) { bf16NarrowFromF32Go(dst, src) }
