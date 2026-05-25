package ime2

import (
	"math"
	"unsafe"
)

// rvvBroadcastPack copies K int8 values into 4× broadcast tile format.
//go:noescape
func rvvBroadcastPack(src *byte, K int, dst *byte)

// rvvMulVecVec computes out[i] = a[i] * b[i] using RVV.
//go:noescape
func rvvMulVecVec(a *float32, b *float32, out *float32, n int)

// RMSNormRVV computes out[i] = x[i] * invRMS * weight[i] with zero allocation.
// Modifies x in-place (scales by invRMS) then uses RVV for x*weight.
// After return, x is scaled (caller should treat as consumed).
func RMSNormRVV(x, weight, out []float32, eps float32) {
	n := len(x)
	var ss float32
	for i := 0; i < n; i++ { ss += x[i] * x[i] }
	invRMS := float32(1.0 / math.Sqrt(float64(ss/float32(n)+eps)))
	for i := 0; i < n; i++ { x[i] *= invRMS }
	rvvMulVecVec(&x[0], &weight[0], &out[0], n)
}

// RMSNormFast is RMSNorm without modifying x (uses out as temp).
func RMSNormFast(x, weight, out []float32, eps float32) {
	n := len(x)
	var ss float32
	for i := 0; i < n; i++ { ss += x[i] * x[i] }
	invRMS := float32(1.0 / math.Sqrt(float64(ss/float32(n)+eps)))
	// Scale into out first, then multiply by weight in-place
	for i := 0; i < n; i++ { out[i] = x[i] * invRMS }
	rvvMulVecVec(&out[0], &weight[0], &out[0], n)
}

// BroadcastPackRVV packs K int8 values into 4× tile format using RVV.
func BroadcastPackRVV(src []int8, K int, dst []int8) {
	rvvBroadcastPack((*byte)(unsafe.Pointer(&src[0])), K, (*byte)(unsafe.Pointer(&dst[0])))
}

// fusedMatVec performs quantize+vmadot+dequant in one assembly function.
// out[i] = sum_k(wPacked[i,k] * quantize(act[k])) * wScale * actInvScale
// M must be multiple of 4. K must be multiple of 8.
//go:noescape

// FusedMatVec is the exported wrapper.

// fusedPackVmadot fuses broadcast-pack + vmadot K-loop.
//go:noescape
func fusedPackVmadot(wPacked *byte, actI8 *byte, M int, K int, out *int32)

// FusedPackVmadot is the exported wrapper.
// Performs C[M] = wPacked[M×K] · actI8[K] (int8 matmul, int32 output at stride 4).
func FusedPackVmadot(M, K int, wPacked []int8, actI8 []int8, out []int32) {
	rawOut := make([]int32, M*4)
	fusedPackVmadot((*byte)(unsafe.Pointer(&wPacked[0])), (*byte)(unsafe.Pointer(&actI8[0])), M, K, &rawOut[0])
	for i := 0; i < M; i++ { out[i] = rawOut[i*4] }
}

// rvvFindMaxAbs returns max(|x[i]|) using RVV vectorized reduction.
//go:noescape
func rvvFindMaxAbs(x *float32, n int) float32

// FindMaxAbsRVV is the exported wrapper.
func FindMaxAbsRVV(x []float32) float32 {
	if len(x) == 0 { return 0 }
	return rvvFindMaxAbs(&x[0], len(x))
}

// rvvQuantizeF32ToI8 quantizes float32 to int8 using RVV.
//go:noescape
func rvvQuantizeF32ToI8(src *float32, scaleBits uint32, dst *byte, n int)

// QuantizeF32ToI8RVV quantizes n float32s to int8 with the given scale.
func QuantizeF32ToI8RVV(src []float32, scale float32, dst []int8) {
	bits := math.Float32bits(scale)
	rvvQuantizeF32ToI8(&src[0], bits, (*byte)(unsafe.Pointer(&dst[0])), len(src))
}

