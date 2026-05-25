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
