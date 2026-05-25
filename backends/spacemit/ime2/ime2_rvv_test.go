package ime2

import (
	"math"
	"testing"
	"unsafe"
)

func TestRVVBroadcastPack(t *testing.T) {
	src := make([]int8, 16)
	for i := range src { src[i] = int8(i + 1) }
	dst := make([]int8, 64) // 4*16

	BroadcastPackRVV(src, 16, dst)

	// Check: dst should have [1..8, 1..8, 1..8, 1..8, 9..16, 9..16, 9..16, 9..16]
	for tile := 0; tile < 2; tile++ {
		for row := 0; row < 4; row++ {
			for col := 0; col < 8; col++ {
				expected := int8(tile*8 + col + 1)
				got := dst[tile*32+row*8+col]
				if got != expected {
					t.Errorf("tile%d row%d col%d: got %d want %d", tile, row, col, got, expected)
					return
				}
			}
		}
	}
	t.Log("RVV BroadcastPack: correct!")
}

func TestRMSNormRVV(t *testing.T) {
	n := 64
	x := make([]float32, n)
	w := make([]float32, n)
	out := make([]float32, n)
	ref := make([]float32, n)

	for i := range x { x[i] = float32(i) * 0.1 - 3.2 }
	for i := range w { w[i] = 1.0 + float32(i)*0.01 }

	// Reference
	var ss float32
	for i := range x { ss += x[i] * x[i] }
	
	invRMS := float32(1.0 / math.Sqrt(float64(ss/float32(n)+1e-5)))
	for i := range ref { ref[i] = x[i] * invRMS * w[i] }

	// RVV
	RMSNormFast(x, w, out, 1e-5)

	maxErr := float32(0)
	for i := range out {
		e := out[i] - ref[i]; if e < 0 { e = -e }
		if e > maxErr { maxErr = e }
	}
	t.Logf("RMSNormRVV maxErr: %e", maxErr)
	if maxErr > 1e-5 { t.Errorf("too much error: %e", maxErr) }
}

var _ = unsafe.Pointer(nil)

func TestRVVMulVecVec(t *testing.T) {
	n := 32
	a := make([]float32, n)
	b := make([]float32, n)
	out := make([]float32, n)
	for i := range a { a[i] = float32(i) + 1; b[i] = 2.0 }
	rvvMulVecVec(&a[0], &b[0], &out[0], n)
	for i := 0; i < 8; i++ {
		t.Logf("out[%d] = %.2f (expected %.2f)", i, out[i], a[i]*b[i])
	}
	if out[0] != 2.0 || out[7] != 16.0 {
		t.Errorf("wrong: out[0]=%f out[7]=%f", out[0], out[7])
	}
}

func BenchmarkRMSNormRVV_2048(b *testing.B) {
	x := make([]float32, 2048)
	w := make([]float32, 2048)
	out := make([]float32, 2048)
	for i := range x { x[i] = 0.1; w[i] = 1.5 }
	b.ResetTimer()
	for i := 0; i < b.N; i++ { RMSNormFast(x, w, out, 1e-6) }
}

func BenchmarkRMSNormScalar_2048(b *testing.B) {
	x := make([]float32, 2048)
	w := make([]float32, 2048)
	out := make([]float32, 2048)
	for i := range x { x[i] = 0.1; w[i] = 1.5 }
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var ss float32
		for j := range x { ss += x[j]*x[j] }
		inv := float32(1.0/math.Sqrt(float64(ss/2048+1e-6)))
		for j := range out { out[j] = x[j]*inv*w[j] }
	}
}


func TestFusedPackVmadot(t *testing.T) {
	M, K := 8, 16
	// Weight: row i = all (i+1)
	wI8 := make([]int8, M*K)
	for i := 0; i < M; i++ { for k := 0; k < K; k++ { wI8[i*K+k] = int8(i+1) } }
	wPacked := PackTiles(wI8, M, K)
	// Activation: all 10
	actI8 := make([]int8, K)
	for i := range actI8 { actI8[i] = 10 }
	
	out := make([]int32, M)
	FusedPackVmadot(M, K, wPacked, actI8, out)
	
	// Expected: out[i] = dot(wI8_row_i, actI8) = (i+1) * 10 * K
	// But vmadot only uses first 8 elements per pass, and we do K/8 passes...
	// Actually with K=16: 2 vmadot passes of 8. Each pass: dot(act[8], wt_row[8]) = (i+1)*10*8
	// Total: 2 * (i+1)*10*8 = (i+1)*160
	t.Logf("out = %v", out[:8])
	expected := int32(1 * 10 * K)  // row 0
	t.Logf("expected[0] = %d", expected)
	if out[0] != expected {
		t.Errorf("out[0]=%d want %d", out[0], expected)
	}
}

func TestFindMaxAbsRVV(t *testing.T) {
	x := []float32{1.0, -3.5, 2.1, -0.5, 4.2, -4.8, 0.1, 3.3,
	               1.0, -3.5, 2.1, -0.5, 4.2, -4.8, 0.1, 3.3,
	               1.0, -3.5, 2.1, -0.5, 4.2, -4.8, 0.1, 3.3,
	               1.0, -3.5, 2.1, -0.5, 4.2, -4.8, 0.1, 3.3}
	result := FindMaxAbsRVV(x)
	t.Logf("FindMaxAbsRVV = %f (expected 4.8)", result)
	if result != 4.8 { t.Errorf("got %f want 4.8", result) }
}

func TestQuantizeF32ToI8RVV(t *testing.T) {
	src := make([]float32, 16)
	for i := range src { src[i] = float32(i) - 8 } // [-8, -7, ..., 7]
	dst := make([]int8, 16)
	scale := float32(127.0 / 8.0) // maps [-8,8] to [-127,127]
	QuantizeF32ToI8RVV(src, scale, dst)
	t.Logf("dst = %v", dst[:16])
	// Expected: dst[0] = -127 (clamped), dst[8] = 0, dst[15] = 111
	if dst[8] != 0 { t.Errorf("dst[8]=%d want 0", dst[8]) }
	if dst[0] > -120 { t.Errorf("dst[0]=%d want ~-127", dst[0]) }
}

func TestI8I2KMatVec(t *testing.T) {
	M, K := 8, 16
	// Weights: row i = all value (i+1)
	wI8 := make([]int8, M*K)
	for i := 0; i < M; i++ { for k := 0; k < K; k++ { wI8[i*K+k] = int8(i+1) } }
	wPacked := PackTiles(wI8, M, K)
	// Activation: all 10 (quantized)
	actI8 := make([]int8, K)
	for i := range actI8 { actI8[i] = 10 }
	bc := make([]int8, 4*K)
	copy(bc[0:K], actI8); copy(bc[K:2*K], actI8); copy(bc[2*K:3*K], actI8); copy(bc[3*K:4*K], actI8)
	actPacked := PackTiles(bc, 4, K)
	
	out := make([]float32, M)
	I8I2KMatVec(M, K, wPacked, actPacked, 1.0, 1.0, out)
	
	// Expected: for row i, dot(act=10, wt=i+1 as low2+4*high2)
	// nibble = i+1. low2 = (i+1)&3, high2 = (i+1)>>2
	// dot_low = sum(10 * low2) = 10 * K * low2
	// dot_high = sum(10 * high2) = 10 * K * high2
	// result = (dot_low + 4*dot_high) * 1 * 1 = 10*K*(low2 + 4*high2) = 10*K*nibble = 10*K*(i+1)
	for i := 0; i < M; i++ {
		expected := float32(10 * K * (i + 1))
		t.Logf("out[%d] = %.0f (expected %.0f)", i, out[i], expected)
		if out[i] != expected { t.Errorf("MISMATCH row %d: got %.0f want %.0f", i, out[i], expected) }
	}
}
