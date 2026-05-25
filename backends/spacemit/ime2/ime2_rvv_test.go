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


func TestVmadotQ4KPacked(t *testing.T) {
	// Simple test: 4 rows × 32 elements (one group)
	// Weight nibbles: row i = all value (i+1) = 1,2,3,4
	// Packed: byte = low_nibble | (high_nibble << 4)
	// For nibble value N: low = N (since N<16, fits in one nibble per byte pair)
	// Actually for our layout: byte[j] = nibble[2j] | (nibble[2j+1] << 4)
	// With all nibble = (i+1): byte = (i+1) | ((i+1) << 4) for all bytes
	
	K := 32
	M := 4
	// Raw Q4K qs bytes: 4 rows × 16 bytes/row
	rawQS := make([]byte, M*K/2) // M*(K/2) = 4*16 = 64 bytes
	for r := 0; r < M; r++ {
		val := byte(r + 1)
		packed := val | (val << 4) // same nibble for even/odd
		for j := 0; j < K/2; j++ {
			rawQS[r*(K/2)+j] = packed
		}
	}
	
	// Pack into tile format
	wPacked := PackQ4KForI2K(rawQS, M, K)
	t.Logf("wPacked size: %d (expected 64)", len(wPacked))
	
	// Activation: all 10, reordered to even/odd
	actI8 := make([]int8, K)
	for i := range actI8 { actI8[i] = 10 }
	actReord := ReorderActEvenOdd(actI8, K)
	// Broadcast pack the reordered activation
	bc := make([]int8, 4*K)
	copy(bc[0:K], actReord); copy(bc[K:2*K], actReord)
	copy(bc[2*K:3*K], actReord); copy(bc[3*K:4*K], actReord)
	actTiled := PackTiles(bc, 4, K)
	
	// Run packed vmadot
	acc := make([]int32, 16)
	vmadotQ4KPackedLoop((*byte)(unsafe.Pointer(&wPacked[0])), (*byte)(unsafe.Pointer(&actTiled[0])), &acc[0], K/32)
	
	// Expected: dot(act=10, weight_nibble=i+1) = 10 * K * (i+1) per row
	// With 2-bit split: low = nibble (since nibble < 4 for rows 0-2, but row 3 has nibble=4)
	// nibble=1: low=1, high=0 → reconstructed = 1+4*0 = 1 ✓
	// nibble=2: low=2, high=0 → reconstructed = 2 ✓
	// nibble=3: low=3, high=0 → reconstructed = 3 ✓
	// nibble=4: low=0, high=1 → reconstructed = 0+4*1 = 4 ✓ (4&3=0, 4>>2=1)
	// So for the vslideup merged tile: positions [0:15] = low nibbles, [16:31] = high nibbles
	// vmadot: C[r][c] = dot(act_row_c, weight_row_r)
	// With act reordered [even(16), odd(16)] and weight [low_nib(16), high_nib(16)]:
	// dot = sum(act_even[i]*low_nib[i]) + sum(act_odd[i]*high_nib[i])
	// For uniform act=10 and uniform nibble=N: low=N&3, high=N>>2
	// dot = 10*16*(N&3) + 10*16*(N>>2) = 160*((N&3)+(N>>2))
	// This is NOT = 10*32*N = 320*N unless (N&3)+(N>>2) = 2*N which is wrong!
	//
	// AH: the formula should be: result = dot_low + 4*dot_high (not just sum)
	// But vmadot gives ONE dot product of the full 32-element tile.
	// The merged tile [low(16), high(16)] × act[even(16), odd(16)] gives:
	// = sum(low[i]*act_even[i]) + sum(high[i]*act_odd[i]) for i=0..15
	// We WANT: sum(nibble[2j]*act[2j]) + sum(nibble[2j+1]*act[2j+1])
	// = sum(low_of_byte_j * act[2j]) + sum(high_of_byte_j * act[2j+1])
	// The merged dot gives: sum(low_nib[i] * act_even[i]) + sum(high_nib[i] * act_odd[i])
	// = sum(nibble[2i]&15 * act[2i]) + sum(nibble[2i]>>4... wait no.
	// 
	// Packed byte[j] = nibble[2j] | (nibble[2j+1] << 4)
	// low = byte & 15 = nibble[2j]
	// high = byte >> 4 = nibble[2j+1]
	// Merged: [low[0..15], high[0..15]] = [nibble[0], nibble[2], ..., nibble[30], nibble[1], nibble[3], ..., nibble[31]]
	// Act reordered: [act[0], act[2], ..., act[30], act[1], act[3], ..., act[31]]
	// vmadot dot: sum(nibble[2i]*act[2i]) + sum(nibble[2i+1]*act[2i+1]) = FULL DOT PRODUCT!
	// 
	// So expected = 10 * 32 * (i+1) = 320*(i+1) for row i.
	
	for r := 0; r < 4; r++ {
		expected := int32(10 * K * (r + 1))
		got := acc[r*4] // diagonal element
		t.Logf("row %d: got=%d expected=%d", r, got, expected)
		if got != expected { t.Errorf("MISMATCH row %d", r) }
	}
}
