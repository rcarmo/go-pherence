package ime2

import (
	"testing"
	"unsafe"
)

func refMatmul4x8(A, B []int8, C []int32) {
	// C[4x4] = A[4x8] * B[4x8]^T (signed x signed)
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			var sum int32
			for k := 0; k < 8; k++ {
				sum += int32(A[i*8+k]) * int32(B[j*8+k])
			}
			C[i*4+j] = sum
		}
	}
}

func TestVmadotSS4x8(t *testing.T) {
	A := make([]int8, 32)
	B := make([]int8, 32)
	C_hw := make([]int32, 16)
	C_ref := make([]int32, 16)

	// Fill with test pattern
	for i := range A {
		A[i] = int8((i*7+3)%127 - 63)
		B[i] = int8((i*13+5)%127 - 63)
	}

	vmadotSS4x8((*byte)(unsafe.Pointer(&A[0])), (*byte)(unsafe.Pointer(&B[0])), &C_hw[0])
	refMatmul4x8(A, B, C_ref)

	errors := 0
	for i := 0; i < 16; i++ {
		if C_hw[i] != C_ref[i] {
			t.Errorf("C[%d]: hw=%d ref=%d", i, C_hw[i], C_ref[i])
			errors++
		}
	}
	if errors == 0 {
		t.Log("vmadotSS4x8: all 16 elements match reference!")
	}
}

func TestVmadotAccumulate(t *testing.T) {
	A := make([]int8, 32)
	B := make([]int8, 32)
	C := make([]int32, 16)

	for i := range A {
		A[i] = 1
		B[i] = 1
	}

	// First call: C = A*B^T
	vmadotSS4x8((*byte)(unsafe.Pointer(&A[0])), (*byte)(unsafe.Pointer(&B[0])), &C[0])
	if C[0] != 8 { // row0 dot row0 = 8*1*1 = 8
		t.Fatalf("first call: C[0]=%d want 8", C[0])
	}

	// Second call (accumulate): C += A*B^T → should be 16
	vmadotAccSS4x8((*byte)(unsafe.Pointer(&A[0])), (*byte)(unsafe.Pointer(&B[0])), &C[0])
	if C[0] != 16 {
		t.Fatalf("accumulate: C[0]=%d want 16", C[0])
	}
	t.Logf("Accumulate works: C[0]=%d after 2 calls", C[0])
}

func BenchmarkVmadotSS4x8(b *testing.B) {
	A := make([]int8, 32)
	B := make([]int8, 32)
	C := make([]int32, 16)
	for i := range A {
		A[i] = int8(i)
		B[i] = int8(i + 1)
	}

	b.SetBytes(32 + 32) // input bytes
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		vmadotSS4x8((*byte)(unsafe.Pointer(&A[0])), (*byte)(unsafe.Pointer(&B[0])), &C[0])
	}
}
