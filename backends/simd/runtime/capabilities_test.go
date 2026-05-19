package simd

import "testing"

func TestRuntimeCapabilities(t *testing.T) {
	c := RuntimeCapabilities()
	if c.Arch == "" {
		t.Fatal("empty architecture")
	}
	if HasSgemmAsm != c.HasSGEMM {
		t.Fatalf("HasSgemmAsm=%v, RuntimeCapabilities.HasSGEMM=%v", HasSgemmAsm, c.HasSGEMM)
	}
	if HasVecAsm != c.HasVec {
		t.Fatalf("HasVecAsm=%v, RuntimeCapabilities.HasVec=%v", HasVecAsm, c.HasVec)
	}
	if HasDotAsm != c.HasDot {
		t.Fatalf("HasDotAsm=%v, RuntimeCapabilities.HasDot=%v", HasDotAsm, c.HasDot)
	}
	if c.HasRoPE != (hasRoPEAsm && c.HasVec) {
		t.Fatalf("HasRoPE=%v, hasRoPEAsm=%v HasVec=%v", c.HasRoPE, hasRoPEAsm, c.HasVec)
	}
	if c.HasRoPE {
		t.Fatal("RoPE asm capability unexpectedly enabled before AVX2/NEON kernels land")
	}
	t.Logf("SIMD capabilities: arch=%s avx2=%v fma=%v neon=%v vec=%v dot=%v sgemm=%v bf16=%v pack=%v rope=%v",
		c.Arch, c.HasAVX2, c.HasFMA, c.HasNEON, c.HasVec, c.HasDot, c.HasSGEMM, c.HasBF16, c.HasPack, c.HasRoPE)
}

func TestSdotLengthMismatch(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	y := []float32{10, 20}
	got := Sdot(x, y)
	want := float32(50)
	if got != want {
		t.Fatalf("Sdot length mismatch=%v want %v", got, want)
	}
}
