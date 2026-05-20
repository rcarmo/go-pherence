package runtime

import "testing"

func TestRuntimeCapabilities(t *testing.T) {
	c := RuntimeCapabilities()
	if c.Arch == "" {
		t.Fatal("empty architecture")
	}
	if c.HasGemvSym {
		t.Fatal("Q4 GEMV asm capability unexpectedly enabled before AVX2/NEON kernels land")
	}
	if c.HasDequant {
		t.Fatal("Q4 dequant asm capability unexpectedly enabled before AVX2/NEON kernels land")
	}
}
