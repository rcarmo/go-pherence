package nvfp4

import "testing"

func TestRuntimeCapabilities(t *testing.T) {
	c := RuntimeCapabilities()
	if c.Arch == "" {
		t.Fatal("empty architecture")
	}
	if c.HasDecode {
		t.Fatal("NVFP4 decode asm capability unexpectedly enabled before AVX2/NEON kernels land")
	}
	if c.HasDequant {
		t.Fatal("NVFP4 dequant asm capability unexpectedly enabled before AVX2/NEON kernels land")
	}
	if c.HasGemv {
		t.Fatal("NVFP4 GEMV asm capability unexpectedly enabled before AVX2/NEON kernels land")
	}
}
