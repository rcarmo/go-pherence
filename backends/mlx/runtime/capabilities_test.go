package runtime

import "testing"

func TestRuntimeCapabilities(t *testing.T) {
	c := RuntimeCapabilities()
	if c.Arch == "" {
		t.Fatal("empty architecture")
	}
	if c.HasGemv4 {
		t.Fatal("MLX GEMV4 asm capability unexpectedly enabled before AVX2/NEON kernels land")
	}
	if c.HasDequant {
		t.Fatal("MLX dequant asm capability unexpectedly enabled before AVX2/NEON kernels land")
	}
}
