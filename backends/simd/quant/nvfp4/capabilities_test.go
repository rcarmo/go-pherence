package nvfp4

import "testing"

func TestRuntimeCapabilities(t *testing.T) {
	c := RuntimeCapabilities()
	if c.Arch == "" {
		t.Fatal("empty architecture")
	}
	if c.HasDecode {
		t.Fatal("NVFP4 decode asm capability unexpectedly enabled before dedicated decode kernels land")
	}
	wantRVV := c.Arch == "riscv64" && c.HasRVV
	if c.HasDequant != wantRVV {
		t.Fatalf("HasDequant=%v want %v arch=%s rvv=%v", c.HasDequant, wantRVV, c.Arch, c.HasRVV)
	}
	if c.HasGemv != wantRVV {
		t.Fatalf("HasGemv=%v want %v arch=%s rvv=%v", c.HasGemv, wantRVV, c.Arch, c.HasRVV)
	}
}
