package q4

import "testing"

func TestRuntimeCapabilities(t *testing.T) {
	c := RuntimeCapabilities()
	if c.Arch == "" {
		t.Fatal("empty architecture")
	}
	wantRVV := c.Arch == "riscv64" && c.HasRVV
	if c.HasGemvSym != wantRVV {
		t.Fatalf("HasGemvSym=%v want %v arch=%s rvv=%v", c.HasGemvSym, wantRVV, c.Arch, c.HasRVV)
	}
	if c.HasDequant != wantRVV {
		t.Fatalf("HasDequant=%v want %v arch=%s rvv=%v", c.HasDequant, wantRVV, c.Arch, c.HasRVV)
	}
}
