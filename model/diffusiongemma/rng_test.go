package diffusiongemma

import "testing"

func TestMT19937KnownVectorDefaultSeed(t *testing.T) {
	r := NewMT19937RNG(5489)
	want := []uint32{3499211612, 581869302, 3890346734, 3586334585, 545404204}
	for i, w := range want {
		if got := r.Uint32(); got != w {
			t.Fatalf("mt19937 output %d=%d want %d", i, got, w)
		}
	}
}

func TestMT19937IntnDomainEdges(t *testing.T) {
	r := NewMT19937RNG(0)
	if got := r.Intn(0); got != 0 {
		t.Fatalf("Intn(0)=%d want 0", got)
	}
	if got := r.Intn(1); got != 0 {
		t.Fatalf("Intn(1)=%d want 0", got)
	}
	if got := r.Intn(int(uint64(1) << 32)); got < 0 {
		t.Fatalf("Intn(2^32) returned negative %d", got)
	}
}
