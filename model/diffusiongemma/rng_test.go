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

func TestMT19937UniformIntMatchesLibStdCPowerOfTwoVocab(t *testing.T) {
	r := NewMT19937RNG(1)
	want := []int{109319, 261406, 188828, 244464, 29, 33587, 79254, 261892, 38471, 61889}
	for i, w := range want {
		if got := r.Intn(262144); got != w {
			t.Fatalf("Intn(262144)[%d]=%d want libstdc++ %d", i, got, w)
		}
	}
}

func TestMT19937MixedUniformDrawsMatchLibStdC(t *testing.T) {
	r := NewMT19937RNG(1)
	for i := 0; i < 10; i++ {
		_ = r.Intn(262144)
	}
	wantU := []float64{0.0923385918, 0.186260208, 0.34556073, 0.396767467, 0.53881675}
	wantI := []int{103961, 101688, 175569, 245245, 221855}
	for i := range wantU {
		u := r.Float64()
		if u < wantU[i]-1e-7 || u > wantU[i]+1e-7 {
			t.Fatalf("Float64[%d]=%.10g want %.10g", i, u, wantU[i])
		}
		if got := r.Intn(262144); got != wantI[i] {
			t.Fatalf("mixed Intn[%d]=%d want libstdc++ %d", i, got, wantI[i])
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
