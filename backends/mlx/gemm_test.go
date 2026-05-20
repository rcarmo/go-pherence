package mlx

import "testing"

func TestGemmMatchesRepeatedGemv(t *testing.T) {
	qw := makeBenchMLXWeight(3, 8, 4)
	x := []float32{
		1, 2, 3, 4, 5, 6, 7, 8,
		-1, -2, -3, -4, -5, -6, -7, -8,
	}
	got := make([]float32, 2*qw.OutDim+1)
	got[len(got)-1] = 123
	if !Gemm(got, x, 2, qw) {
		t.Fatal("Gemm returned false for valid inputs")
	}
	want := make([]float32, 2*qw.OutDim)
	Gemv(want[:qw.OutDim], x[:qw.InDim], qw)
	Gemv(want[qw.OutDim:], x[qw.InDim:], qw)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%g want %g (all=%v)", i, got[i], want[i], got)
		}
	}
	if got[len(got)-1] != 123 {
		t.Fatalf("Gemm mutated tail: %v", got)
	}
}

func TestGemmParallelMatchesRepeatedGemv(t *testing.T) {
	qw := makeBenchMLXWeight(16, 64, 32)
	batch := 5
	x := make([]float32, batch*qw.InDim)
	for i := range x {
		x[i] = float32((i%13)-6) * 0.25
	}
	got := make([]float32, batch*qw.OutDim)
	if !Gemm(got, x, batch, qw) {
		t.Fatal("Gemm returned false for valid parallel-sized inputs")
	}
	want := make([]float32, len(got))
	for b := 0; b < batch; b++ {
		Gemv(want[b*qw.OutDim:(b+1)*qw.OutDim], x[b*qw.InDim:(b+1)*qw.InDim], qw)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%g want %g", i, got[i], want[i])
		}
	}
}

func TestGemvToRejectsMalformedInputs(t *testing.T) {
	qw := makeBenchMLXWeight(3, 8, 4)
	if !GemvTo(make([]float32, 3), make([]float32, 8), qw) {
		t.Fatal("GemvTo returned false for valid input")
	}
	if GemvTo(make([]float32, 2), make([]float32, 8), qw) {
		t.Fatal("GemvTo accepted short out")
	}
	if GemvTo(make([]float32, 3), make([]float32, 7), qw) {
		t.Fatal("GemvTo accepted short x")
	}
	if GemvTo(make([]float32, 3), make([]float32, 8), nil) {
		t.Fatal("GemvTo accepted nil weight")
	}
}

func TestGemmRejectsMalformedInputs(t *testing.T) {
	qw := makeBenchMLXWeight(3, 8, 4)
	if Gemm(make([]float32, 3), make([]float32, 8), 0, qw) {
		t.Fatal("Gemm accepted zero batch")
	}
	if Gemm(make([]float32, 3), make([]float32, 7), 1, qw) {
		t.Fatal("Gemm accepted short x")
	}
	if Gemm(make([]float32, 2), make([]float32, 8), 1, qw) {
		t.Fatal("Gemm accepted short out")
	}
}
