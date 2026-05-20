package gemm

import "testing"

func TestGemmSymMatchesRepeatedGemv(t *testing.T) {
	inDim, outDim, batch := 8, 3, 3
	x := []float32{
		1, 2, 3, 4, 5, 6, 7, 8,
		-1, -2, -3, -4, -5, -6, -7, -8,
		0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 3.5, -3.5,
	}
	qweight := []int32{packQ4(0, 1, 2, 3, 4, 5, 6, 7), packQ4(7, 6, 5, 4, 3, 2, 1, 0), packQ4(8, 9, 10, 11, 12, 13, 14, 15)}
	gIdx := make([]int32, inDim)
	scales := []float32{0.5, -0.25, 1.5}
	got := make([]float32, batch*outDim+1)
	got[len(got)-1] = 123
	if !GemmSym(got, x, batch, qweight, gIdx, scales, inDim, outDim) {
		t.Fatal("GemmSym returned false")
	}
	want := make([]float32, batch*outDim)
	for b := 0; b < batch; b++ {
		if !GemvSymTo(want[b*outDim:(b+1)*outDim], x[b*inDim:(b+1)*inDim], qweight, gIdx, scales, inDim, outDim) {
			t.Fatal("GemvSymTo returned false")
		}
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%g want %g", i, got[i], want[i])
		}
	}
	if got[len(got)-1] != 123 {
		t.Fatal("GemmSym mutated tail")
	}
}

func TestGemmAsymMatchesRepeatedGemv(t *testing.T) {
	inDim, outDim, batch := 8, 8, 2
	x := []float32{
		1, -1, 2, -2, 3, -3, 4, -4,
		0.25, 0.5, 0.75, 1, -0.25, -0.5, -0.75, -1,
	}
	qweight := []int32{
		packQ4(8, 9, 10, 11, 12, 13, 14, 15), packQ4(7, 6, 5, 4, 3, 2, 1, 0),
		packQ4(0, 2, 4, 6, 8, 10, 12, 14), packQ4(15, 13, 11, 9, 7, 5, 3, 1),
		packQ4(1, 3, 5, 7, 9, 11, 13, 15), packQ4(14, 12, 10, 8, 6, 4, 2, 0),
		packQ4(8, 8, 9, 9, 10, 10, 11, 11), packQ4(4, 5, 6, 7, 8, 9, 10, 11),
	}
	qzeros := []int32{packQ4(8, 7, 6, 5, 4, 3, 2, 1)}
	gIdx := make([]int32, inDim)
	scales := []float32{0.5, 0.25, -0.75, 1, -0.5, 0.125, 0.75, -0.25}
	got := make([]float32, batch*outDim+1)
	got[len(got)-1] = 123
	if !Gemm(got, x, batch, qweight, qzeros, gIdx, scales, inDim, outDim, false) {
		t.Fatal("Gemm returned false")
	}
	want := make([]float32, batch*outDim)
	for b := 0; b < batch; b++ {
		if !GemvTo(want[b*outDim:(b+1)*outDim], x[b*inDim:(b+1)*inDim], qweight, qzeros, gIdx, scales, inDim, outDim, false) {
			t.Fatal("GemvTo returned false")
		}
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%g want %g", i, got[i], want[i])
		}
	}
	if got[len(got)-1] != 123 {
		t.Fatal("Gemm mutated tail")
	}
}

func TestGemmRejectsMalformedInputs(t *testing.T) {
	out, x, qw, g, s, inDim, outDim := validGemvQ4Inputs()
	if Gemm(out, x, 0, qw, nil, g, s, inDim, outDim, true) {
		t.Fatal("Gemm accepted zero batch")
	}
	if Gemm(out[:1], x, 1, qw, nil, g, s, inDim, outDim, true) {
		t.Fatal("Gemm accepted short out")
	}
	if Gemm(out, x[:1], 1, qw, nil, g, s, inDim, outDim, true) {
		t.Fatal("Gemm accepted short x")
	}
	if Gemm(out, x, 1, nil, nil, g, s, inDim, outDim, true) {
		t.Fatal("Gemm accepted bad qweight")
	}
}
