package q4

import "testing"

func makeSymGemmInputs(batch, inDim, outDim int) ([]float32, []int32, []int32, []float32) {
	x := make([]float32, batch*inDim)
	for b := 0; b < batch; b++ {
		row := x[b*inDim : (b+1)*inDim]
		for i := 0; i < inDim; i++ {
			v := float32(((b + 1) * (i%9 + 1)) - 5)
			if (b+i)&1 == 1 {
				v = -v
			}
			row[i] = v * 0.5
		}
	}
	qweight := make([]int32, (inDim/8)*outDim)
	for pack := 0; pack < inDim/8; pack++ {
		for j := 0; j < outDim; j++ {
			vals := []int32{
				int32((pack*5 + j*3 + 0) & 0xF),
				int32((pack*5 + j*3 + 1) & 0xF),
				int32((pack*5 + j*3 + 2) & 0xF),
				int32((pack*5 + j*3 + 3) & 0xF),
				int32((pack*5 + j*3 + 4) & 0xF),
				int32((pack*5 + j*3 + 5) & 0xF),
				int32((pack*5 + j*3 + 6) & 0xF),
				int32((pack*5 + j*3 + 7) & 0xF),
			}
			qweight[pack*outDim+j] = packQ4(vals...)
		}
	}
	gIdx := make([]int32, inDim)
	for i := range gIdx {
		gIdx[i] = int32((i / 4) % 3)
	}
	scales := make([]float32, 3*outDim)
	for g := 0; g < 3; g++ {
		row := scales[g*outDim : (g+1)*outDim]
		for j := 0; j < outDim; j++ {
			row[j] = float32((g+1)*(j%7+1)-5) * 0.25
		}
	}
	return x, qweight, gIdx, scales
}

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

func TestGemmSymMatchesRepeatedGemvBatchTails(t *testing.T) {
	const inDim, outDim = 16, 9
	cases := []struct {
		name  string
		batch int
	}{
		{name: "batch1", batch: 1},
		{name: "batch2", batch: 2},
		{name: "batch3", batch: 3},
		{name: "batch5", batch: 5},
		{name: "batch7", batch: 7},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			x, qweight, gIdx, scales := makeSymGemmInputs(tc.batch, inDim, outDim)
			got := make([]float32, tc.batch*outDim+2)
			got[len(got)-2] = 123
			got[len(got)-1] = 456
			if !GemmSym(got, x, tc.batch, qweight, gIdx, scales, inDim, outDim) {
				t.Fatal("GemmSym returned false")
			}
			want := make([]float32, tc.batch*outDim)
			for b := 0; b < tc.batch; b++ {
				if !GemvSymTo(want[b*outDim:(b+1)*outDim], x[b*inDim:(b+1)*inDim], qweight, gIdx, scales, inDim, outDim) {
					t.Fatal("GemvSymTo returned false")
				}
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("got[%d]=%g want %g", i, got[i], want[i])
				}
			}
			if got[len(got)-2] != 123 || got[len(got)-1] != 456 {
				t.Fatalf("GemmSym mutated tail: %v", got[len(got)-2:])
			}
		})
	}
}

func TestGemmSymMalformedBatchInputsDoNotMutateOutput(t *testing.T) {
	const batch, inDim, outDim = 5, 16, 9
	x, qweight, gIdx, scales := makeSymGemmInputs(batch, inDim, outDim)
	cases := []struct {
		name     string
		outEnd   int
		xEnd     int
		qwEnd    int
		scaleEnd int
	}{
		{name: "short out", outEnd: batch*outDim - 1, xEnd: len(x), qwEnd: len(qweight), scaleEnd: len(scales)},
		{name: "short x", outEnd: batch * outDim, xEnd: len(x) - 1, qwEnd: len(qweight), scaleEnd: len(scales)},
		{name: "short qweight", outEnd: batch * outDim, xEnd: len(x), qwEnd: len(qweight) - 1, scaleEnd: len(scales)},
		{name: "short scales", outEnd: batch * outDim, xEnd: len(x), qwEnd: len(qweight), scaleEnd: len(scales) - 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			buf := make([]float32, batch*outDim+1)
			for i := range buf {
				buf[i] = 77
			}
			if GemmSym(buf[:tc.outEnd], x[:tc.xEnd], batch, qweight[:tc.qwEnd], gIdx, scales[:tc.scaleEnd], inDim, outDim) {
				t.Fatal("GemmSym accepted malformed batch input")
			}
			for i, v := range buf {
				if v != 77 {
					t.Fatalf("output mutated at %d: %g", i, v)
				}
			}
		})
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
