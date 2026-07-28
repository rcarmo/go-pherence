package mlx

import "testing"

func makeGemmInput(batch, inDim int) []float32 {
	x := make([]float32, batch*inDim)
	for b := 0; b < batch; b++ {
		row := x[b*inDim : (b+1)*inDim]
		for i := 0; i < inDim; i++ {
			v := float32((((b+1)*(i%17+1))%23)-11) * 0.125
			if (b+i)&1 == 1 {
				v = -v
			}
			row[i] = v
		}
	}
	return x
}

func repeatedGemvTo(out, x []float32, batch int, qw *QuantWeight) bool {
	for b := 0; b < batch; b++ {
		if !GemvTo(out[b*qw.OutDim:(b+1)*qw.OutDim], x[b*qw.InDim:(b+1)*qw.InDim], qw) {
			return false
		}
	}
	return true
}

func TestGemmMatchesRepeatedGemvExact(t *testing.T) {
	q4 := makeBenchMLXWeight(19, 64, 16)
	q8 := makeQuantWeight(8, 64, 19, 16, 7)
	cases := []struct {
		name  string
		batch int
		qw    *QuantWeight
	}{
		{name: "batch1_q4", batch: 1, qw: q4},
		{name: "batch2_q4", batch: 2, qw: q4},
		{name: "batch4_q4", batch: 4, qw: q4},
		{name: "batch5_q4_tail1", batch: 5, qw: q4},
		{name: "batch7_q4_tail3", batch: 7, qw: q4},
		{name: "batch3_q8_generic", batch: 3, qw: q8},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			x := makeGemmInput(tc.batch, tc.qw.InDim)
			want := make([]float32, tc.batch*tc.qw.OutDim)
			if !repeatedGemvTo(want, x, tc.batch, tc.qw) {
				t.Fatal("repeated GemvTo failed")
			}
			got := make([]float32, len(want)+1)
			got[len(got)-1] = 123
			if !Gemm(got, x, tc.batch, tc.qw) {
				t.Fatal("Gemm returned false for valid inputs")
			}
			for i := range want {
				if got[i] != want[i] {
					t.Fatalf("got[%d]=%g want %g", i, got[i], want[i])
				}
			}
			if got[len(got)-1] != 123 {
				t.Fatalf("Gemm mutated tail: %v", got)
			}
		})
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
	if Gemm(make([]float32, 3), make([]float32, 8), 1, nil) {
		t.Fatal("Gemm accepted nil weight")
	}
	bad := &QuantWeight{Bits: 4, OutDim: 3, InDim: 8, GroupSize: 4, Groups: 2}
	if Gemm(make([]float32, 3), make([]float32, 8), 1, bad) {
		t.Fatal("Gemm accepted malformed weight")
	}
}
