package mlx

import "testing"

func TestGemv2ToMatchesTwoGemvExact(t *testing.T) {
	const (
		inDim     = 256
		groupSize = 64
	)
	x := makeGemmInput(1, inDim)[:inDim]
	cases := []struct {
		name string
		qwA  *QuantWeight
		qwB  *QuantWeight
	}{
		{name: "q4_shared_xsums", qwA: makeQuantWeight(4, inDim, 192, groupSize, 7), qwB: makeQuantWeight(4, inDim, 160, groupSize, 11)},
		{name: "q8_fallback", qwA: makeQuantWeight(8, inDim, 192, groupSize, 13), qwB: makeQuantWeight(8, inDim, 160, groupSize, 17)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			wantA := make([]float32, tc.qwA.OutDim)
			wantB := make([]float32, tc.qwB.OutDim)
			if !GemvTo(wantA, x, tc.qwA) || !GemvTo(wantB, x, tc.qwB) {
				t.Fatal("reference GemvTo failed")
			}
			gotA := make([]float32, tc.qwA.OutDim)
			gotB := make([]float32, tc.qwB.OutDim)
			if !Gemv2To(gotA, gotB, x, tc.qwA, tc.qwB) {
				t.Fatal("Gemv2To failed")
			}
			assertExactFloat32s(t, "outA", gotA, wantA)
			assertExactFloat32s(t, "outB", gotB, wantB)
		})
	}
}

func TestGemv2ToRejectsMalformedInputs(t *testing.T) {
	qwA := makeBenchMLXWeight(3, 8, 4)
	qwB := makeBenchMLXWeight(2, 8, 4)
	if !Gemv2To(make([]float32, 3), make([]float32, 2), make([]float32, 8), qwA, qwB) {
		t.Fatal("Gemv2To returned false for valid input")
	}
	if Gemv2To(make([]float32, 2), make([]float32, 2), make([]float32, 8), qwA, qwB) {
		t.Fatal("Gemv2To accepted short outA")
	}
	if Gemv2To(make([]float32, 3), make([]float32, 1), make([]float32, 8), qwA, qwB) {
		t.Fatal("Gemv2To accepted short outB")
	}
	if Gemv2To(make([]float32, 3), make([]float32, 2), make([]float32, 7), qwA, qwB) {
		t.Fatal("Gemv2To accepted short x")
	}
	if Gemv2To(make([]float32, 3), make([]float32, 2), make([]float32, 8), qwA, nil) {
		t.Fatal("Gemv2To accepted nil weight")
	}
	qwBad := makeBenchMLXWeight(2, 16, 4)
	if Gemv2To(make([]float32, 3), make([]float32, 2), make([]float32, 16), qwA, qwBad) {
		t.Fatal("Gemv2To accepted mismatched input dimensions")
	}
}

func BenchmarkGemv2ToPaired1536x2048(b *testing.B) {
	const (
		inDim     = 1536
		outDim    = 2048
		groupSize = 64
	)
	x := makeGemmInput(1, inDim)[:inDim]
	b.Run("q4_shared_xsums", func(b *testing.B) {
		benchmarkGemv2ToAgainstTwoGemv(b, x, makeQuantWeight(4, inDim, outDim, groupSize, 7), makeQuantWeight(4, inDim, outDim, groupSize, 11))
	})
	b.Run("q8_fallback", func(b *testing.B) {
		benchmarkGemv2ToAgainstTwoGemv(b, x, makeQuantWeight(8, inDim, outDim, groupSize, 13), makeQuantWeight(8, inDim, outDim, groupSize, 17))
	})
}

func benchmarkGemv2ToAgainstTwoGemv(b *testing.B, x []float32, qwA, qwB *QuantWeight) {
	wantA := make([]float32, qwA.OutDim)
	wantB := make([]float32, qwB.OutDim)
	if !GemvTo(wantA, x, qwA) || !GemvTo(wantB, x, qwB) {
		b.Fatal("reference GemvTo failed")
	}
	gotA := make([]float32, qwA.OutDim)
	gotB := make([]float32, qwB.OutDim)
	if !Gemv2To(gotA, gotB, x, qwA, qwB) {
		b.Fatal("Gemv2To failed")
	}
	assertExactFloat32sBench(b, "outA", gotA, wantA)
	assertExactFloat32sBench(b, "outB", gotB, wantB)

	bytes := int64(qwA.InDim*(qwA.OutDim+qwB.OutDim)) * 4
	b.Run("gemv2_to", func(b *testing.B) {
		outA := make([]float32, qwA.OutDim)
		outB := make([]float32, qwB.OutDim)
		b.ReportAllocs()
		b.SetBytes(bytes)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if !Gemv2To(outA, outB, x, qwA, qwB) {
				b.Fatal("Gemv2To failed")
			}
		}
	})
	b.Run("two_gemv_to", func(b *testing.B) {
		outA := make([]float32, qwA.OutDim)
		outB := make([]float32, qwB.OutDim)
		b.ReportAllocs()
		b.SetBytes(bytes)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if !GemvTo(outA, x, qwA) || !GemvTo(outB, x, qwB) {
				b.Fatal("GemvTo failed")
			}
		}
	})
}

func assertExactFloat32s(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s len=%d want %d", name, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s[%d]=%v want %v", name, i, got[i], want[i])
		}
	}
}

func assertExactFloat32sBench(b *testing.B, name string, got, want []float32) {
	b.Helper()
	if len(got) != len(want) {
		b.Fatalf("%s len=%d want %d", name, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			b.Fatalf("%s[%d]=%v want %v", name, i, got[i], want[i])
		}
	}
}
