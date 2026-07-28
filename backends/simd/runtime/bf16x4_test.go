package simd

import (
	"fmt"
	"math"
	"testing"
)

func bf16Dotx4Tolerance(cols int) float64 {
	return 1e-3 + 5e-6*float64(cols)
}

func testBF16Weights(rows, cols int) []uint16 {
	out := make([]uint16, rows*cols)
	for i := range out {
		out[i] = F32ToBF16(float32((i%29)-14) * 0.0625)
	}
	return out
}

func testF32Activation(cols int) []float32 {
	out := make([]float32, cols)
	for i := range out {
		out[i] = float32((i%23)-11) * 0.03125
	}
	return out
}

func testBF16Activation(cols int) []uint16 {
	out := make([]uint16, cols)
	for i := range out {
		out[i] = F32ToBF16(float32((i%19)-9) * 0.046875)
	}
	return out
}

func TestBF16DotF32x4MatchesScalar(t *testing.T) {
	for _, cols := range []int{1, 7, 8, 9, 15, 16, 17, 48, 1024, 4096} {
		w := testBF16Weights(4, cols)
		x := testF32Activation(cols)
		got0, got1, got2, got3, ok := BF16DotF32x4(w, x, cols)
		if !ok {
			t.Fatalf("BF16DotF32x4 rejected cols=%d", cols)
		}
		want := [4]float32{
			BF16DotF32(w[:cols], x),
			BF16DotF32(w[cols:2*cols], x),
			BF16DotF32(w[2*cols:3*cols], x),
			BF16DotF32(w[3*cols:4*cols], x),
		}
		got := [4]float32{got0, got1, got2, got3}
		tol := bf16Dotx4Tolerance(cols)
		for i := range got {
			if diff := math.Abs(float64(got[i] - want[i])); diff > tol {
				t.Fatalf("cols=%d dot%d=%g want %g diff=%g tol=%g", cols, i, got[i], want[i], diff, tol)
			}
		}
	}
}

func TestBF16DotBF16x4MatchesScalar(t *testing.T) {
	for _, cols := range []int{1, 7, 8, 9, 15, 16, 17, 48, 1024, 4096} {
		w := testBF16Weights(4, cols)
		x := testBF16Activation(cols)
		got0, got1, got2, got3, ok := BF16DotBF16x4(w, x, cols)
		if !ok {
			t.Fatalf("BF16DotBF16x4 rejected cols=%d", cols)
		}
		want := [4]float32{
			BF16Dot(w[:cols], x),
			BF16Dot(w[cols:2*cols], x),
			BF16Dot(w[2*cols:3*cols], x),
			BF16Dot(w[3*cols:4*cols], x),
		}
		got := [4]float32{got0, got1, got2, got3}
		tol := bf16Dotx4Tolerance(cols)
		for i := range got {
			if diff := math.Abs(float64(got[i] - want[i])); diff > tol {
				t.Fatalf("cols=%d dot%d=%g want %g diff=%g tol=%g", cols, i, got[i], want[i], diff, tol)
			}
		}
	}
}

func TestBF16Dotx4RejectsMalformedInputs(t *testing.T) {
	w := testBF16Weights(4, 8)
	xf32 := testF32Activation(8)
	xbf16 := testBF16Activation(8)
	if _, _, _, _, ok := BF16DotF32x4(w[:31], xf32, 8); ok {
		t.Fatal("BF16DotF32x4 accepted short weights")
	}
	if _, _, _, _, ok := BF16DotF32x4(w, xf32[:7], 8); ok {
		t.Fatal("BF16DotF32x4 accepted short activation")
	}
	if _, _, _, _, ok := BF16DotBF16x4(w[:31], xbf16, 8); ok {
		t.Fatal("BF16DotBF16x4 accepted short weights")
	}
	if _, _, _, _, ok := BF16DotBF16x4(w, xbf16[:7], 8); ok {
		t.Fatal("BF16DotBF16x4 accepted short activation")
	}
}

func TestGemvRowsBF16TailMatchesDirectDots(t *testing.T) {
	rows, cols := 7, 17
	w := testBF16Weights(rows, cols)
	x := testF32Activation(cols)
	got := make([]float32, rows)
	if !GemvRowsBF16(got, x, w, rows, cols) {
		t.Fatal("GemvRowsBF16 failed")
	}
	tol := bf16Dotx4Tolerance(cols)
	for row := 0; row < rows; row++ {
		want := BF16DotF32(w[row*cols:(row+1)*cols], x)
		if diff := math.Abs(float64(got[row] - want)); diff > tol {
			t.Fatalf("row=%d got=%g want=%g diff=%g tol=%g", row, got[row], want, diff, tol)
		}
	}
}

func TestGemvRowsBF16BF16TailParity(t *testing.T) {
	rows, cols := 7, 17
	w := testBF16Weights(rows, cols)
	x := testBF16Activation(cols)
	want := make([]float32, rows)
	got := make([]float32, rows)
	if !GemvRowsBF16BF16(want, x, w, rows, cols) {
		t.Fatal("GemvRowsBF16BF16 failed")
	}
	if !GemvRowsBF16BF16Parallel(got, x, w, rows, cols) {
		t.Fatal("GemvRowsBF16BF16Parallel failed")
	}
	if !floatsEqual(want, got) {
		t.Fatalf("GemvRowsBF16BF16 tail mismatch\nwant=%v\n got=%v", want, got)
	}
}

func TestGemmRowsBF16ParallelTailParity(t *testing.T) {
	batch, rows, cols := 3, 7, 17
	x := randFloats(batch*cols, 21)
	wf32 := randFloats(rows*cols, 22)
	w := make([]uint16, len(wf32))
	for i, value := range wf32 {
		w[i] = F32ToBF16(value)
	}
	want := make([]float32, batch*rows)
	got := make([]float32, batch*rows)
	for b := 0; b < batch; b++ {
		if !GemvRowsBF16(want[b*rows:(b+1)*rows], x[b*cols:(b+1)*cols], w, rows, cols) {
			t.Fatal("GemvRowsBF16 failed")
		}
	}
	if !GemmRowsBF16Parallel(got, x, w, batch, rows, cols) {
		t.Fatal("GemmRowsBF16Parallel failed")
	}
	for i := range want {
		if diff := math.Abs(float64(want[i] - got[i])); diff > bf16Dotx4Tolerance(cols) {
			t.Fatalf("GemmRowsBF16Parallel tail[%d]=%g want=%g diff=%g", i, got[i], want[i], diff)
		}
	}
}

func benchmarkBF16DotF32x4Shape(b *testing.B, cols int) {
	w := testBF16Weights(4, cols)
	x := testF32Activation(cols)
	b.Run(fmt.Sprintf("cols_%d_x4", cols), func(b *testing.B) {
		var sink float32
		for i := 0; i < b.N; i++ {
			d0, d1, d2, d3, ok := BF16DotF32x4(w, x, cols)
			if !ok {
				b.Fatal("BF16DotF32x4 rejected benchmark shape")
			}
			sink += d0 + d1 + d2 + d3
		}
		_ = sink
	})
	b.Run(fmt.Sprintf("cols_%d_four_scalar", cols), func(b *testing.B) {
		var sink float32
		for i := 0; i < b.N; i++ {
			sink += BF16DotF32(w[:cols], x)
			sink += BF16DotF32(w[cols:2*cols], x)
			sink += BF16DotF32(w[2*cols:3*cols], x)
			sink += BF16DotF32(w[3*cols:4*cols], x)
		}
		_ = sink
	})
}

func BenchmarkBF16DotF32x4ParityShapes(b *testing.B) {
	for _, cols := range []int{1024, 4096} {
		benchmarkBF16DotF32x4Shape(b, cols)
	}
}

func benchmarkBF16DotBF16x4Shape(b *testing.B, cols int) {
	w := testBF16Weights(4, cols)
	x := testBF16Activation(cols)
	b.Run(fmt.Sprintf("cols_%d_x4", cols), func(b *testing.B) {
		var sink float32
		for i := 0; i < b.N; i++ {
			d0, d1, d2, d3, ok := BF16DotBF16x4(w, x, cols)
			if !ok {
				b.Fatal("BF16DotBF16x4 rejected benchmark shape")
			}
			sink += d0 + d1 + d2 + d3
		}
		_ = sink
	})
	b.Run(fmt.Sprintf("cols_%d_four_scalar", cols), func(b *testing.B) {
		var sink float32
		for i := 0; i < b.N; i++ {
			sink += BF16Dot(w[:cols], x)
			sink += BF16Dot(w[cols:2*cols], x)
			sink += BF16Dot(w[2*cols:3*cols], x)
			sink += BF16Dot(w[3*cols:4*cols], x)
		}
		_ = sink
	})
}

func BenchmarkBF16DotBF16x4ParityShapes(b *testing.B) {
	for _, cols := range []int{1024, 4096} {
		benchmarkBF16DotBF16x4Shape(b, cols)
	}
}
