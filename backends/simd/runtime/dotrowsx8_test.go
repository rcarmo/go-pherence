package simd

import (
	"fmt"
	"math"
	"testing"
)

func TestDotRowsx8MatchesTwoDotRowsx4(t *testing.T) {
	for _, cols := range []int{1, 7, 8, 9, 16, 17, 32, 64, 288, 576, 1152, 2304, 4096, 5120} {
		w := make([]float32, 8*cols)
		x := make([]float32, cols)
		for i := range w {
			w[i] = float32(math.Sin(float64(i)*.17)) * .125
		}
		for i := range x {
			x[i] = float32(math.Cos(float64(i)*.13)) * .25
		}
		got := make([]float32, 8)
		dotRowsx8(got, w, x, cols)
		for block := 0; block < 2; block++ {
			d0, d1, d2, d3 := dotRowsx4(w[block*4*cols:(block+1)*4*cols], x, cols)
			want := [...]float32{d0, d1, d2, d3}
			for i, v := range want {
				at := block*4 + i
				if math.Float32bits(got[at]) != math.Float32bits(v) {
					t.Fatalf("cols=%d row=%d got=%g want=%g", cols, at, got[at], v)
				}
			}
		}
	}
}

func TestDotRowsx8DisabledISAPreservesBits(t *testing.T) {
	old := HasDotAsm
	HasDotAsm = false
	t.Cleanup(func() { HasDotAsm = old })
	const cols = 288
	w := make([]float32, 8*cols)
	x := make([]float32, cols)
	for i := range w {
		w[i] = float32((i%37)-18) * .00390625
	}
	for i := range x {
		x[i] = float32((i%23)-11) * .015625
	}
	got := make([]float32, 8)
	dotRowsx8(got, w, x, cols)
	for row := range got {
		want := Sdot(x, w[row*cols:(row+1)*cols])
		if math.Float32bits(got[row]) != math.Float32bits(want) {
			t.Fatalf("row=%d got=%g want=%g", row, got[row], want)
		}
	}
}

func TestGemvRowsEightRowDispatchPreservesBits(t *testing.T) {
	for _, shape := range [][2]int{{8, 16}, {9, 17}, {32, 288}, {64, 576}, {128, 1152}, {256, 2304}, {256, 4096}, {256, 5120}} {
		rows, cols := shape[0], shape[1]
		w := make([]float32, rows*cols)
		x := make([]float32, cols)
		for i := range w {
			w[i] = float32((i%31)-15) * .0078125
		}
		for i := range x {
			x[i] = float32((i%29)-14) * .015625
		}
		got := make([]float32, rows)
		if !GemvRows(got, x, w, rows, cols) {
			t.Fatal("GemvRows rejected", shape)
		}
		want := make([]float32, rows)
		row := 0
		for ; row+4 <= rows; row += 4 {
			d0, d1, d2, d3 := dotRowsx4(w[row*cols:(row+4)*cols], x, cols)
			want[row+0], want[row+1], want[row+2], want[row+3] = d0, d1, d2, d3
		}
		for ; row < rows; row++ {
			want[row] = Sdot(x, w[row*cols:(row+1)*cols])
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("shape=%v row=%d got=%g want=%g", shape, i, got[i], want[i])
			}
		}
	}
}

func BenchmarkDotRowsx8VsTwoX4(b *testing.B) {
	for _, cols := range []int{288, 576, 1152, 2304} {
		w := make([]float32, 8*cols)
		x := make([]float32, cols)
		out := make([]float32, 8)
		for i := range w {
			w[i] = float32((i%31)-15) * .0078125
		}
		for i := range x {
			x[i] = float32((i%29)-14) * .015625
		}
		b.Run(fmt.Sprintf("rowsx8_cols_%d", cols), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				dotRowsx8(out, w, x, cols)
			}
		})
		b.Run(fmt.Sprintf("two_x4_cols_%d", cols), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				d0, d1, d2, d3 := dotRowsx4(w[:4*cols], x, cols)
				out[0], out[1], out[2], out[3] = d0, d1, d2, d3
				d0, d1, d2, d3 = dotRowsx4(w[4*cols:], x, cols)
				out[4], out[5], out[6], out[7] = d0, d1, d2, d3
			}
		})
	}
}
