package simd

import (
	"fmt"
	"runtime"
	"testing"
)

type gemvBenchShape struct {
	rows int
	cols int
}

var gemvBenchShapes = []gemvBenchShape{
	{rows: 128, cols: 256},
	{rows: 128, cols: 1024},
	{rows: 128, cols: 4096},
	{rows: 512, cols: 256},
	{rows: 512, cols: 1024},
	{rows: 512, cols: 4096},
	{rows: 2048, cols: 256},
	{rows: 2048, cols: 1024},
	{rows: 2048, cols: 4096},
}

func gemvDenseBenchBytes(rows, cols int, weightElemBytes int64) int64 {
	return int64(rows*cols)*weightElemBytes + int64(cols*4) + int64(rows*4)
}

func gemvRowsLegacySdot(out, x, w []float32, rows, cols int) {
	for row := 0; row < rows; row++ {
		out[row] = Sdot(x[:cols], w[row*cols:(row+1)*cols])
	}
}

func BenchmarkGemvRowsF32Candidates(b *testing.B) {
	old := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(old)
	for _, shape := range gemvBenchShapes {
		rows, cols := shape.rows, shape.cols
		x := randFloats(cols, int64(rows<<16|cols))
		w := randFloats(rows*cols, int64(rows<<20|cols<<4|1))
		out := make([]float32, rows)
		name := fmt.Sprintf("rows_%d_cols_%d", rows, cols)
		b.Run(name+"/legacy_sdot", func(b *testing.B) {
			b.SetBytes(gemvDenseBenchBytes(rows, cols, 4))
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				gemvRowsLegacySdot(out, x, w, rows, cols)
			}
		})
		b.Run(name+"/rowsx4", func(b *testing.B) {
			b.SetBytes(gemvDenseBenchBytes(rows, cols, 4))
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if !GemvRows(out, x, w, rows, cols) {
					b.Fatal("GemvRows rejected benchmark shape")
				}
			}
		})
	}
}

func BenchmarkGemvRowsThroughput(b *testing.B) {
	old := runtime.GOMAXPROCS(1)
	defer runtime.GOMAXPROCS(old)
	for _, shape := range gemvBenchShapes {
		rows, cols := shape.rows, shape.cols
		x := randFloats(cols, int64(rows<<16|cols))
		wf32 := randFloats(rows*cols, int64(rows<<20|cols<<4|3))
		wbf16 := make([]uint16, len(wf32))
		for i, v := range wf32 {
			wbf16[i] = F32ToBF16(v)
		}
		out := make([]float32, rows)
		name := fmt.Sprintf("rows_%d_cols_%d", rows, cols)
		b.Run(name+"/f32", func(b *testing.B) {
			b.SetBytes(gemvDenseBenchBytes(rows, cols, 4))
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if !GemvRows(out, x, wf32, rows, cols) {
					b.Fatal("GemvRows rejected benchmark shape")
				}
			}
		})
		b.Run(name+"/bf16", func(b *testing.B) {
			b.SetBytes(gemvDenseBenchBytes(rows, cols, 2))
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if !GemvRowsBF16(out, x, wbf16, rows, cols) {
					b.Fatal("GemvRowsBF16 rejected benchmark shape")
				}
			}
		})
	}
}
