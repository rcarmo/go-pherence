package gguf

import (
	"math"
	"math/rand"
	"runtime"
	"sync/atomic"
	"testing"
)

func TestQuantizeQ8_0BatchParallelExact(t *testing.T) {
	const batch, width = 124, 256
	rng := rand.New(rand.NewSource(0x8040124))
	x := make([]float32, batch*width)
	for i := range x {
		x[i] = (rng.Float32()*2 - 1) * 8
	}
	want := make([]q8_0Block, batch*width/qk8_0)
	if err := quantizeQ8_0To(want, x); err != nil {
		t.Fatal(err)
	}
	got := make([]q8_0Block, len(want))
	previous := runtime.GOMAXPROCS(6)
	defer runtime.GOMAXPROCS(previous)
	if err := quantizeQ8_0BatchTo(got, x, batch, width); err != nil {
		t.Fatal(err)
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("block %d differs: got=%+v want=%+v", i, got[i], want[i])
		}
	}
}

func TestGemvRowBlocksParallelCoverage(t *testing.T) {
	previous := runtime.GOMAXPROCS(6)
	defer runtime.GOMAXPROCS(previous)

	for _, outDim := range []int{1, 64, 65, 513, 10240} {
		seen := make([]atomic.Int32, outDim)
		if !gemvRowBlocksParallel(outDim, 64, func(start, end int) bool {
			if start < 0 || end <= start || end > outDim || end-start > 64 {
				return false
			}
			for row := start; row < end; row++ {
				seen[row].Add(1)
			}
			return true
		}) {
			t.Fatalf("outDim=%d failed", outDim)
		}
		for row := range seen {
			if got := seen[row].Load(); got != 1 {
				t.Fatalf("outDim=%d row=%d visits=%d want=1", outDim, row, got)
			}
		}
	}

	if gemvRowBlocksParallel(513, 64, func(_, _ int) bool { return false }) {
		t.Fatal("worker failure was not propagated")
	}
	if gemvRowBlocksParallel(0, 64, func(_, _ int) bool { return true }) ||
		gemvRowBlocksParallel(64, 0, func(_, _ int) bool { return true }) {
		t.Fatal("invalid shape accepted")
	}
}

func TestQuantizeQ8_0BatchRejectsMalformedShape(t *testing.T) {
	if err := quantizeQ8_0BatchTo(make([]q8_0Block, 1), make([]float32, 32), 64, 32); err == nil {
		t.Fatal("expected malformed batch error")
	}
}

func TestQuantizeQ8_0BlockSIMDExact(t *testing.T) {
	rng := rand.New(rand.NewSource(0x8040a2))
	rows := make([][qk8_0]float32, 0, 10002)
	rows = append(rows,
		[qk8_0]float32{},
		[qk8_0]float32{127, 0.5, -0.5, 1.5, -1.5, 126.5, -126.5},
		[qk8_0]float32{1, float32(math.NaN()), -1},
		[qk8_0]float32{float32(math.Inf(1)), float32(math.Inf(-1)), 1},
		[qk8_0]float32{math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32},
	)
	for nanPos := 0; nanPos < qk8_0; nanPos++ {
		var row [qk8_0]float32
		for i := range row {
			row[i] = float32(i - 17)
		}
		row[nanPos] = float32(math.NaN())
		rows = append(rows, row)
	}
	for n := -126; n <= 126; n++ {
		tie := float32(n) + 0.5
		for _, value := range []float32{
			math.Nextafter32(tie, float32(math.Inf(-1))),
			tie,
			math.Nextafter32(tie, float32(math.Inf(1))),
		} {
			var row [qk8_0]float32
			row[0], row[1] = 127, value
			rows = append(rows, row)
		}
	}
	for range 10000 {
		var row [qk8_0]float32
		for i := range row {
			row[i] = (rng.Float32()*2 - 1) * float32(1+rng.Intn(10000))
		}
		rows = append(rows, row)
	}
	for i := range rows {
		var want, got q8_0Block
		quantizeQ8_0BlockScalarTo(&want.d, &want.qs, rows[i][:])
		if !quantizeQ8_0BlockSIMD(&got.d, &got.qs, rows[i][:]) {
			t.Skip("AVX2 unavailable")
		}
		if got != want {
			t.Fatalf("row %d differs: got=%+v want=%+v input=%v", i, got, want, rows[i])
		}
	}
}

func TestQuantizeQ8_0Tiles8DirectExact(t *testing.T) {
	previous := runtime.GOMAXPROCS(6)
	defer runtime.GOMAXPROCS(previous)

	for _, tc := range []struct {
		batch int
		width int
	}{
		{batch: 64, width: 32},
		{batch: 65, width: 256},
		{batch: 124, width: 2560},
	} {
		rng := rand.New(rand.NewSource(int64(tc.batch*10000 + tc.width)))
		x := make([]float32, tc.batch*tc.width)
		for i := range x {
			x[i] = (rng.Float32()*2 - 1) * 130
		}
		for i := 0; i < tc.width && i < len(x); i++ {
			x[i] = 0
		}

		blocksPerRow := tc.width / qk8_0
		want := make([]q8_0Block, tc.batch*blocksPerRow)
		if err := quantizeQ8_0To(want, x); err != nil {
			t.Fatal(err)
		}
		fullTokens := tc.batch / 8 * 8
		tiles := make([]q8_0Tile8, fullTokens/8*blocksPerRow)
		tail := make([]q8_0Block, (tc.batch-fullTokens)*blocksPerRow)
		if err := quantizeQ8_0Tiles8To(tiles, tail, x, tc.batch, tc.width); err != nil {
			t.Fatal(err)
		}
		for pos := 0; pos < fullTokens; pos++ {
			for bi := 0; bi < blocksPerRow; bi++ {
				got := q8_0Block{
					d:  tiles[pos/8*blocksPerRow+bi].d[pos%8],
					qs: tiles[pos/8*blocksPerRow+bi].qs[pos%8],
				}
				if got != want[pos*blocksPerRow+bi] {
					t.Fatalf("batch=%d width=%d token=%d block=%d differs", tc.batch, tc.width, pos, bi)
				}
			}
		}
		for i := range tail {
			if tail[i] != want[fullTokens*blocksPerRow+i] {
				t.Fatalf("batch=%d width=%d tail block=%d differs", tc.batch, tc.width, i)
			}
		}
	}
}

func TestQuantizeQ8_0Tiles8RejectsMalformedShape(t *testing.T) {
	if err := quantizeQ8_0Tiles8To(nil, nil, make([]float32, 64*32), 64, 32); err == nil {
		t.Fatal("expected tile-length error")
	}
	if err := quantizeQ8_0Tiles8To(make([]q8_0Tile8, 8), nil, make([]float32, 64*31), 64, 31); err == nil {
		t.Fatal("expected width error")
	}
}
