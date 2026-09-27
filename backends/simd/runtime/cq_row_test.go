package simd

import (
	"math"
	"math/rand"
	"sync"
	"testing"
)

func TestCQMatrixDecodeRowMatchesDenseOracle(t *testing.T) {
	for _, bits := range []int{1, 2, 3, 4, 5} {
		for _, shape := range []struct{ rows, cols int }{{1, 1}, {2, 17}, {3, 127}, {4, 128}, {5, 129}, {6, 255}, {7, 257}} {
			rng := rand.New(rand.NewSource(int64(bits*10000 + shape.rows*100 + shape.cols)))
			book := testCQCodebook()
			blob := makeRandomCQBlob(shape.rows, shape.cols, bits, rng)
			m, err := NewCQMatrix(shape.rows, shape.cols, bits, blob, book)
			if err != nil {
				t.Fatal(err)
			}
			want := decodeCQDenseOracle(shape.rows, shape.cols, bits, blob, book)
			clear(blob)
			clear(book) // constructor owns original payload and codebook
			for row := 0; row < shape.rows; row++ {
				got, err := m.DecodeRow(row)
				if err != nil || len(got) != shape.cols {
					t.Fatalf("bits=%d shape=%+v row=%d length=%d err=%v", bits, shape, row, len(got), err)
				}
				for col, v := range got {
					ref := want[row*shape.cols+col]
					if math.Float32bits(v) != math.Float32bits(ref) {
						t.Fatalf("bits=%d shape=%+v row=%d col=%d got=%g want=%g", bits, shape, row, col, v, ref)
					}
				}
				if row == 0 {
					clear(got)
				}
			}
			again, err := m.DecodeRow(0)
			if err != nil {
				t.Fatal(err)
			}
			for col, v := range again {
				if math.Float32bits(v) != math.Float32bits(want[col]) {
					t.Fatal("returned row aliases immutable CQ storage")
				}
			}
		}
	}
}

func TestCQMatrixDecodeRowBoundsAndConcurrentOwnership(t *testing.T) {
	if _, err := (*CQMatrix)(nil).DecodeRow(0); err == nil {
		t.Fatal("nil matrix accepted")
	}
	m, err := NewCQMatrix(4, 129, 4, makeRandomCQBlob(4, 129, 4, rand.New(rand.NewSource(42))), testCQCodebook())
	if err != nil {
		t.Fatal(err)
	}
	for _, row := range []int{-1, 4, int(^uint(0) >> 1)} {
		if got, err := m.DecodeRow(row); err == nil || got != nil {
			t.Fatalf("invalid row=%d got=%v err=%v", row, got, err)
		}
	}
	ref := make([][]float32, 4)
	for row := range ref {
		ref[row], err = m.DecodeRow(row)
		if err != nil {
			t.Fatal(err)
		}
	}
	var wg sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 25; i++ {
				row := i % 4
				got, e := m.DecodeRow(row)
				if e != nil || len(got) != 129 {
					t.Errorf("concurrent row=%d err=%v", row, e)
					return
				}
				for j, v := range got {
					if math.Float32bits(v) != math.Float32bits(ref[row][j]) {
						t.Errorf("concurrent row=%d col=%d", row, j)
						return
					}
				}
				clear(got)
			}
		}()
	}
	wg.Wait()
}
