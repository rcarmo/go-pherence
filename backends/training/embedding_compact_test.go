package training

import (
	"math"
	"reflect"
	"sync"
	"testing"
)

func TestTrainingCompactEmbeddingIndex(t *testing.T) {
	ids := []float32{2, 0, 2, 3, 0}
	saved := append([]float32(nil), ids...)
	index, err := BuildCompactEmbeddingIndex(ids, 65536)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{0, 2, 3, -1, -1, 0, 2, 4, 5, 5, 5, 1, 4, 0, 2, 3}
	if !reflect.DeepEqual(index, want) || !reflect.DeepEqual(ids, saved) {
		t.Fatal("index/source", index)
	}
	if err := ValidateCompactEmbeddingIndex(index, len(ids), 65536); err != nil {
		t.Fatal(err)
	}
	other, _ := BuildCompactEmbeddingIndex(ids, 65536)
	index[0] = -1
	if !reflect.DeepEqual(other, want) {
		t.Fatal("crosscall alias")
	}
	long := make([]float32, 1024)
	for i := range long {
		long[i] = float32(i)
	}
	for _, ids := range [][]float32{{65535}, {1, 1, 1}, {0, 1, 2}, long} {
		idx, err := BuildCompactEmbeddingIndex(ids, 65536)
		if err != nil {
			t.Fatal(err)
		}
		if err := ValidateCompactEmbeddingIndex(idx, len(ids), 65536); err != nil {
			t.Fatal(err)
		}
	}
}
func TestTrainingCompactEmbeddingRejects(t *testing.T) {
	for _, ids := range [][]float32{nil, make([]float32, 1025), {-1}, {4}, {.5}, {float32(math.NaN())}, {float32(math.Inf(1))}} {
		if _, err := BuildCompactEmbeddingIndex(ids, 4); err == nil {
			t.Fatal("bad IDs", ids)
		}
	}
	if _, err := BuildCompactEmbeddingIndex([]float32{0}, 0); err == nil {
		t.Fatal("vocabulary")
	}
	base, _ := BuildCompactEmbeddingIndex([]float32{2, 0, 2, 3, 0}, 5)
	for _, change := range []func([]float32) []float32{
		func(x []float32) []float32 { return x[:len(x)-1] },
		func(x []float32) []float32 { x[5] = 1; return x },
		func(x []float32) []float32 { x[10] = 4; return x },
		func(x []float32) []float32 { x[6] = 3; x[7] = 2; return x },
		func(x []float32) []float32 { x[6] = .5; return x },
		func(x []float32) []float32 { x[6] = float32(math.NaN()); return x },
		func(x []float32) []float32 { x[6] = float32(math.Inf(1)); return x },
		func(x []float32) []float32 { x[6] = -1; return x },
		func(x []float32) []float32 { x[6] = 6; return x },
		func(x []float32) []float32 { x[0] = .5; return x },
		func(x []float32) []float32 { x[0] = float32(math.NaN()); return x },
		func(x []float32) []float32 { x[0] = 5; return x },
		func(x []float32) []float32 { x[1] = 0; return x },
		func(x []float32) []float32 { x[0] = -1; return x },
		func(x []float32) []float32 { x[3] = 4; return x },
		func(x []float32) []float32 { x[11] = 5; return x },
		func(x []float32) []float32 { x[11] = .5; return x },
		func(x []float32) []float32 { x[11] = 4; x[12] = 1; return x },
		func(x []float32) []float32 { x[13] = 1; return x },
	} {
		if err := ValidateCompactEmbeddingIndex(change(append([]float32(nil), base...)), 5, 5); err == nil {
			t.Fatal("invalid compact index admitted")
		}
	}
	if err := ValidateCompactEmbeddingIndex(base, 0, 5); err == nil {
		t.Fatal("rows")
	}
	// Ordered nonempty active bucket is mandatory even before padding.
	empty := append([]float32(nil), base...)
	empty[6] = 0
	if ValidateCompactEmbeddingIndex(empty, 5, 5) == nil {
		t.Fatal("empty active bucket")
	}
}
func TestTrainingCompactEmbeddingTape(t *testing.T) {
	index, _ := BuildCompactEmbeddingIndex([]float32{2, 0, 2}, 65536)
	makeTape := func() Tape {
		return Tape{Tensors: []TensorSpec{{Shape: []int{6}, Values: make([]float32, 6)}, {Shape: []int{10}, Values: append([]float32(nil), index...)}, {Shape: []int{6}}}, Commands: []Command{{Op: "embedding", Inputs: []int{0, 1}, Output: 2, Rows: 3, Width: 2, Parts: 65536, Mode: 3}}, Outputs: []int{2}}
	}
	if err := makeTape().Validate(88); err != nil {
		t.Fatal(err)
	}
	if makeTape().Validate(87) == nil {
		t.Fatal("index budget")
	}
	for _, mutate := range []func(*Tape){func(x *Tape) { x.Tensors[1].Values = nil }, func(x *Tape) { x.Tensors[1].Values[9] = 0 }, func(x *Tape) { x.Tensors[2].Shape = []int{131072} }, func(x *Tape) { x.Commands[0].Mode = 4 }, func(x *Tape) {
		x.Tensors = append(x.Tensors, TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
	}} {
		x := makeTape()
		mutate(&x)
		if x.Validate(1<<20) == nil {
			t.Fatal("bad compact source/tape")
		}
	}
}
func TestTrainingCompactEmbeddingConcurrent(t *testing.T) {
	ids := []float32{2, 0, 2, 3, 0}
	want, _ := BuildCompactEmbeddingIndex(ids, 65536)
	var wg sync.WaitGroup
	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 20; i++ {
				idx, err := BuildCompactEmbeddingIndex(ids, 65536)
				if err != nil || !reflect.DeepEqual(idx, want) {
					t.Error(err)
					return
				}
				idx[0] = -1
			}
		}()
	}
	wg.Wait()
	if !reflect.DeepEqual(ids, []float32{2, 0, 2, 3, 0}) {
		t.Fatal("source writes")
	}
}
func TestTrainingCompactEmbeddingVJP(t *testing.T) {
	long := make([]float32, 1024)
	for i := range long {
		long[i] = float32(i % 7)
	}
	maxError, total, squares := float64(0), float64(0), float64(0)
	count := 0
	for _, ids := range [][]float32{{0}, {2, 0, 2, 3, 0}, {1, 1, 1, 1}, long} {
		const width, vocabulary = 3, 8
		index, err := BuildCompactEmbeddingIndex(ids, vocabulary)
		if err != nil {
			t.Fatal(err)
		}
		rows := len(ids)
		x := make([]float32, rows*width)
		for i := range x {
			x[i] = float32(i%13-6) * .00137
		}
		ref := make([]float64, vocabulary*width)
		for row, id := range ids {
			for col := 0; col < width; col++ {
				ref[int(id)*width+col] += float64(x[row*width+col])
			}
		}
		seen := make([]bool, vocabulary)
		for slot := 0; slot < rows; slot++ {
			token := int(index[slot])
			for col := 0; col < width; col++ {
				var sum float32
				if token >= 0 {
					seen[token] = true
					for at := int(index[rows+slot]); at < int(index[rows+slot+1]); at++ {
						row := int(index[2*rows+1+at])
						sum += x[row*width+col]
					}
				}
				expected := float64(0)
				if token >= 0 {
					expected = ref[token*width+col]
				}
				e := math.Abs(float64(sum) - expected)
				maxError = math.Max(maxError, e)
				total += e
				squares += e * e
				count++
				if e > 2e-6*math.Max(1, math.Abs(expected)) {
					t.Fatal("compact VJP", slot, col, sum, expected)
				}
			}
		}
		for token, present := range seen {
			if !present {
				for col := 0; col < width; col++ {
					if ref[token*width+col] != 0 {
						t.Fatal("dropped token")
					}
				}
			}
		}
	}
	t.Logf("compact F32/F64 n=%d mean=%.9g rms=%.9g max=%.9g; existing 2e-6 gate unchanged", count, total/float64(count), math.Sqrt(squares/float64(count)), maxError)
	// Sequential F32 order is the contract for a cancellation-sensitive bucket.
	index, _ := BuildCompactEmbeddingIndex([]float32{1, 0, 1, 1}, 2)
	x := []float32{1e8, 7, 1, -1e8}
	var got, want float32
	for at := int(index[5]); at < int(index[6]); at++ {
		got += x[int(index[9+at])]
	}
	for _, row := range []int{0, 2, 3} {
		want += x[row]
	}
	if got != want {
		t.Fatal("sequential reduction order")
	}
}

var compactEmbeddingSink []float32

func BenchmarkTrainingCompactEmbeddingIndex(b *testing.B) {
	ids := make([]float32, 512)
	for i := range ids {
		ids[i] = float32((i * 17) % 1024)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		index, err := BuildCompactEmbeddingIndex(ids, 32768)
		if err != nil {
			b.Fatal(err)
		}
		compactEmbeddingSink = index
	}
}
