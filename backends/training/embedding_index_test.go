package training

import (
	"math"
	"reflect"
	"sync"
	"testing"
)

func TestTrainingEmbeddingIndex(t *testing.T) {
	ids := []float32{2, 0, 2, 3, 0}
	saved := append([]float32(nil), ids...)
	index, err := BuildEmbeddingIndex(ids, 5)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{0, 2, 2, 4, 5, 5, 1, 4, 0, 2, 3}
	if !reflect.DeepEqual(index, want) || !reflect.DeepEqual(ids, saved) {
		t.Fatal("unstable index/source", index)
	}
	if err := ValidateEmbeddingIndex(index, len(ids), 5); err != nil {
		t.Fatal(err)
	}
	other, _ := BuildEmbeddingIndex(ids, 5)
	index[0] = -1
	if !reflect.DeepEqual(other, want) || !reflect.DeepEqual(ids, saved) {
		t.Fatal("borrowed index")
	}
	for _, ids := range [][]float32{{0}, {1, 1, 1}, {0, 3}, {65535}} {
		v := 4
		if ids[0] == 65535 {
			v = 65536
		}
		idx, err := BuildEmbeddingIndex(ids, v)
		if err != nil {
			t.Fatal(err)
		}
		if err := ValidateEmbeddingIndex(idx, len(ids), v); err != nil {
			t.Fatal(err)
		}
	}
}
func TestTrainingEmbeddingIndexRejects(t *testing.T) {
	for _, ids := range [][]float32{nil, make([]float32, 1025), {-1}, {4}, {.5}, {float32(math.NaN())}, {float32(math.Inf(1))}} {
		if _, err := BuildEmbeddingIndex(ids, 4); err == nil {
			t.Fatal("bad IDs", ids)
		}
	}
	if _, err := BuildEmbeddingIndex([]float32{0}, 0); err == nil {
		t.Fatal("vocabulary")
	}
	base, _ := BuildEmbeddingIndex([]float32{2, 0, 2, 3, 0}, 5)
	for _, change := range []func([]float32) []float32{
		func(x []float32) []float32 { return x[:len(x)-1] },
		func(x []float32) []float32 { x[0] = 1; return x },
		func(x []float32) []float32 { x[5] = 4; return x },
		func(x []float32) []float32 { x[1] = 3; x[2] = 2; return x },
		func(x []float32) []float32 { x[1] = .5; return x },
		func(x []float32) []float32 { x[1] = float32(math.NaN()); return x },
		func(x []float32) []float32 { x[1] = float32(math.Inf(1)); return x },
		func(x []float32) []float32 { x[1] = -1; return x },
		func(x []float32) []float32 { x[1] = 6; return x },
		func(x []float32) []float32 { x[6] = 5; return x },
		func(x []float32) []float32 { x[6] = .5; return x },
		func(x []float32) []float32 { x[6] = float32(math.NaN()); return x },
		func(x []float32) []float32 { x[6] = 4; x[7] = 1; return x },
		func(x []float32) []float32 { x[8] = 1; return x },
	} {
		if err := ValidateEmbeddingIndex(change(append([]float32(nil), base...)), 5, 5); err == nil {
			t.Fatal("invalid index admitted")
		}
	}
	if err := ValidateEmbeddingIndex(base, 0, 5); err == nil {
		t.Fatal("rows")
	}
}
func TestTrainingEmbeddingIndexTape(t *testing.T) {
	index, _ := BuildEmbeddingIndex([]float32{2, 0, 2}, 4)
	makeTape := func() Tape {
		return Tape{Tensors: []TensorSpec{{Shape: []int{6}, Values: make([]float32, 6)}, {Shape: []int{8}, Values: append([]float32(nil), index...)}, {Shape: []int{8}}}, Commands: []Command{{Op: "embedding", Inputs: []int{0, 1}, Output: 2, Rows: 3, Width: 2, Parts: 4, Mode: 2}}, Outputs: []int{2}}
	}
	if err := makeTape().Validate(1024); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*Tape){
		func(x *Tape) { x.Tensors[1].Values = nil },
		func(x *Tape) { x.Tensors[1].Values[7] = 0 },
		func(x *Tape) { x.Tensors[1].Shape = []int{4} },
		func(x *Tape) { x.Commands[0].Mode = 3 },
		func(x *Tape) {
			x.Tensors = append(x.Tensors, TensorSpec{Shape: []int{1}, Values: []float32{float32(math.NaN())}})
		},
	} {
		x := makeTape()
		mutate(&x)
		if x.Validate(1024) == nil {
			t.Fatal("malformed/dead source index admission")
		}
	}
	if makeTape().Validate(87) == nil {
		t.Fatal("complete index budget")
	}
}
func TestTrainingEmbeddingIndexConcurrent(t *testing.T) {
	ids := []float32{2, 0, 2, 3, 0}
	want, _ := BuildEmbeddingIndex(ids, 5)
	var wg sync.WaitGroup
	for w := 0; w < 4; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 20; i++ {
				idx, err := BuildEmbeddingIndex(ids, 5)
				if err != nil || !reflect.DeepEqual(idx, want) {
					t.Error("shared index", err)
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

// Independent row-scatter reference versus the gather-by-index kernel formula.
// F64 comparisons allow F32 reduction rounding; no hash/signal equality gate.
func TestTrainingEmbeddingIndexVJP(t *testing.T) {
	longIDs := make([]float32, 1024)
	for i := range longIDs {
		longIDs[i] = float32((i * 7) % 8)
	}
	maxError, totalError, squaredError := float64(0), float64(0), float64(0)
	comparisons := 0
	for _, ids := range [][]float32{{0}, {2, 0, 2, 3, 0}, {1, 1, 1, 1}, {0, 7, 0, 7, 3, 1}, longIDs} {
		const width, vocabulary = 3, 8
		index, err := BuildEmbeddingIndex(ids, vocabulary)
		if err != nil {
			t.Fatal(err)
		}
		x := make([]float32, len(ids)*width)
		for i := range x {
			x[i] = float32((i*7)%13-6) * .00137
		}
		ref := make([]float64, vocabulary*width)
		for row, id := range ids {
			for col := 0; col < width; col++ {
				ref[int(id)*width+col] += float64(x[row*width+col])
			}
		}
		adds := 0
		for token := 0; token < vocabulary; token++ {
			for col := 0; col < width; col++ {
				var sum float32
				for at := int(index[token]); at < int(index[token+1]); at++ {
					row := int(index[vocabulary+1+at])
					sum += x[row*width+col]
					adds++
				}
				r := ref[token*width+col]
				e := math.Abs(float64(sum) - r)
				maxError = math.Max(maxError, e)
				totalError += e
				squaredError += e * e
				comparisons++
				if math.Abs(float64(sum)-r) > 2e-6*math.Max(1, math.Abs(r)) {
					t.Fatal("VJP", token, col, sum, r)
				}
			}
		}
		if adds != len(ids)*width {
			t.Fatal("indexed work", adds)
		}
	}
	t.Logf("indexed F32/F64 error: n=%d mean=%.9g rms=%.9g max=%.9g over fixed small/mixed/max-row cases; existing 2e-6 gate unchanged", comparisons, totalError/float64(comparisons), math.Sqrt(squaredError/float64(comparisons)), maxError)
	// Cancellation-sensitive rows must retain sequence order, not token sorting
	// within a bucket or a tree sum. F32 expected is derived by sequential adds.
	index, _ := BuildEmbeddingIndex([]float32{1, 0, 1, 1}, 2)
	x := []float32{1e8, 7, 1, -1e8}
	sum := float32(0)
	for at := int(index[1]); at < int(index[2]); at++ {
		sum += x[int(index[3+at])]
	}
	var sequential float32
	for _, row := range []int{0, 2, 3} {
		sequential += x[row]
	}
	if sum != sequential {
		t.Fatal("sequential rounding contract")
	}
}

var embeddingIndexSink []float32

func BenchmarkTrainingEmbeddingIndex(b *testing.B) {
	ids := make([]float32, 512)
	for i := range ids {
		ids[i] = float32((i * 17) % 1024)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx, err := BuildEmbeddingIndex(ids, 32768)
		if err != nil {
			b.Fatal(err)
		}
		embeddingIndexSink = idx
	}
}
