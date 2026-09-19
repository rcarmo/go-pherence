package omnivoice

import (
	"context"
	"fmt"
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
	"math"
	"os"
	"testing"
)

// Test-only experiment; production chunk size remains fixed until measured.
func setTestHeadChunk(b *Backbone, chunk int) {
	b.chunk = chunk
	b.head = make([]float32, chunk*b.weights.Config.LLMConfig.HiddenSize)
	b.headOutput = make([]float32, b.tokens*chunk)
}
func TestRealHeadChunkParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_REAL_OMNIVOICE")
	if path == "" {
		t.Skip("set GO_PHERENCE_REAL_OMNIVOICE")
	}
	w, err := loader.OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	b, err := NewBackbone(w, 85)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	original := make([]float32, len(b.hidden))
	for i := range original {
		original[i] = float32(math.Sin(float64(i) * .1))
	}
	const target = 75
	want, got := make([]float32, w.Config.NumAudioCodebook*target*w.Config.AudioVocabSize), make([]float32, w.Config.NumAudioCodebook*target*w.Config.AudioVocabSize)
	active := make([]bool, target)
	for i := range active {
		active[i] = i%3 == 0 || i == target-1
	}
	for _, mask := range [][]bool{nil, active} {
		for _, columns := range []bool{false, true} {
			b.columnWorkers = columns
			if err := b.EnableWorkers(2); err != nil {
				t.Fatal(err)
			}
			b.bindExecution(context.Background())
			setTestHeadChunk(b, 128)
			copy(b.hidden, original)
			if err := b.projectForward(context.Background(), want, target, mask); err != nil {
				t.Fatal(err)
			}
			for _, chunk := range []int{256, 512, 1024} {
				setTestHeadChunk(b, chunk)
				copy(b.hidden, original)
				if err := b.projectForward(context.Background(), got, target, mask); err != nil {
					t.Fatal(err)
				}
				assertFloat32Exact(t, got, want)
				if n := testing.AllocsPerRun(3, func() {
					copy(b.hidden, original)
					if err := b.projectForward(context.Background(), got, target, mask); err != nil {
						t.Fatal(err)
					}
				}); n != 0 {
					t.Fatalf("allocs %v", n)
				}
			}
			b.clearExecution()
		}
	}
}
func BenchmarkRealHeadChunk(b *testing.B) {
	path := os.Getenv("GO_PHERENCE_REAL_OMNIVOICE")
	if path == "" {
		b.Skip("set GO_PHERENCE_REAL_OMNIVOICE")
	}
	w, err := loader.OpenWeights(path)
	if err != nil {
		b.Fatal(err)
	}
	defer w.Close()
	for _, chunk := range []int{128, 256, 512, 1024} {
		b.Run(fmt.Sprint(chunk), func(b *testing.B) {
			bb, err := NewBackbone(w, 85)
			if err != nil {
				b.Fatal(err)
			}
			defer bb.Close()
			if err := bb.EnableColumnWorkers(2); err != nil {
				b.Fatal(err)
			}
			setTestHeadChunk(bb, chunk)
			bb.bindExecution(context.Background())
			defer bb.clearExecution()
			original := make([]float32, len(bb.hidden))
			for i := range original {
				original[i] = float32(math.Sin(float64(i) * .1))
			}
			out := make([]float32, w.Config.NumAudioCodebook*75*w.Config.AudioVocabSize)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				copy(bb.hidden, original)
				if err := bb.projectForward(context.Background(), out, 75, nil); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
