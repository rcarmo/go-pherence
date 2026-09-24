package mojev

import (
	"fmt"
	"testing"
)

// Benchmark includes validation, LayerNorm, pooling, projection and owned logits.
// It uses synthetic hidden rows; no checkpoint I/O is timed.
func BenchmarkMoJevHead(b *testing.B) {
	const width, rank = 1024, 512
	gamma, beta := make([]float32, width), make([]float32, width)
	context, option := make([]float32, width*rank), make([]float32, width*rank)
	for i := range gamma {
		gamma[i] = 1
	}
	for i := range context {
		context[i] = float32(i%7-3) * 0.001
		option[i] = float32(i%11-5) * 0.001
	}
	head, err := NewHeadWeights(width, rank, gamma, beta, context, option)
	if err != nil {
		b.Fatal(err)
	}
	for _, length := range []int{128, 512} {
		state, questions, candidates, _ := benchmarkMaskSpans(length)
		mask := make([][]bool, len(candidates))
		for f := range mask {
			mask[f] = make([]bool, len(candidates[f]))
			for n := range mask[f] {
				mask[f][n] = true
			}
		}
		hidden := make([]float32, length*width)
		for i := range hidden {
			hidden[i] = float32(i%17-8) * 0.01
		}
		b.Run(fmt.Sprintf("tokens-%d", length), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				got, err := head.ScoreHidden(hidden, state, questions, candidates, mask)
				if err != nil || len(got) != len(questions) {
					b.Fatal(err)
				}
			}
		})
	}
}
