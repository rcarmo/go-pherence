package mojev

import (
	"fmt"
	"testing"
)

// Benchmarks include validation and fresh output allocation, as the public APIs do.
func benchmarkMaskSpans(length int) ([]bool, [][]bool, [][][]bool, []bool) {
	state := make([]bool, length)
	questions := make([][]bool, 4)
	candidates := make([][][]bool, 4)
	packed := make([]bool, length)
	cursor := length / 4
	for i := range state[:cursor] {
		state[i] = true
	}
	for f := range questions {
		questions[f] = make([]bool, length)
		candidates[f] = make([][]bool, 4)
		for pos := cursor; pos < cursor+4; pos++ {
			questions[f][pos] = true
		}
		cursor += 4
		for n := range candidates[f] {
			c := make([]bool, length)
			for pos := cursor; pos < cursor+4; pos++ {
				c[pos] = true
			}
			cursor += 4
			candidates[f][n] = c
		}
	}
	for i := range packed[:cursor] {
		packed[i] = true
	}
	return state, questions, candidates, packed
}

func BenchmarkMoJevMask(b *testing.B) {
	for _, length := range []int{128, 512, 1024, 2048} {
		state, questions, candidates, packed := benchmarkMaskSpans(length)
		b.Run(fmt.Sprintf("tree/%d", length), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				v, err := TreeMask(state, questions, candidates)
				if err != nil || len(v) != length {
					b.Fatal(err)
				}
			}
		})
		b.Run(fmt.Sprintf("additive/%d", length), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				v, err := AdditiveTreeMask(state, questions, candidates, packed)
				if err != nil || len(v) != length {
					b.Fatal(err)
				}
			}
		})
	}
}
