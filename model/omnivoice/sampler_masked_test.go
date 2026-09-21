package omnivoice

import (
	"fmt"
	"math"
	"testing"
)

func TestMaskedSamplerExact(t *testing.T) {
	const books, target, vocab, maskID = 3, 7, 19, 18
	w, err := NewSamplerWorkspace(books, target, vocab)
	if err != nil {
		t.Fatal(err)
	}
	rows := books * target
	size := rows * vocab
	cond, uncond, noise := make([]float32, size), make([]float32, size), make([]float32, size)
	for i := range cond {
		cond[i] = float32(math.Sin(float64(i)))
		uncond[i] = float32(math.Cos(float64(i) * .3))
		noise[i] = float32(i%97+1) / 99
	}
	tokens := make([]int, rows)
	for i := range tokens {
		if i%3 == 0 {
			tokens[i] = maskID
		}
	}
	for _, guidance := range []float32{0, 2} {
		for _, temp := range []float32{0, .3} {
			want, got := make([]float32, size), make([]float32, size)
			if err := w.GuidedLogProbsInto(want, cond, uncond, target, guidance, maskID); err != nil {
				t.Fatal(err)
			}
			for i := range got {
				got[i] = -12345
			}
			if err := w.guidedLogProbsInto(got, cond, uncond, target, guidance, maskID, tokens); err != nil {
				t.Fatal(err)
			}
			for i := range got {
				if tokens[i/vocab] == maskID {
					if got[i] != want[i] {
						t.Fatalf("logit %d mismatch", i)
					}
				} else if got[i] != -12345 {
					t.Fatal("revealed logit changed")
				}
			}
			wp, gp := make([]int, rows), make([]int, rows)
			wc, gc := make([]float32, rows), make([]float32, rows)
			for i := range gp {
				gp[i] = -1
				gc[i] = -12345
			}
			if err := w.PredictTokensWithConfidenceInto(wp, wc, want, target, temp, .1, GumbelNoise{Uniforms: noise}); err != nil {
				t.Fatal(err)
			}
			run := func() {
				if err := w.guidedLogProbsInto(got, cond, uncond, target, guidance, maskID, tokens); err != nil {
					t.Fatal(err)
				}
				if err := w.predictTokensWithConfidenceInto(gp, gc, got, target, temp, .1, GumbelNoise{Uniforms: noise}, tokens, maskID); err != nil {
					t.Fatal(err)
				}
			}
			run()
			for i := range gp {
				if tokens[i] == maskID {
					if gp[i] != wp[i] || gc[i] != wc[i] {
						t.Fatalf("prediction %d mismatch", i)
					}
				} else if gp[i] != -1 || gc[i] != -12345 {
					t.Fatal("revealed prediction changed")
				}
			}
			if n := testing.AllocsPerRun(3, run); n != 0 {
				t.Fatalf("allocs %v", n)
			}
			if err := w.guidedLogProbsInto(got, cond, uncond, target, guidance, maskID, tokens[:1]); err == nil {
				t.Fatal("bad token shape")
			}
			if err := w.predictTokensWithConfidenceInto(gp, gc, got, target, temp, .1, GumbelNoise{Uniforms: noise}, tokens[:1], maskID); err == nil {
				t.Fatal("bad prediction token shape")
			}
		}
	}
}

func BenchmarkMaskedSampler(b *testing.B) {
	const books, target, vocab = 8, 75, 1025
	for _, fraction := range []int{1, 2, 4, 8} {
		for _, sparse := range []bool{false, true} {
			b.Run(fmt.Sprintf("one-in-%d/sparse%v", fraction, sparse), func(b *testing.B) {
				w, err := NewSamplerWorkspace(books, target, vocab)
				if err != nil {
					b.Fatal(err)
				}
				rows, size := books*target, books*target*vocab
				tokens := make([]int, rows)
				for i := range tokens {
					if i%fraction == 0 {
						tokens[i] = vocab - 1
					}
				}
				cond, uncond, lp := make([]float32, size), make([]float32, size), make([]float32, size)
				for i := range cond {
					cond[i] = float32(i%97) * .01
					uncond[i] = float32(i%31) * .02
				}
				pred, confidence := make([]int, rows), make([]float32, rows)
				if !sparse {
					tokens = nil
				}
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					if err := w.guidedLogProbsInto(lp, cond, uncond, target, 2, vocab-1, tokens); err != nil {
						b.Fatal(err)
					}
					if err := w.predictTokensWithConfidenceInto(pred, confidence, lp, target, 0, .1, GumbelNoise{}, tokens, vocab-1); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

func TestSelectionExcludesRevealedNegativeInfinityTies(t *testing.T) {
	w, err := NewSamplerWorkspace(1, 4, 5)
	if err != nil {
		t.Fatal(err)
	}
	tokens := []int{1, 4, 2, 4}
	pred := []int{3, 0, 3, 1}
	confidence := []float32{123, float32(math.Inf(-1)), 456, float32(math.Inf(-1))}
	selected, err := w.ApplyConfidenceSelection(tokens, pred, confidence, 4, 4, 2, 0, 0, GumbelNoise{})
	if err != nil {
		t.Fatal(err)
	}
	if len(selected) != 2 || selected[0] != 1 || selected[1] != 3 {
		t.Fatalf("selected %v", selected)
	}
	want := []int{1, 0, 2, 1}
	for i := range tokens {
		if tokens[i] != want[i] {
			t.Fatalf("tokens %v", tokens)
		}
	}
}
