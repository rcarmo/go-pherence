package omnivoice

import (
	"context"
	"math"
	"strconv"
	"testing"
)

func suffixLogitsFromFull(full []float32, books, tokens, vocab, target int) []float32 {
	out := make([]float32, books*target*vocab)
	start := tokens - target
	for book := 0; book < books; book++ {
		for t := 0; t < target; t++ {
			copy(out[(book*target+t)*vocab:(book*target+t+1)*vocab], full[(book*tokens+start+t)*vocab:(book*tokens+start+t+1)*vocab])
		}
	}
	return out
}

func targetTestPositions(tokens int) []int {
	positions := make([]int, tokens)
	for i := range positions {
		positions[i] = 11 + i*7
	}
	return positions
}

func targetTestMask(tokens int) []float32 {
	mask := make([]float32, tokens*tokens)
	negInf := float32(math.Inf(-1))
	for i := 0; i < tokens; i++ {
		mask[i*tokens+(i+1)%tokens] = negInf
		mask[i*tokens+i] = 0
	}
	return mask
}

func TestBackboneForwardTargetIntoRealFixtureExactSuffix(t *testing.T) {
	b, f := loadBackboneFixture(t)
	books := len(f.IDs) / f.Tokens
	vocab := b.weights.Config.AudioVocabSize
	positions := targetTestPositions(f.Tokens)
	mask := targetTestMask(f.Tokens)
	full := make([]float32, len(f.Logits))
	if err := b.ForwardInto(context.Background(), full, f.IDs, f.AudioMask, positions, mask); err != nil {
		t.Fatal(err)
	}
	for _, target := range []int{1, 2, f.Tokens} {
		t.Run("target="+strconv.Itoa(target), func(t *testing.T) {
			got := make([]float32, books*target*vocab)
			if err := b.ForwardTargetInto(context.Background(), got, f.IDs, f.AudioMask, positions, mask, target); err != nil {
				t.Fatal(err)
			}
			assertFloat32Exact(t, got, suffixLogitsFromFull(full, books, f.Tokens, vocab, target))
		})
	}
}

func TestBackboneForwardTargetIntoExactSuffixAcrossModes(t *testing.T) {
	modes := []struct {
		name  string
		setup func(*testing.T, *Backbone)
	}{
		{name: "streaming"},
		{name: "workers", setup: func(t *testing.T, b *Backbone) {
			t.Helper()
			if err := b.EnableWorkers(3); err != nil {
				t.Fatal(err)
			}
		}},
		{name: "resident-workers", setup: func(t *testing.T, b *Backbone) {
			t.Helper()
			if err := b.EnableResident(context.Background(), b.ResidentRequiredBytes()); err != nil {
				t.Fatal(err)
			}
			if err := b.EnableWorkers(3); err != nil {
				t.Fatal(err)
			}
		}},
		{name: "resident-prepacked-workers", setup: func(t *testing.T, b *Backbone) {
			t.Helper()
			if err := b.EnableResidentPrepacked(context.Background(), b.ResidentPrepackedRequiredBytes()); err != nil {
				t.Fatal(err)
			}
			if err := b.EnableWorkers(3); err != nil {
				t.Fatal(err)
			}
		}},
	}
	for _, mode := range modes {
		t.Run(mode.name, func(t *testing.T) {
			b, f, ids, audio := makeWorkerFixture(t, 13)
			if mode.setup != nil {
				mode.setup(t, b)
			}
			positions := targetTestPositions(len(audio))
			mask := targetTestMask(len(audio))
			full := make([]float32, workerLogitsLen(f, len(audio)))
			if err := b.ForwardInto(context.Background(), full, ids, audio, positions, mask); err != nil {
				t.Fatal(err)
			}
			books := f.config.NumAudioCodebook
			vocab := f.config.AudioVocabSize
			for _, target := range []int{1, 2, 3, 5, 7, len(audio)} {
				t.Run("target="+strconv.Itoa(target), func(t *testing.T) {
					got := make([]float32, books*target*vocab)
					if err := b.ForwardTargetInto(context.Background(), got, ids, audio, positions, mask, target); err != nil {
						t.Fatal(err)
					}
					assertFloat32Exact(t, got, suffixLogitsFromFull(full, books, len(audio), vocab, target))
				})
			}
		})
	}
}

func TestBackboneForwardTargetIntoReconfigureValidationAndZeroAllocs(t *testing.T) {
	b, f, ids, audio := makeWorkerFixture(t, 13)
	if err := b.EnableResidentPrepacked(context.Background(), b.ResidentPrepackedRequiredBytes()); err != nil {
		t.Fatal(err)
	}
	if err := b.EnableWorkers(3); err != nil {
		t.Fatal(err)
	}
	if err := b.Reconfigure(9); err != nil {
		t.Fatal(err)
	}
	books := f.config.NumAudioCodebook
	vocab := f.config.AudioVocabSize
	ids9 := sliceBookMajor(ids, books, 13, 9)
	audio9 := append([]bool(nil), audio[:9]...)
	positions9 := targetTestPositions(9)
	mask9 := targetTestMask(9)
	full9 := make([]float32, books*9*vocab)
	if err := b.ForwardInto(context.Background(), full9, ids9, audio9, positions9, mask9); err != nil {
		t.Fatal(err)
	}
	got := make([]float32, books*5*vocab)
	if err := b.ForwardTargetInto(context.Background(), got, ids9, audio9, positions9, mask9, 5); err != nil {
		t.Fatal(err)
	}
	assertFloat32Exact(t, got, suffixLogitsFromFull(full9, books, 9, vocab, 5))
	var err error
	if n := testing.AllocsPerRun(10, func() {
		err = b.ForwardTargetInto(context.Background(), got, ids9, audio9, positions9, mask9, 5)
	}); err != nil {
		t.Fatal(err)
	} else if n != 0 {
		t.Fatalf("target worker allocs %g", n)
	}
	if err := b.ForwardTargetInto(context.Background(), got, ids9, audio9, positions9, mask9, 0); err == nil {
		t.Fatal("zero target accepted")
	}
	if err := b.ForwardTargetInto(context.Background(), got, ids9, audio9, positions9, mask9, 10); err == nil {
		t.Fatal("target above active tokens accepted")
	}
	if err := b.ForwardTargetInto(context.Background(), got[:len(got)-1], ids9, audio9, positions9, mask9, 5); err == nil {
		t.Fatal("bad target output shape accepted")
	}
}
