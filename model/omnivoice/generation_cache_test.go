package omnivoice

import (
	"context"
	"reflect"
	"testing"
)

func TestSparseCachedHeadsExact(t *testing.T) {
	for _, mode := range []string{"streaming", "workers", "resident", "prepacked"} {
		t.Run(mode, func(t *testing.T) {
			b, _, ids, audio := makeWorkerFixture(t, 25)
			if mode != "streaming" {
				if err := b.EnableWorkers(3); err != nil {
					t.Fatal(err)
				}
			}
			if mode == "resident" {
				if err := b.EnableResident(context.Background(), b.ResidentRequiredBytes()); err != nil {
					t.Fatal(err)
				}
			}
			if mode == "prepacked" {
				if err := b.EnableResidentPrepacked(context.Background(), b.ResidentPrepackedRequiredBytes()); err != nil {
					t.Fatal(err)
				}
			}
			positions, mask := targetTestPositions(25), targetTestMask(25)
			const target = 23
			h := b.weights.Config.LLMConfig.HiddenSize
			prefix := make([]float32, 2*h)
			if err := b.embedRangeInto(prefix, ids, audio, 0, 2); err != nil {
				t.Fatal(err)
			}
			want := make([]float32, b.weights.Config.NumAudioCodebook*target*b.weights.Config.AudioVocabSize)
			got := make([]float32, len(want))
			if err := b.ForwardTargetInto(context.Background(), want, ids, audio, positions, mask, target); err != nil {
				t.Fatal(err)
			}
			for _, times := range [][]int{{0}, {4}, {10, 22}, {1, 7, 19, 22}, {}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22}} {
				active := make([]bool, target)
				for _, i := range times {
					active[i] = true
				}
				for i := range got {
					got[i] = -12345
				}
				run := func() {
					if err := b.forwardInto(context.Background(), got, ids, audio, positions, mask, target, prefix, active); err != nil {
						t.Fatal(err)
					}
				}
				run()
				if allocs := testing.AllocsPerRun(3, run); allocs != 0 {
					t.Fatalf("allocs %v", allocs)
				}
				for i, v := range got {
					if active[(i/b.weights.Config.AudioVocabSize)%target] {
						if v != want[i] {
							t.Fatalf("times %v logit %d: %g != %g", times, i, v, want[i])
						}
					} else if v != -12345 {
						t.Fatal("inactive logit overwritten")
					}
				}
			}
			if err := b.forwardInto(context.Background(), got, ids, audio, positions, mask, target, prefix[:1], nil); err == nil {
				t.Fatal("accepted bad prefix")
			}
			if err := b.forwardInto(context.Background(), got, ids, audio, positions, mask, target, prefix, []bool{true}); err == nil {
				t.Fatal("accepted bad active shape")
			}
		})
	}
}

func TestGenerationPrefixRefreshAndReconfigure(t *testing.T) {
	b, _, _, _ := makeWorkerFixture(t, 25)
	u, err := NewBackboneSibling(b, 25)
	if err != nil {
		t.Fatal(err)
	}
	defer u.Close()
	cfg := DefaultGenerationConfig()
	cfg.Steps = 16
	cfg.ClassTemperature = .3
	g, err := NewGeneration(b, u, 23, cfg)
	if err != nil {
		t.Fatal(err)
	}
	for n, shape := range [][2]int{{25, 23}, {19, 13}, {25, 23}} {
		tokens, target := shape[0], shape[1]
		if err := b.Reconfigure(tokens); err != nil {
			t.Fatal(err)
		}
		if err := u.Reconfigure(tokens); err != nil {
			t.Fatal(err)
		}
		if err := g.Reconfigure(target); err != nil {
			t.Fatal(err)
		}
		ids, audio := make([]int, g.books*tokens), make([]bool, tokens)
		for i := 1; i < tokens; i++ {
			audio[i] = true
		}
		// Change prefix text and audio between calls; never reuse stale request data.
		ids[0] = (n + 1) % b.weights.Config.LLMConfig.VocabSize
		audio[1] = true
		for book := 0; book < g.books; book++ {
			ids[book*tokens+1] = (n + 2) % g.vocab
		}
		uids := append([]int(nil), ids...)
		want, got := make([]int, len(g.output)), make([]int, len(g.output))
		if err := g.legacyGenerateInto(context.Background(), want, ids, audio, uids, audio); err != nil {
			t.Fatal(err)
		}
		rngWant := g.rng.Uint64()
		if err := g.GenerateInto(context.Background(), got, ids, audio, uids, audio); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(want, got) || g.rng.Uint64() != rngWant {
			t.Fatalf("request %d mismatch", n)
		}
		if allocs := testing.AllocsPerRun(3, func() {
			if err := g.GenerateInto(context.Background(), got, ids, audio, uids, audio); err != nil {
				t.Fatal(err)
			}
		}); allocs != 0 {
			t.Fatalf("allocs %v", allocs)
		}
	}
}

func TestHeadSpanPruning(t *testing.T) {
	b, _, _, _ := makeWorkerFixture(t, 25)
	active := make([]bool, 23)
	active[0] = true
	active[22] = true
	b.prepareHeadSpans(b.audioHeadProjectStart(23), 2, active)
	rows := 0
	for _, span := range b.projectSpans {
		rows += span.end - span.start
	}
	if len(b.projectSpans) != 2 || rows >= 25 {
		t.Fatalf("no sparse saving: %v", b.projectSpans)
	}
}

// Model-free span accounting makes the skipped work measurable independently
// of wall-clock noise and denoising-dependent mask distribution.
func BenchmarkHeadSpanSelection(b *testing.B) {
	const tokens = 210
	for _, sparse := range []bool{false, true} {
		name := "dense"
		if sparse {
			name = "sparse"
		}
		b.Run(name, func(b *testing.B) {
			bb := &Backbone{tokens: tokens, projectSpans: make([]headSpan, 0, tokens)}
			active := make([]bool, 75)
			for i := range active {
				active[i] = !sparse || i == 74
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				bb.prepareHeadSpans(bb.audioHeadProjectStart(75), 135, active)
			}
			b.StopTimer()
			rows := 0
			for _, s := range bb.projectSpans {
				rows += s.end - s.start
			}
			b.ReportMetric(float64(rows), "projected-rows/op")
		})
	}
}
