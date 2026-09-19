package omnivoice

import (
	"context"
	"reflect"
	"testing"
)

func TestSharedTraversalParityAndAllocations(t *testing.T) {
	for _, workers := range []int{0, 3} {
		t.Run(string(rune('0'+workers)), func(t *testing.T) {
			b, _, _, _ := makeWorkerFixture(t, 25)
			if workers > 0 {
				if err := b.EnableWorkers(workers); err != nil {
					t.Fatal(err)
				}
			}
			u, err := NewBackboneSibling(b, 19)
			if err != nil {
				t.Fatal(err)
			}
			defer u.Close()
			cfg := DefaultGenerationConfig()
			cfg.ClassTemperature = .3
			cfg.Steps = 16
			g, err := NewGeneration(b, u, 13, cfg)
			if err != nil {
				t.Fatal(err)
			}
			for _, shape := range [][3]int{{25, 19, 13}, {19, 17, 11}, {25, 19, 13}} {
				if err := b.Reconfigure(shape[0]); err != nil {
					t.Fatal(err)
				}
				if err := u.Reconfigure(shape[1]); err != nil {
					t.Fatal(err)
				}
				if err := g.Reconfigure(shape[2]); err != nil {
					t.Fatal(err)
				}
				ids, audio := make([]int, g.books*b.tokens), make([]bool, b.tokens)
				uids, uaudio := make([]int, g.books*u.tokens), make([]bool, u.tokens)
				for i := 1; i < len(audio); i++ {
					audio[i] = true
				}
				for i := 1; i < len(uaudio); i++ {
					uaudio[i] = true
				}
				ids[0] = 2
				uids[0] = 3
				want, got := make([]int, len(g.output)), make([]int, len(g.output))
				g.config.SharedTraversal = false
				if err := g.GenerateInto(context.Background(), want, ids, audio, uids, uaudio); err != nil {
					t.Fatal(err)
				}
				rng := g.rng.Uint64()
				g.config.SharedTraversal = true
				if err := g.GenerateInto(context.Background(), got, ids, audio, uids, uaudio); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(want, got) || rng != g.rng.Uint64() {
					t.Fatal("paired token/RNG mismatch")
				}
				if n := testing.AllocsPerRun(3, func() {
					if err := g.GenerateInto(context.Background(), got, ids, audio, uids, uaudio); err != nil {
						t.Fatal(err)
					}
				}); n != 0 {
					t.Fatalf("allocs %v", n)
				}
				ctx, cancel := context.WithCancel(context.Background())
				cancel()
				if err := g.GenerateInto(ctx, got, ids, audio, uids, uaudio); err == nil {
					t.Fatal("ignored cancellation")
				}
				if b.scratch.executionContext != nil || u.scratch.executionContext != nil {
					t.Fatal("retained execution context")
				}
			}
		})
	}
}

func TestSharedTraversalCompatibility(t *testing.T) {
	b, f := loadBackboneFixture(t)
	u, err := NewBackboneSibling(b, f.Tokens)
	if err != nil {
		t.Fatal(err)
	}
	defer u.Close()
	cfg := DefaultGenerationConfig()
	cfg.SharedTraversal = true
	if _, err := NewGeneration(b, u, 2, cfg); err != nil {
		t.Fatal(err)
	}
	cfg.Guidance = 0
	if _, err := NewGeneration(b, u, 2, cfg); err == nil {
		t.Fatal("accepted no guidance")
	}
	cfg.Guidance = 2
	if _, err := NewGeneration(b, b, 2, cfg); err == nil {
		t.Fatal("accepted same backbone")
	}
	independent, err := NewBackbone(b.weights, f.Tokens)
	if err != nil {
		t.Fatal(err)
	}
	defer independent.Close()
	if _, err := NewGeneration(b, independent, 2, cfg); err == nil {
		t.Fatal("accepted independent arena")
	}
	if err := b.EnableResident(context.Background(), b.ResidentRequiredBytes()); err != nil {
		t.Fatal(err)
	}
	if _, err := NewGeneration(b, u, 2, cfg); err == nil {
		t.Fatal("accepted resident")
	}
}

func TestSharedTraversalExactLogits(t *testing.T) {
	for _, workers := range []int{0, 3} {
		b, _, ids, audio := makeWorkerFixture(t, 25)
		if workers > 0 {
			if err := b.EnableWorkers(workers); err != nil {
				t.Fatal(err)
			}
		}
		u, err := NewBackboneSibling(b, 19)
		if err != nil {
			t.Fatal(err)
		}
		defer u.Close()
		uids, uaudio := make([]int, b.weights.Config.NumAudioCodebook*19), make([]bool, 19)
		for i := range uaudio {
			uaudio[i] = true
		}
		cfg := DefaultGenerationConfig()
		cfg.SharedTraversal = true
		g, err := NewGeneration(b, u, 13, cfg)
		if err != nil {
			t.Fatal(err)
		}
		h := b.weights.Config.LLMConfig.HiddenSize
		cp, up := g.condPrefix[:(25-13)*h], g.uncondPrefix[:(19-13)*h]
		if err := b.embedRangeInto(cp, ids, audio, 0, 12); err != nil {
			t.Fatal(err)
		}
		if err := u.embedRangeInto(up, uids, uaudio, 0, 6); err != nil {
			t.Fatal(err)
		}
		for _, times := range [][]int{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {0, 12}, {12}} {
			clear(g.activeTimes)
			for _, i := range times {
				g.activeTimes[i] = true
			}
			if err := b.forwardInto(context.Background(), g.condTarget, ids, audio, nil, nil, 13, cp, g.activeTimes); err != nil {
				t.Fatal(err)
			}
			if err := u.forwardInto(context.Background(), g.uncondTarget, uids, uaudio, nil, nil, 13, up, g.activeTimes); err != nil {
				t.Fatal(err)
			}
			cw, uw := append([]float32(nil), g.condTarget...), append([]float32(nil), g.uncondTarget...)
			if err := g.forwardPair(context.Background(), ids, audio, uids, uaudio, cp, up); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(cw, g.condTarget) || !reflect.DeepEqual(uw, g.uncondTarget) {
				t.Fatalf("workers=%d times=%v logits differ", workers, times)
			}
		}
	}
}

type cancelPairContext struct {
	context.Context
	checks, after int
}

func (c *cancelPairContext) Err() error {
	c.checks++
	if c.checks >= c.after {
		return context.Canceled
	}
	return nil
}

func TestSharedTraversalMidLayerCancelAndReuse(t *testing.T) {
	b, f := loadBackboneFixture(t)
	u, err := NewBackboneSibling(b, f.Tokens)
	if err != nil {
		t.Fatal(err)
	}
	defer u.Close()
	cfg := DefaultGenerationConfig()
	cfg.SharedTraversal = true
	g, err := NewGeneration(b, u, 2, cfg)
	if err != nil {
		t.Fatal(err)
	}
	ids, uids := append([]int(nil), f.IDs...), append([]int(nil), f.IDs...)
	out := make([]int, len(g.output))
	ctx := &cancelPairContext{Context: context.Background(), after: 6}
	if err := g.GenerateInto(ctx, out, ids, f.AudioMask, uids, f.AudioMask); err != context.Canceled {
		t.Fatalf("cancel error %v", err)
	}
	if b.scratch.executionContext != nil || u.scratch.executionContext != nil {
		t.Fatal("retained execution context")
	}
	if err := g.GenerateInto(context.Background(), out, ids, f.AudioMask, uids, f.AudioMask); err != nil {
		t.Fatal(err)
	}
}
