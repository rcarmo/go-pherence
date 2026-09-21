package omnivoice

import (
	"context"
	"reflect"
	"testing"
)

func TestGuidanceScaleParityAndAllocation(t *testing.T) {
	for _, scale := range []float32{0, .5, 1, 1.5, 2} {
		b, f := loadBackboneFixture(t)
		u, err := NewBackboneSibling(b, f.Tokens)
		if err != nil {
			t.Fatal(err)
		}
		defer u.Close()
		cfg := DefaultGenerationConfig()
		cfg.Steps = 8
		cfg.Guidance = scale
		cfg.ClassTemperature = .3
		g, err := NewGeneration(b, u, 2, cfg)
		if err != nil {
			t.Fatal(err)
		}
		ids, uids := append([]int(nil), f.IDs...), append([]int(nil), f.IDs...)
		want, got := make([]int, len(g.output)), make([]int, len(g.output))
		if err := g.legacyGenerateInto(context.Background(), want, ids, f.AudioMask, uids, f.AudioMask); err != nil {
			t.Fatal(err)
		}
		rng := g.rng.Uint64()
		if err := g.GenerateInto(context.Background(), got, ids, f.AudioMask, uids, f.AudioMask); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(want, got) || rng != g.rng.Uint64() {
			t.Fatalf("scale %g mismatch", scale)
		}
		if n := testing.AllocsPerRun(3, func() {
			if err := g.GenerateInto(context.Background(), got, ids, f.AudioMask, uids, f.AudioMask); err != nil {
				t.Fatal(err)
			}
		}); n != 0 {
			t.Fatalf("allocs %v", n)
		}
		if scale == 0 {
			// No valid unconditional IDs are needed and no unconditional scratch is touched.
			for i := range u.hidden {
				u.hidden[i] = -12345
			}
			if err := g.GenerateInto(context.Background(), got, ids, f.AudioMask, nil, nil); err != nil {
				t.Fatal(err)
			}
			for _, v := range u.hidden {
				if v != -12345 {
					t.Fatal("unconditional forward executed")
				}
			}
		}
	}
}
