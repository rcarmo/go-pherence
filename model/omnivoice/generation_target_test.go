package omnivoice

import (
	"context"
	"fmt"
	"reflect"
	"testing"
)

func (g *Generation) legacyGenerateInto(ctx context.Context, dst, condIDs []int, condAudio []bool, uncondIDs []int, uncondAudio []bool) error {
	if g == nil || g.conditional == nil || ctx == nil {
		return fmt.Errorf("omnivoice: nil generation/context")
	}
	c := g.config
	if g.target <= 0 || g.target > g.conditional.tokens || (c.Guidance != 0 && (g.unconditional == nil || g.target > g.unconditional.tokens)) {
		return fmt.Errorf("omnivoice: target exceeds active backbone; reconfigure generation after backbones")
	}
	maskID := g.conditional.weights.Config.AudioMaskID
	if len(dst) != len(g.output) || len(condIDs) != g.books*g.conditional.tokens || len(condAudio) != g.conditional.tokens {
		return fmt.Errorf("omnivoice: generation input shape mismatch")
	}
	if c.Guidance != 0 && (len(uncondIDs) != g.books*g.unconditional.tokens || len(uncondAudio) != g.unconditional.tokens) {
		return fmt.Errorf("omnivoice: unconditional input shape mismatch")
	}
	for _, a := range condAudio[len(condAudio)-g.target:] {
		if !a {
			return fmt.Errorf("omnivoice: target must be audio")
		}
	}
	if c.Guidance != 0 {
		for _, a := range uncondAudio[len(uncondAudio)-g.target:] {
			if !a {
				return fmt.Errorf("omnivoice: unconditional target must be audio")
			}
		}
	}
	for i := range g.output {
		g.output[i] = maskID
	}
	g.copyTarget(condIDs, g.conditional.tokens)
	if c.Guidance != 0 {
		g.copyTarget(uncondIDs, g.unconditional.tokens)
	}
	g.source.Seed(c.Seed, c.Seed^0x9e3779b97f4a7c15)
	rng := g.rng
	condLogits := make([]float32, g.books*g.conditional.tokens*g.vocab)
	var uncondLogits []float32
	if c.Guidance != 0 {
		uncondLogits = make([]float32, g.books*g.unconditional.tokens*g.vocab)
	}
	for _, k := range g.schedule {
		if err := g.conditional.ForwardInto(ctx, condLogits, condIDs, condAudio, nil, nil); err != nil {
			return err
		}
		g.targetLogits(g.condTarget, condLogits, g.conditional.tokens)
		if c.Guidance != 0 {
			if err := g.unconditional.ForwardInto(ctx, uncondLogits, uncondIDs, uncondAudio, nil, nil); err != nil {
				return err
			}
			g.targetLogits(g.uncondTarget, uncondLogits, g.unconditional.tokens)
		}
		if k <= 0 {
			continue
		}
		if err := g.sampler.GuidedLogProbsInto(g.logProbs, g.condTarget, g.uncondTarget, g.target, c.Guidance, maskID); err != nil {
			return err
		}
		if c.ClassTemperature > 0 {
			for i := range g.classNoise {
				g.classNoise[i] = rng.Float32()
			}
		}
		if err := g.sampler.PredictTokensWithConfidenceInto(g.pred, g.confidence, g.logProbs, g.target, c.ClassTemperature, 0.1, GumbelNoise{Uniforms: g.classNoise}); err != nil {
			return err
		}
		temp := c.PositionTemperature
		if temp > 0 {
			for i := range g.positionNoise {
				g.positionNoise[i] = rng.Float32()
			}
		}
		if _, err := g.sampler.ApplyConfidenceSelection(g.output, g.pred, g.confidence, g.target, maskID, k, c.LayerPenalty, temp, GumbelNoise{Uniforms: g.positionNoise}); err != nil {
			return err
		}
		g.copyTarget(condIDs, g.conditional.tokens)
		if c.Guidance != 0 {
			g.copyTarget(uncondIDs, g.unconditional.tokens)
		}
	}
	copy(dst, g.output)
	return nil
}
func (g *Generation) targetLogits(dst, src []float32, tokens int) {
	for book := 0; book < g.books; book++ {
		start := ((book+1)*tokens - g.target) * g.vocab
		copy(dst[book*g.target*g.vocab:(book+1)*g.target*g.vocab], src[start:start+g.target*g.vocab])
	}
}

// Legacy loop intentionally retains full logits and late zero-reveal checks as
// an independent baseline for output/RNG parity, not a production fallback.
func TestGenerationTargetAndZeroStepsMatchLegacy(t *testing.T) {
	for _, steps := range []int{4, 16, 128} {
		for _, guidance := range []float32{0, 2} {
			b, f := loadBackboneFixture(t)
			u, err := NewBackboneSibling(b, f.Tokens)
			if err != nil {
				t.Fatal(err)
			}
			defer u.Close()
			cfg := DefaultGenerationConfig()
			cfg.Steps = steps
			cfg.Guidance = guidance
			cfg.ClassTemperature = .3
			g, err := NewGeneration(b, u, 2, cfg)
			if err != nil {
				t.Fatal(err)
			}
			ids := append([]int(nil), f.IDs...)
			uids := append([]int(nil), f.IDs...)
			want := make([]int, len(g.output))
			got := make([]int, len(want))
			if err := g.legacyGenerateInto(context.Background(), want, ids, f.AudioMask, uids, f.AudioMask); err != nil {
				t.Fatal(err)
			}
			rngWant := g.rng.Uint64()
			if err := g.GenerateInto(context.Background(), got, ids, f.AudioMask, uids, f.AudioMask); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(want, got) || g.rng.Uint64() != rngWant {
				t.Fatalf("steps %d guidance %g mismatch", steps, guidance)
			}
			zero := 0
			for _, n := range g.schedule {
				if n == 0 {
					zero++
				}
			}
			if steps == 128 && zero == 0 {
				t.Fatal("test needs zero-reveal steps")
			}
		}
	}
}

type countForwardContext struct {
	context.Context
	checks int
}

func (c *countForwardContext) Err() error { c.checks++; return nil }
func TestZeroRevealSkipsForwardChecks(t *testing.T) {
	b, f := loadBackboneFixture(t)
	cfg := DefaultGenerationConfig()
	cfg.Guidance = 0
	cfg.Steps = 128
	g, err := NewGeneration(b, nil, 2, cfg)
	if err != nil {
		t.Fatal(err)
	}
	ids := append([]int(nil), f.IDs...)
	out := make([]int, len(g.output))
	old := &countForwardContext{Context: context.Background()}
	if err := g.legacyGenerateInto(old, out, ids, f.AudioMask, nil, nil); err != nil {
		t.Fatal(err)
	}
	now := &countForwardContext{Context: context.Background()}
	if err := g.GenerateInto(now, out, ids, f.AudioMask, nil, nil); err != nil {
		t.Fatal(err)
	}
	if now.checks >= old.checks {
		t.Fatalf("no checks saved %d vs %d", now.checks, old.checks)
	}
}
