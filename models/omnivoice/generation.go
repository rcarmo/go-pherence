package omnivoice

import (
	"context"
	"fmt"
	"math"
	"math/rand/v2"
)

// GenerationConfig controls token-space denoising. Go PCG noise is deterministic
// for a seed but does not reproduce PyTorch's random stream. Steps and schedule
// formulas follow upstream; equal-score top-k ties use lowest flattened index.
type GenerationConfig struct {
	Steps                                                                    int
	TimeShift, Guidance, ClassTemperature, PositionTemperature, LayerPenalty float32
	Seed                                                                     uint64
}

func DefaultGenerationConfig() GenerationConfig {
	return GenerationConfig{Steps: 16, TimeShift: 0.1, Guidance: 2, ClassTemperature: 0, PositionTemperature: 5, LayerPenalty: 5, Seed: 42}
}

// Generation owns two preallocated batch-one forwards for conditional and
// unconditional inputs. Input prompt construction remains the caller's job.
// Target frames must occupy the final positions of each sequence.
type Generation struct {
	rng                                                                      *rand.Rand
	source                                                                   *rand.PCG
	conditional, unconditional                                               *Backbone
	config                                                                   GenerationConfig
	target, books, vocab                                                     int
	sampler                                                                  *SamplerWorkspace
	condLogits, uncondLogits, condTarget, uncondTarget, logProbs, confidence []float32
	pred, output, schedule                                                   []int
	times, classNoise, positionNoise                                         []float32
}

func NewGeneration(conditional, unconditional *Backbone, target int, c GenerationConfig) (*Generation, error) {
	if conditional == nil || target <= 0 || target > conditional.tokens || c.Steps < 1 || c.Steps > 128 {
		return nil, fmt.Errorf("omnivoice: invalid generation shape/steps")
	}
	for _, v := range []float32{c.TimeShift, c.Guidance, c.ClassTemperature, c.PositionTemperature, c.LayerPenalty} {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) || v < 0 {
			return nil, fmt.Errorf("omnivoice: invalid generation parameter")
		}
	}
	if c.TimeShift <= 0 {
		return nil, fmt.Errorf("omnivoice: time shift must be positive")
	}
	cfg := conditional.weights.Config
	if c.Guidance != 0 && (unconditional == nil || target > unconditional.tokens || unconditional.weights != conditional.weights) {
		return nil, fmt.Errorf("omnivoice: guidance requires compatible unconditional backbone")
	}
	rows := target * cfg.NumAudioCodebook
	size := rows * cfg.AudioVocabSize
	sampler, err := NewSamplerWorkspace(cfg.NumAudioCodebook, target, cfg.AudioVocabSize)
	if err != nil {
		return nil, err
	}
	g := &Generation{conditional: conditional, unconditional: unconditional, config: c, target: target, books: cfg.NumAudioCodebook, vocab: cfg.AudioVocabSize, sampler: sampler, condLogits: make([]float32, cfg.NumAudioCodebook*conditional.tokens*cfg.AudioVocabSize), condTarget: make([]float32, size), uncondTarget: make([]float32, size), logProbs: make([]float32, size), confidence: make([]float32, rows), pred: make([]int, rows), output: make([]int, rows), schedule: make([]int, c.Steps), times: make([]float32, c.Steps+1), classNoise: make([]float32, size), positionNoise: make([]float32, rows)}
	if c.Guidance != 0 {
		g.uncondLogits = make([]float32, cfg.NumAudioCodebook*unconditional.tokens*cfg.AudioVocabSize)
	}
	g.source = rand.NewPCG(c.Seed, c.Seed^0x9e3779b97f4a7c15)
	g.rng = rand.New(g.source)
	if err = TimeStepsInto(g.times, 0, 1, c.Steps, c.TimeShift); err != nil {
		return nil, err
	}
	if err = BuildUnmaskScheduleInto(g.schedule, target, cfg.NumAudioCodebook, g.times); err != nil {
		return nil, err
	}
	return g, nil
}

// GenerateInto updates conditional/unconditional target IDs in place and fills
// dst [codebook,target]. Prompt IDs are preserved. Each invocation restarts the
// seed and mask state; output after failure/cancellation must be discarded.
func (g *Generation) GenerateInto(ctx context.Context, dst, condIDs []int, condAudio []bool, uncondIDs []int, uncondAudio []bool) error {
	c := g.config
	maskID := g.conditional.weights.Config.AudioMaskID
	if ctx == nil || len(dst) != len(g.output) || len(condIDs) != g.books*g.conditional.tokens || len(condAudio) != g.conditional.tokens {
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
	for _, k := range g.schedule {
		if err := g.conditional.ForwardInto(ctx, g.condLogits, condIDs, condAudio, nil, nil); err != nil {
			return err
		}
		g.targetLogits(g.condTarget, g.condLogits, g.conditional.tokens)
		if c.Guidance != 0 {
			if err := g.unconditional.ForwardInto(ctx, g.uncondLogits, uncondIDs, uncondAudio, nil, nil); err != nil {
				return err
			}
			g.targetLogits(g.uncondTarget, g.uncondLogits, g.unconditional.tokens)
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
func (g *Generation) copyTarget(ids []int, tokens int) {
	for book := 0; book < g.books; book++ {
		copy(ids[(book+1)*tokens-g.target:(book+1)*tokens], g.output[book*g.target:(book+1)*g.target])
	}
}
func (g *Generation) targetLogits(dst, src []float32, tokens int) {
	for book := 0; book < g.books; book++ {
		start := ((book+1)*tokens - g.target) * g.vocab
		copy(dst[book*g.target*g.vocab:(book+1)*g.target*g.vocab], src[start:start+g.target*g.vocab])
	}
}
