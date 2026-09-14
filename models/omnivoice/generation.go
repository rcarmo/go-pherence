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
	// SharedTraversal is opt-in and requires guided streamed float32 siblings.
	SharedTraversal                                                          bool
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
	condPrefix, uncondPrefix                       []float32
	activeTimes                                    []bool
	rng                                            *rand.Rand
	source                                         *rand.PCG
	conditional, unconditional                     *Backbone
	config                                         GenerationConfig
	target, maxTarget, books, vocab                int
	sampler                                        *SamplerWorkspace
	condTarget, uncondTarget, logProbs, confidence []float32
	pred, output, schedule                         []int
	times, classNoise, positionNoise               []float32
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
	if c.SharedTraversal && (c.Guidance == 0 || unconditional == nil || conditional == unconditional || conditional.layer != unconditional.layer || conditional.resident != nil || unconditional.resident != nil || conditional.directQ8 != nil || unconditional.directQ8 != nil) {
		return nil, fmt.Errorf("omnivoice: shared traversal requires guided streamed float32 siblings")
	}
	cfg := conditional.weights.Config
	if c.Guidance != 0 && (unconditional == nil || target > unconditional.tokens || unconditional.weights != conditional.weights) {
		return nil, fmt.Errorf("omnivoice: guidance requires compatible unconditional backbone")
	}
	rows, ok := product(target, cfg.NumAudioCodebook)
	if !ok {
		return nil, fmt.Errorf("omnivoice: generation shape overflow")
	}
	size, ok := product(rows, cfg.AudioVocabSize)
	if !ok {
		return nil, fmt.Errorf("omnivoice: generation shape overflow")
	}
	sampler, err := NewSamplerWorkspace(cfg.NumAudioCodebook, target, cfg.AudioVocabSize)
	if err != nil {
		return nil, err
	}
	g := &Generation{conditional: conditional, unconditional: unconditional, config: c, target: target, maxTarget: target, books: cfg.NumAudioCodebook, vocab: cfg.AudioVocabSize, sampler: sampler, condTarget: make([]float32, size), uncondTarget: make([]float32, size), logProbs: make([]float32, size), confidence: make([]float32, rows), pred: make([]int, rows), output: make([]int, rows), schedule: make([]int, c.Steps), times: make([]float32, c.Steps+1), classNoise: make([]float32, size), positionNoise: make([]float32, rows)}
	// Only raw prefix embeddings are cached: transformer activations still depend
	// on every target token through full non-causal attention. Rebuild on each call.
	h := cfg.LLMConfig.HiddenSize
	g.condPrefix = make([]float32, (conditional.maxTokens-1)*h)
	if c.Guidance != 0 {
		g.uncondPrefix = make([]float32, (unconditional.maxTokens-1)*h)
	}
	g.activeTimes = make([]bool, target)
	g.source = rand.NewPCG(c.Seed, c.Seed^0x9e3779b97f4a7c15)
	g.rng = rand.New(g.source)
	if err = TimeStepsInto(g.times, 0, 1, c.Steps, c.TimeShift); err != nil {
		return nil, err
	}
	if err = g.Reconfigure(target); err != nil {
		return nil, err
	}
	return g, nil
}

// Reconfigure narrows or restores the active target-frame views within the
// original reservation and recomputes the schedule for the new target length.
func (g *Generation) Reconfigure(target int) error {
	if g == nil || g.conditional == nil {
		return fmt.Errorf("omnivoice: nil generation/backbone")
	}
	if target <= 0 || target > g.maxTarget || target > g.conditional.tokens {
		return fmt.Errorf("omnivoice: target=%d outside [1,%d] or conditional tokens=%d", target, g.maxTarget, g.conditional.tokens)
	}
	if g.config.Guidance != 0 {
		if g.unconditional == nil {
			return fmt.Errorf("omnivoice: nil unconditional backbone")
		}
		if target > g.unconditional.tokens {
			return fmt.Errorf("omnivoice: target=%d exceeds unconditional tokens=%d", target, g.unconditional.tokens)
		}
	}
	rows, ok := product(target, g.books)
	if !ok {
		return fmt.Errorf("omnivoice: generation shape overflow")
	}
	size, ok := product(rows, g.vocab)
	if !ok {
		return fmt.Errorf("omnivoice: generation shape overflow")
	}
	g.activeTimes = g.activeTimes[:target]
	g.target = target
	g.condTarget = g.condTarget[:size]
	g.uncondTarget = g.uncondTarget[:size]
	g.logProbs = g.logProbs[:size]
	g.confidence = g.confidence[:rows]
	g.pred = g.pred[:rows]
	g.output = g.output[:rows]
	g.classNoise = g.classNoise[:size]
	g.positionNoise = g.positionNoise[:rows]
	return BuildUnmaskScheduleInto(g.schedule, target, g.books, g.times)
}

// GenerateInto updates conditional/unconditional target IDs in place and fills
// dst [codebook,target]. Prompt IDs are preserved. Each invocation restarts the
// seed and mask state; output after failure/cancellation must be discarded.
func (g *Generation) GenerateInto(ctx context.Context, dst, condIDs []int, condAudio []bool, uncondIDs []int, uncondAudio []bool) error {
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
	if err := ctx.Err(); err != nil {
		return err
	}
	h := g.conditional.weights.Config.LLMConfig.HiddenSize
	condPrefix := g.condPrefix[:(g.conditional.tokens-g.target)*h]
	if err := g.conditional.embedRangeInto(condPrefix, condIDs, condAudio, 0, g.conditional.tokens-g.target); err != nil {
		return err
	}
	var uncondPrefix []float32
	if c.Guidance != 0 {
		uncondPrefix = g.uncondPrefix[:(g.unconditional.tokens-g.target)*h]
		if err := g.unconditional.embedRangeInto(uncondPrefix, uncondIDs, uncondAudio, 0, g.unconditional.tokens-g.target); err != nil {
			return err
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
		if err := ctx.Err(); err != nil {
			return err
		}
		// No IDs or RNG values change on zero-reveal steps. Full attention
		// would repeat the preceding state and its logits would be discarded.
		if k <= 0 {
			continue
		}
		clear(g.activeTimes)
		for row, token := range g.output {
			if token == maskID {
				g.activeTimes[row%g.target] = true
			}
		}
		// Dense sampler/noise indexing is deliberately retained. Revealed rows
		// may contain stale logits, but confidence selection excludes them.
		if c.SharedTraversal {
			if err := g.forwardPair(ctx, condIDs, condAudio, uncondIDs, uncondAudio, condPrefix, uncondPrefix); err != nil {
				return err
			}
		} else {
			if err := g.conditional.forwardInto(ctx, g.condTarget, condIDs, condAudio, nil, nil, g.target, condPrefix, g.activeTimes); err != nil {
				return err
			}
			if c.Guidance != 0 {
				if err := g.unconditional.forwardInto(ctx, g.uncondTarget, uncondIDs, uncondAudio, nil, nil, g.target, uncondPrefix, g.activeTimes); err != nil {
					return err
				}
			}
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
