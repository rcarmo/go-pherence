package diffusiongemma

import (
	"math"
	"math/rand"
	"slices"
	"sort"
)

// SamplerConfig mirrors the reference EntropyBoundSamplerConfig.
type SamplerConfig struct {
	EntropyBound float64 `json:"entropy_bound"`
}

// DenoisingConfig captures the non-model generation controls needed by the
// block-diffusion loop.
type DenoisingConfig struct {
	MaxDenoisingSteps   int           `json:"max_denoising_steps"`
	TMin                float64       `json:"t_min"`
	TMax                float64       `json:"t_max"`
	StabilityThreshold  int           `json:"stability_threshold"`
	ConfidenceThreshold float64       `json:"confidence_threshold"`
	Sampler             SamplerConfig `json:"sampler"`
}

func DefaultDenoisingConfig() DenoisingConfig {
	return DenoisingConfig{MaxDenoisingSteps: 48, TMin: 0.4, TMax: 0.8, StabilityThreshold: 1, ConfidenceThreshold: 0.005, Sampler: SamplerConfig{EntropyBound: 0.1}}
}

func DenoisingConfigFromDefaults(g GenerationDefaults) DenoisingConfig {
	cfg := DefaultDenoisingConfig()
	if g.MaxDenoisingSteps > 0 {
		cfg.MaxDenoisingSteps = g.MaxDenoisingSteps
	}
	if g.TMin > 0 {
		cfg.TMin = g.TMin
	}
	if g.TMax > 0 {
		cfg.TMax = g.TMax
	}
	if g.StabilityThreshold >= 0 {
		cfg.StabilityThreshold = g.StabilityThreshold
	}
	if g.ConfidenceThreshold > 0 {
		cfg.ConfidenceThreshold = g.ConfidenceThreshold
	}
	if g.EntropyBound > 0 {
		cfg.Sampler.EntropyBound = g.EntropyBound
	}
	return cfg
}

func LinearTemperature(tMin, tMax float64, maxDenoisingSteps, curStep int) float64 {
	if maxDenoisingSteps <= 0 {
		return tMin
	}
	return tMin + ((tMax - tMin) * (float64(curStep) / float64(maxDenoisingSteps)))
}

func ApplyTemperature(logits []float32, temperature float64) []float32 {
	out := make([]float32, len(logits))
	if temperature <= 0 {
		copy(out, logits)
		return out
	}
	inv := float32(1.0 / temperature)
	for i, v := range logits {
		out[i] = v * inv
	}
	return out
}

// TokenEntropyFromLogits computes categorical entropy for one token position.
func TokenEntropyFromLogits(logits []float32) float64 {
	if len(logits) == 0 {
		return 0
	}
	maxLogit := logits[0]
	for _, v := range logits[1:] {
		if v > maxLogit {
			maxLogit = v
		}
	}
	var sum float64
	var sumXLogX float64
	for _, v := range logits {
		x := math.Exp(float64(v - maxLogit))
		sum += x
		if x > 0 {
			sumXLogX += x * math.Log(x)
		}
	}
	if sum <= 0 {
		return 0
	}
	return math.Log(sum) - sumXLogX/sum
}

func Argmax(logits []float32) int {
	if len(logits) == 0 {
		return -1
	}
	best := 0
	for i, v := range logits[1:] {
		if v > logits[best] {
			best = i + 1
		}
	}
	return best
}

func TokenStatsFromLogits(logits []float32, temperature float64, rng *rand.Rand) (argmaxID int, sampledID int, entropy float64) {
	if len(logits) == 0 {
		return -1, -1, 0
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(1))
	}
	invTemp := float32(1)
	if temperature > 0 {
		invTemp = float32(1.0 / temperature)
	}
	const sparseLimit = 4096
	argmaxID = -1
	maxLogit := float32(math.Inf(-1))
	finiteIDs := make([]int, 0, 16)
	finiteVals := make([]float32, 0, 16)
	dense := false
	for id, raw := range logits {
		if math.IsNaN(float64(raw)) {
			continue
		}
		v := raw * invTemp
		if v > maxLogit {
			maxLogit = v
			argmaxID = id
		}
		if math.IsInf(float64(v), -1) {
			continue
		}
		if len(finiteIDs) < sparseLimit {
			finiteIDs = append(finiteIDs, id)
			finiteVals = append(finiteVals, v)
		} else {
			dense = true
		}
	}
	if argmaxID < 0 || math.IsInf(float64(maxLogit), -1) || math.IsNaN(float64(maxLogit)) {
		return argmaxID, argmaxID, 0
	}
	var sum float64
	var sumXLogX float64
	if !dense {
		weights := make([]float64, len(finiteVals))
		for i, v := range finiteVals {
			w := math.Exp(float64(v - maxLogit))
			weights[i] = w
			sum += w
			if w > 0 {
				sumXLogX += w * math.Log(w)
			}
		}
		if sum <= 0 || math.IsNaN(sum) {
			return argmaxID, argmaxID, 0
		}
		draw := rng.Float64() * sum
		var cumulative float64
		sampledID = argmaxID
		for i, w := range weights {
			cumulative += w
			if draw <= cumulative {
				sampledID = finiteIDs[i]
				break
			}
		}
		return argmaxID, sampledID, math.Log(sum) - sumXLogX/sum
	}
	for _, raw := range logits {
		if math.IsNaN(float64(raw)) || math.IsInf(float64(raw), -1) {
			continue
		}
		v := raw * invTemp
		w := math.Exp(float64(v - maxLogit))
		sum += w
		if w > 0 {
			sumXLogX += w * math.Log(w)
		}
	}
	if sum <= 0 || math.IsNaN(sum) {
		return argmaxID, argmaxID, 0
	}
	draw := rng.Float64() * sum
	var cumulative float64
	sampledID = argmaxID
	for id, raw := range logits {
		if math.IsNaN(float64(raw)) || math.IsInf(float64(raw), -1) {
			continue
		}
		v := raw * invTemp
		cumulative += math.Exp(float64(v - maxLogit))
		if draw <= cumulative {
			sampledID = id
			break
		}
	}
	return argmaxID, sampledID, math.Log(sum) - sumXLogX/sum
}

func SampleFromLogits(logits []float32, rng *rand.Rand) int {
	if len(logits) == 0 {
		return -1
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(1))
	}
	maxLogit := float32(math.Inf(-1))
	for _, v := range logits {
		if v > maxLogit {
			maxLogit = v
		}
	}
	if math.IsInf(float64(maxLogit), -1) || math.IsNaN(float64(maxLogit)) {
		return Argmax(logits)
	}
	weights := make([]float64, len(logits))
	var sum float64
	for i, v := range logits {
		if math.IsInf(float64(v), -1) || math.IsNaN(float64(v)) {
			continue
		}
		w := math.Exp(float64(v - maxLogit))
		weights[i] = w
		sum += w
	}
	if sum <= 0 || math.IsNaN(sum) {
		return Argmax(logits)
	}
	draw := rng.Float64() * sum
	var cumulative float64
	for i, w := range weights {
		cumulative += w
		if draw <= cumulative {
			return i
		}
	}
	return Argmax(logits)
}

type AcceptanceResult struct {
	Canvas       []int  `json:"canvas"`
	AcceptedMask []bool `json:"accepted_mask"`
	Accepted     int    `json:"accepted"`
}

// AcceptCanvas implements the reference entropy-bound selection rule for a
// single canvas: sort token positions by entropy and accept the lowest-entropy
// prefix where cumulative_entropy - max_entropy <= entropy_bound.
func AcceptCanvas(currentCanvas, denoiserCanvas []int, tokenEntropy []float64, entropyBound float64) AcceptanceResult {
	n := len(currentCanvas)
	out := AcceptanceResult{Canvas: append([]int(nil), currentCanvas...), AcceptedMask: make([]bool, n)}
	if n == 0 || len(denoiserCanvas) < n || len(tokenEntropy) < n || entropyBound < 0 {
		return out
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool { return tokenEntropy[order[i]] < tokenEntropy[order[j]] })
	var cumulative float64
	for _, idx := range order {
		cumulative += tokenEntropy[idx]
		if cumulative-tokenEntropy[idx] <= entropyBound {
			out.AcceptedMask[idx] = true
			out.Canvas[idx] = denoiserCanvas[idx]
			out.Accepted++
		}
	}
	return out
}

func RenoiseCanvas(acceptedCanvas []int, acceptedMask []bool, vocabSize int, rng *rand.Rand) []int {
	out := append([]int(nil), acceptedCanvas...)
	if vocabSize <= 0 || rng == nil {
		return out
	}
	for i := range out {
		accepted := i < len(acceptedMask) && acceptedMask[i]
		if !accepted {
			out[i] = rng.Intn(vocabSize)
		}
	}
	return out
}

type StableConfidentStopper struct {
	StabilityThreshold  int
	ConfidenceThreshold float64
	history             [][]int
}

func NewStableConfidentStopper(stabilityThreshold int, confidenceThreshold float64) *StableConfidentStopper {
	if stabilityThreshold < 0 {
		stabilityThreshold = 0
	}
	return &StableConfidentStopper{StabilityThreshold: stabilityThreshold, ConfidenceThreshold: confidenceThreshold}
}

func (s *StableConfidentStopper) Reset() { s.history = nil }

func (s *StableConfidentStopper) ShouldStop(argmaxCanvas []int, tokenEntropy []float64) bool {
	if s == nil {
		return false
	}
	stable := s.StabilityThreshold == 0
	if s.StabilityThreshold > 0 {
		if len(s.history) == s.StabilityThreshold {
			stable = true
			for _, prev := range s.history {
				if !slices.Equal(prev, argmaxCanvas) {
					stable = false
					break
				}
			}
		}
		s.history = append(s.history, append([]int(nil), argmaxCanvas...))
		if len(s.history) > s.StabilityThreshold {
			s.history = s.history[len(s.history)-s.StabilityThreshold:]
		}
	}
	var meanEntropy float64
	if len(tokenEntropy) > 0 {
		for _, v := range tokenEntropy {
			meanEntropy += v
		}
		meanEntropy /= float64(len(tokenEntropy))
	}
	confident := s.ConfidenceThreshold > 0 && meanEntropy < s.ConfidenceThreshold
	return stable && confident
}
