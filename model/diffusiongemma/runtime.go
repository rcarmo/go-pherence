package diffusiongemma

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// BlockDiffusionState keeps block-diffusion decoding state separate from
// autoregressive KV state. DiffusionGemma denoises a fixed-size token canvas
// over multiple steps, then appends accepted canvases to the prompt context.
type BlockDiffusionState struct {
	CanvasLength int  `json:"canvas_length"`
	Step         int  `json:"step"`
	Stopped      bool `json:"stopped"`
}

// ForwardInput describes one denoising forward call. Prompt/cache ownership is
// intentionally abstract at this layer; the concrete model runtime will decide
// how to encode prior canvases and expose cross-attention/prompt-cache state.
type ForwardInput struct {
	PromptIDs              []int            `json:"prompt_ids,omitempty"`
	Canvas                 []int            `json:"canvas"`
	Step                   int              `json:"step"`
	SelfConditioning       []float32        `json:"-"`
	SelfConditioningLogits [][]float32      `json:"-"` // previous raw canvas logits [canvas][vocab], matching llama.cpp sc_logits
	DeviceSelfConditioning any              `json:"-"` // backend-owned previous canvas logits/state (llama.cpp sc_dev analogue)
	SCTempInv              float32          `json:"-"` // 1/t for self-conditioning softmax (set by runtime loop)
	SampleDraws            []float64        `json:"-"` // pre-drawn per-position multinomial uniforms for backend/device sampling
	EncoderKV              []EncoderKVLayer `json:"-"`
}

// ForwardOutput contains per-canvas-position logits from the denoiser. Logits
// is laid out as [canvas_length][vocab_size]. SelfConditioning, when present,
// is the hidden-size soft embedding signal to feed into the next denoising step.
type ForwardOutput struct {
	Logits                 [][]float32 `json:"-"`
	SelfConditioning       []float32   `json:"-"`
	DeviceSelfConditioning any         `json:"-"` // backend-owned state to feed the next denoise step without host copy
	ArgmaxCanvas           []int       `json:"-"` // optional backend/device-computed argmax per canvas position
	SampledCanvas          []int       `json:"-"` // optional backend/device-computed multinomial sample per canvas position
	Entropy                []float64   `json:"-"` // optional backend/device-computed entropy per canvas position
}

// Denoiser is the narrow interface needed by the block-diffusion sampler. A
// future native implementation should wrap the DiffusionGemma text/vision model
// forward pass behind this interface.
type Denoiser interface {
	Denoise(ForwardInput) (ForwardOutput, error)
}

type EntropyProbe struct {
	Position  int       `json:"position"`
	Argmax    int       `json:"argmax"`
	Sampled   int       `json:"sampled"`
	Entropy   float64   `json:"entropy"`
	Accepted  bool      `json:"accepted"`
	TopIDs    []int     `json:"top_ids,omitempty"`
	TopLogits []float32 `json:"top_logits,omitempty"`
}

// CanvasStep records one denoising iteration for diagnostics and future
// streaming hooks.
type CanvasStep struct {
	Step          int            `json:"step"`
	Temperature   float64        `json:"temperature"`
	Accepted      int            `json:"accepted"`
	MeanEntropy   float64        `json:"mean_entropy"`
	FirstArgmax   int            `json:"first_argmax,omitempty"`
	FirstEntropy  float64        `json:"first_entropy,omitempty"`
	FirstSampled  int            `json:"first_sampled,omitempty"`
	FirstAccepted bool           `json:"first_accepted,omitempty"`
	MaxEntropy    float64        `json:"max_entropy,omitempty"`
	MaxEntropyPos int            `json:"max_entropy_pos,omitempty"`
	EntropyProbes []EntropyProbe `json:"entropy_probes,omitempty"`
	Held          int            `json:"held,omitempty"`
	Confident     bool           `json:"confident,omitempty"`
	Stopped       bool           `json:"stopped"`
}

// CanvasResult is the output of a single block-diffusion canvas generation.
type CanvasResult struct {
	Canvas     []int               `json:"canvas"`
	Steps      []CanvasStep        `json:"steps"`
	State      BlockDiffusionState `json:"state"`
	TrimCut    int                 `json:"trim_cut"`
	TrimReason string              `json:"trim_reason,omitempty"`
}

// GenerateCanvas runs the reference block-diffusion loop around an abstract
// denoiser. It is model-agnostic scaffold code: correctness fixtures should be
// built against Transformers before using it for real generation.
func GenerateCanvas(denoiser Denoiser, promptIDs []int, cfg DenoisingConfig, canvasLength, vocabSize int, rng canvasRNG) (CanvasResult, error) {
	return GenerateCanvasWithCallback(denoiser, promptIDs, cfg, canvasLength, vocabSize, rng, nil)
}

func diffusionGemmaEntropyProbePositions(canvasLength int) []int {
	v := strings.TrimSpace(os.Getenv("GO_PHERENCE_DIFFUSIONGEMMA_ENTROPY_PROBE_POSITIONS"))
	if v == "" || canvasLength <= 0 {
		return nil
	}
	seen := map[int]bool{}
	out := make([]int, 0)
	for _, part := range strings.Split(v, ",") {
		pos, err := strconv.Atoi(strings.TrimSpace(part))
		if err != nil || pos < 0 || pos >= canvasLength || seen[pos] {
			continue
		}
		seen[pos] = true
		out = append(out, pos)
	}
	return out
}

func equalIntSlices(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func retainLogitRows(rows [][]float32, n int) [][]float32 {
	if n <= 0 || len(rows) < n {
		return nil
	}
	for i := 0; i < n; i++ {
		if len(rows[i]) == 0 {
			return nil
		}
	}
	return rows[:n]
}

func topLogits(row []float32, k int) ([]int, []float32) {
	if k <= 0 || len(row) == 0 {
		return nil, nil
	}
	if k > len(row) {
		k = len(row)
	}
	ids := make([]int, 0, k)
	vals := make([]float32, 0, k)
	for id, v := range row {
		insert := len(vals)
		for insert > 0 && v > vals[insert-1] {
			insert--
		}
		if insert >= k {
			continue
		}
		ids = append(ids, 0)
		vals = append(vals, 0)
		copy(ids[insert+1:], ids[insert:])
		copy(vals[insert+1:], vals[insert:])
		ids[insert] = id
		vals[insert] = v
		if len(ids) > k {
			ids = ids[:k]
			vals = vals[:k]
		}
	}
	return ids, vals
}

// GenerateCanvasWithCallback is GenerateCanvas with an optional per-step callback.
func GenerateCanvasWithCallback(denoiser Denoiser, promptIDs []int, cfg DenoisingConfig, canvasLength, vocabSize int, rng canvasRNG, onStep func(CanvasStep, []int)) (CanvasResult, error) {
	if denoiser == nil {
		return CanvasResult{}, fmt.Errorf("nil DiffusionGemma denoiser")
	}
	if canvasLength <= 0 || vocabSize <= 0 {
		return CanvasResult{}, fmt.Errorf("invalid canvas/vocab dimensions canvas=%d vocab=%d", canvasLength, vocabSize)
	}
	defaults := DefaultDenoisingConfig()
	if cfg.MaxDenoisingSteps <= 0 {
		cfg.MaxDenoisingSteps = defaults.MaxDenoisingSteps
	}
	if cfg.TMin <= 0 {
		cfg.TMin = defaults.TMin
	}
	if cfg.TMax <= 0 {
		cfg.TMax = defaults.TMax
	}
	if cfg.StabilityThreshold < 0 {
		cfg.StabilityThreshold = defaults.StabilityThreshold
	}
	if cfg.ConfidenceThreshold < 0 {
		cfg.ConfidenceThreshold = defaults.ConfidenceThreshold
	}
	if cfg.Sampler.EntropyBound < 0 {
		cfg.Sampler.EntropyBound = defaults.Sampler.EntropyBound
	}
	if rng == nil {
		rng = NewMT19937RNG(0)
	}
	state := BlockDiffusionState{CanvasLength: canvasLength}
	canvas := make([]int, canvasLength)
	for i := range canvas {
		canvas[i] = rng.Intn(vocabSize)
	}
	// llama.cpp keeps two canvases: current_canvas is the renoised input to the
	// next step, while output_tokens stores the argmax canvas from the latest
	// model forward. Return the latter for text generation correctness.
	outputCanvas := append([]int(nil), canvas...)
	prevArgmax := make([]int, canvasLength)
	for i := range prevArgmax {
		prevArgmax[i] = -1
	}
	held := 0
	steps := make([]CanvasStep, 0, cfg.MaxDenoisingSteps)
	var selfConditioning []float32
	var selfConditioningLogits [][]float32
	var deviceSelfConditioning any
	defer func() {
		if closer, ok := deviceSelfConditioning.(interface{ Free() }); ok {
			closer.Free()
		}
	}()
	prevTempInv := float32(1)
	probePositions := diffusionGemmaEntropyProbePositions(canvasLength)
	for step := cfg.MaxDenoisingSteps; step > 0; step-- {
		state.Step = step
		temperature := LinearTemperature(cfg.TMin, cfg.TMax, cfg.MaxDenoisingSteps, step)
		tempInv := float32(1.0 / temperature)
		// llama.cpp feeds the previous step's raw canvas logits plus the previous
		// step's temp_inv into the self-conditioning graph. Device backends may carry
		// those logits as an opaque resident buffer instead of host rows.
		sampleDraws := make([]float64, canvasLength)
		renoiseTokens := make([]int, canvasLength)
		for i := 0; i < canvasLength; i++ {
			sampleDraws[i] = rng.Float64()
			renoiseTokens[i] = rng.Intn(vocabSize)
		}
		scTempInv := prevTempInv
		out, err := denoiser.Denoise(ForwardInput{PromptIDs: promptIDs, Canvas: canvas, Step: step, SelfConditioning: selfConditioning, SelfConditioningLogits: selfConditioningLogits, DeviceSelfConditioning: deviceSelfConditioning, SCTempInv: scTempInv, SampleDraws: sampleDraws})
		if err != nil {
			return CanvasResult{}, err
		}
		if len(out.Logits) < canvasLength && (len(out.ArgmaxCanvas) < canvasLength || len(out.SampledCanvas) < canvasLength || len(out.Entropy) < canvasLength) {
			return CanvasResult{}, fmt.Errorf("denoiser returned logits=%d argmax=%d sampled=%d entropy=%d, want canvas=%d", len(out.Logits), len(out.ArgmaxCanvas), len(out.SampledCanvas), len(out.Entropy), canvasLength)
		}
		denoiserCanvas := make([]int, canvasLength)
		argmaxCanvas := make([]int, canvasLength)
		entropy := make([]float64, canvasLength)
		var meanEntropy float64
		if len(out.ArgmaxCanvas) >= canvasLength && len(out.SampledCanvas) >= canvasLength && len(out.Entropy) >= canvasLength {
			copy(argmaxCanvas, out.ArgmaxCanvas[:canvasLength])
			copy(denoiserCanvas, out.SampledCanvas[:canvasLength])
			copy(entropy, out.Entropy[:canvasLength])
			for _, v := range entropy {
				meanEntropy += v
			}
		} else {
			for i := 0; i < canvasLength; i++ {
				logits := ApplyTemperature(out.Logits[i], temperature)
				argmaxCanvas[i] = Argmax(logits)
				denoiserCanvas[i] = SampleFromLogitsWithDraw(logits, sampleDraws[i])
				entropy[i] = TokenEntropyFromLogits(logits)
				meanEntropy += entropy[i]
			}
		}
		meanEntropy /= float64(canvasLength)
		if closer, ok := deviceSelfConditioning.(interface{ Free() }); ok && out.DeviceSelfConditioning != deviceSelfConditioning {
			closer.Free()
		}
		deviceSelfConditioning = out.DeviceSelfConditioning
		if len(out.SelfConditioning) > 0 {
			selfConditioning = append(selfConditioning[:0], out.SelfConditioning...)
		} else {
			selfConditioning = nil
		}
		selfConditioningLogits = retainLogitRows(out.Logits, canvasLength)
		prevTempInv = tempInv
		copy(outputCanvas, argmaxCanvas)
		if equalIntSlices(prevArgmax, argmaxCanvas) {
			held++
		} else {
			held = 0
		}
		confident := cfg.ConfidenceThreshold > 0 && meanEntropy < cfg.ConfidenceThreshold
		stopped := held >= cfg.StabilityThreshold && confident
		copy(prevArgmax, argmaxCanvas)
		// Always use entropy-bound acceptance + renoise for the next input canvas
		// (same as llama.cpp); do not return this working canvas as generated text.
		accepted := AcceptCanvas(canvas, denoiserCanvas, entropy, cfg.Sampler.EntropyBound)
		firstArgmax, firstSampled := 0, 0
		var firstEntropy float64
		var firstAccepted bool
		var maxEntropy float64
		maxEntropyPos := 0
		if canvasLength > 0 {
			firstArgmax = argmaxCanvas[0]
			firstSampled = denoiserCanvas[0]
			firstEntropy = entropy[0]
			firstAccepted = len(accepted.AcceptedMask) > 0 && accepted.AcceptedMask[0]
			maxEntropy = entropy[0]
			for i := 1; i < canvasLength; i++ {
				if entropy[i] > maxEntropy {
					maxEntropy = entropy[i]
					maxEntropyPos = i
				}
			}
		}
		probes := make([]EntropyProbe, 0, len(probePositions))
		for _, pos := range probePositions {
			probe := EntropyProbe{Position: pos, Argmax: argmaxCanvas[pos], Sampled: denoiserCanvas[pos], Entropy: entropy[pos], Accepted: len(accepted.AcceptedMask) > pos && accepted.AcceptedMask[pos]}
			if len(out.Logits) > pos {
				probe.TopIDs, probe.TopLogits = topLogits(out.Logits[pos], 5)
			}
			probes = append(probes, probe)
		}
		canvas = accepted.Canvas
		if !stopped {
			canvas = RenoiseCanvasWithTokens(canvas, accepted.AcceptedMask, renoiseTokens)
		}
		steps = append(steps, CanvasStep{Step: step, Temperature: temperature, Accepted: accepted.Accepted, MeanEntropy: meanEntropy, FirstArgmax: firstArgmax, FirstEntropy: firstEntropy, FirstSampled: firstSampled, FirstAccepted: firstAccepted, MaxEntropy: maxEntropy, MaxEntropyPos: maxEntropyPos, EntropyProbes: probes, Held: held, Confident: confident, Stopped: stopped})
		if onStep != nil {
			onStep(steps[len(steps)-1], canvas)
		}
		if stopped {
			state.Stopped = true
			break
		}
	}
	return CanvasResult{Canvas: outputCanvas, Steps: steps, State: state, TrimCut: len(outputCanvas)}, nil
}
