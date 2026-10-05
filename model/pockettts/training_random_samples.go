package pockettts

import (
	"fmt"
	"math"
	"slices"

	"github.com/rcarmo/go-pherence/internal/checked"
)

// TrainingRandomAlgorithm identifies the raw-word consumption and rounding
// policy. Numerical qualification against independent references is pending.
const TrainingRandomAlgorithm = "pcg-bm52x53-cos-discard-f32-lsd040-always-v1"

// TrainingRandomRecipe declares the restricted numerical recipe. OptionsSHA256
// must match the proposal identity and bind this policy in caller-owned options.
// A matching digest cannot prove what options were hashed or later used by loss.
type TrainingRandomRecipe struct {
	Algorithm, OptionsSHA256  string
	DistillProbability        float32
	TextDropout, VoiceDropout float32
	FlowBatchMultiplier       int
	StopGradient              string
	Normalize                 bool
	PEqual, EOSLossWeight     float32
	LogitMean, LogitStd       float32
}

type TrainingRandomLimits struct {
	Samples     TrainingSampleLimits
	MaxRawDraws int
}

// SampleTrainingInto samples only always-distill, dropout0, row1, multiplier1,
// normalised minimal LSD. It owns no RNG/cache and retains no buffer references.
// Destination views are exact full-frame shapes, disjoint typed float storage;
// unsafe-forged bool/float aliases and concurrent mutation are unsupported.
// Admission errors leave outputs and proposal unchanged. After sampling begins,
// any error makes outputs unusable/possibly partial: abandon the entire proposal.
// No implicit rewind/retry, metadata commit or model rollback is performed.
// After success make no further draws; commit only after completed optimiser+EMA.
// Failed/uncertain updates also require coherent model checkpoint recovery.
// This is a native finite-grid policy, not PyTorch RNG-sequence equivalence.
func SampleTrainingInto(dst TrainingStepSamples, p *TrainingRowProposal, frames, validFrames, channels int, recipe TrainingRandomRecipe, limits TrainingRandomLimits) error {
	return sampleTrainingRandomInto(dst, p, frames, validFrames, channels, recipe, limits, nil)
}

// The hook is used only by local failure tests, after a generated value is
// written. It cannot change the public API or introduce cached normal state.
func sampleTrainingRandomInto(dst TrainingStepSamples, p *TrainingRowProposal, frames, validFrames, channels int, recipe TrainingRandomRecipe, limits TrainingRandomLimits, hook func(string, int) error) error {
	active, err := admitTrainingRandom(dst, p, frames, validFrames, channels, recipe, limits)
	if err != nil {
		return err
	}
	for i := 0; i < active; i++ {
		z, err := proposalNormal(p, false)
		if err != nil {
			return err
		}
		dst.Noise[i] = z
		if hook != nil {
			if err := hook("noise", i); err != nil {
				return err
			}
		}
	}
	for row := 0; row < validFrames; row++ {
		z, err := proposalNormal(p, true)
		if err != nil {
			return err
		}
		time, err := trainingTimeFromNormal(z)
		if err != nil {
			return err
		}
		dst.DiagonalTime[row] = time
		if hook != nil {
			if err := hook("diagonal", row); err != nil {
				return err
			}
		}
	}
	for row := 0; row < validFrames; row++ {
		a, err := proposalNormal(p, true)
		if err != nil {
			return err
		}
		b, err := proposalNormal(p, true)
		if err != nil {
			return err
		}
		s, t, err := trainingPairFromNormals(a, b)
		if err != nil {
			return err
		}
		dst.DistillS[row], dst.DistillT[row] = s, t
		if hook != nil {
			if err := hook("pair", row); err != nil {
				return err
			}
		}
	}
	clear(dst.Noise[active:])
	clear(dst.DiagonalTime[validFrames:])
	clear(dst.DistillS[validFrames:])
	clear(dst.DistillT[validFrames:])
	for row := range dst.Mask {
		dst.Mask[row] = row < validFrames
	}
	return nil
}

func admitTrainingRandom(dst TrainingStepSamples, p *TrainingRowProposal, frames, validFrames, channels int, r TrainingRandomRecipe, limits TrainingRandomLimits) (int, error) {
	if r.Algorithm != TrainingRandomAlgorithm || r.DistillProbability != 1 || r.TextDropout != 0 || r.VoiceDropout != 0 || r.FlowBatchMultiplier != 1 || r.StopGradient != "minimal" || !r.Normalize || r.PEqual != .75 || r.EOSLossWeight != float32(.1) || r.LogitMean != float32(.4) || r.LogitStd != 1 || !runHex(r.OptionsSHA256, 64) {
		return 0, fmt.Errorf("unsupported Pocket TTS random recipe")
	}
	if limits.Samples.MaxFrames <= 0 || limits.Samples.MaxNoiseElements <= 0 || limits.MaxRawDraws <= 0 || frames <= 0 || frames > limits.Samples.MaxFrames || validFrames <= 0 || validFrames > frames || channels < 2 {
		return 0, fmt.Errorf("invalid Pocket TTS random sample geometry/limits")
	}
	full, ok := checked.MulInt(frames, channels)
	if !ok || full > limits.Samples.MaxNoiseElements {
		return 0, fmt.Errorf("Pocket TTS random noise bound exceeded")
	}
	if _, ok := checked.MulInt(full, 4); !ok {
		return 0, fmt.Errorf("Pocket TTS random byte range overflows")
	}
	active := validFrames * channels // bounded by full, with positive channels
	noiseWords, ok := checked.MulInt(2, active)
	if !ok {
		return 0, fmt.Errorf("Pocket TTS random noise draw count overflows")
	}
	timeWords, ok := checked.MulInt(6, validFrames)
	if !ok {
		return 0, fmt.Errorf("Pocket TTS random time draw count overflows")
	}
	total, ok := checked.AddInt(noiseWords, timeWords)
	if !ok || total > limits.MaxRawDraws {
		return 0, fmt.Errorf("Pocket TTS random raw draw bound exceeded")
	}
	if len(dst.Mask) != frames || len(dst.Noise) != full || len(dst.DiagonalTime) != frames || len(dst.DistillS) != frames || len(dst.DistillT) != frames {
		return 0, fmt.Errorf("invalid Pocket TTS random destination shape")
	}
	outputs := [4][]float32{dst.Noise, dst.DiagonalTime, dst.DistillS, dst.DistillT}
	for i, output := range outputs {
		for _, other := range outputs[i+1:] {
			if f32SlicesOverlap(output, other) {
				return 0, fmt.Errorf("Pocket TTS random destinations overlap")
			}
		}
	}
	if err := validateRandomProposal(p, r.OptionsSHA256); err != nil {
		return 0, err
	}
	return active, nil
}

func validateRandomProposal(p *TrainingRowProposal, options string) error {
	if p == nil || p.parent == nil || p.committed {
		return fmt.Errorf("inactive Pocket TTS random proposal")
	}
	parent := &p.parent.core
	if err := parent.validate(); err != nil {
		return err
	}
	if parent.step == math.MaxUint64 || p.parentStep != parent.step || p.next.identity != parent.identity || options != parent.identity.OptionsSHA256 || p.next.rows != parent.rows || p.next.step != parent.step+1 || p.next.timeRNG != parent.timeRNG || p.next.noiseRNG != parent.noiseRNG {
		return fmt.Errorf("stale/consumed/inconsistent Pocket TTS random proposal")
	}
	if err := p.next.validate(); err != nil {
		return err
	}
	if p.row >= uint32(parent.rows) || p.row != p.next.order[p.next.cursor-1] {
		return fmt.Errorf("Pocket TTS random proposal row mismatch")
	}
	if parent.cursor < parent.rows {
		if p.next.epoch != parent.epoch || p.next.cursor != parent.cursor+1 || p.next.orderRNG != parent.orderRNG || !slices.Equal(p.next.order, parent.order) {
			return fmt.Errorf("Pocket TTS random proposal order mismatch")
		}
	} else {
		// Verify rollover against a private bounded copy. This replays only
		// shuffle metadata in scratch, never draws live Time/Noise or mutates
		// parent/proposal. Core validation work is bounded by MaxTrainingRunRows.
		expected := parent.clone()
		expected.epoch++
		expected.cursor = 1
		shuffleRunOrder(&expected)
		if p.next.epoch != expected.epoch || p.next.cursor != expected.cursor || p.next.orderRNG != expected.orderRNG || !slices.Equal(p.next.order, expected.order) {
			return fmt.Errorf("Pocket TTS random rollover mismatch")
		}
	}
	return nil
}

func trainingUniforms(a, b uint64) (u, v float64) {
	midpoint := float64(float64(a>>12) + 0.5)
	u = float64(midpoint * 0x1p-52)
	v = float64(float64(b>>11) * 0x1p-53)
	return
}

func trainingNormalFromRaw(a, b uint64) (float32, error) {
	u, v := trainingUniforms(a, b)
	// Each explicit conversion is a rounding boundary; no contraction or
	// reassociation is assumed. math transcendental rounding is toolchain/ISA
	// dependent and remains UNQUALIFIED against independent references.
	logU := float64(math.Log(u))
	radialSquare := float64(-2 * logU)
	radius := float64(math.Sqrt(radialSquare))
	tau := math.Float64frombits(0x401921fb54442d18) // nearest F64 to 2*pi
	angle := float64(tau * v)
	cosine := float64(math.Cos(angle))
	product := float64(radius * cosine)
	z := float32(product)
	for _, value := range [...]float64{u, v, logU, radialSquare, radius, tau, angle, cosine, product} {
		if !finite64(value) {
			return 0, fmt.Errorf("non-finite Pocket TTS Gaussian intermediate")
		}
	}
	if !isFinite(z) {
		return 0, fmt.Errorf("non-finite Pocket TTS Gaussian")
	}
	return z, nil
}

func proposalNormal(p *TrainingRowProposal, time bool) (float32, error) {
	var a, b uint64
	var err error
	if time {
		a, err = p.TimeUint64()
		if err == nil {
			b, err = p.TimeUint64()
		}
	} else {
		a, err = p.NoiseUint64()
		if err == nil {
			b, err = p.NoiseUint64()
		}
	}
	if err != nil {
		return 0, err
	}
	return trainingNormalFromRaw(a, b)
}

func trainingLogit(z float32) (float32, error) {
	product := float32(z * float32(1))
	logit := float32(product + float32(.4))
	if !isFinite(z) || !isFinite(product) || !isFinite(logit) {
		return 0, fmt.Errorf("non-finite Pocket TTS logit")
	}
	return logit, nil
}
func trainingTimeFromLogit(logit float32) (float32, error) {
	if !isFinite(logit) {
		return 0, fmt.Errorf("non-finite Pocket TTS time logit")
	}
	var exponential, denominator, result float64
	if logit >= 0 {
		negated := float64(-float64(logit))
		exponential = float64(math.Exp(negated))
		denominator = float64(1 + exponential)
		result = float64(1 / denominator)
	} else {
		exponential = float64(math.Exp(float64(logit)))
		denominator = float64(1 + exponential)
		result = float64(exponential / denominator)
	}
	value := float32(result)
	if !finite64(exponential) || !finite64(denominator) || !finite64(result) || !isFinite(value) || value < 0 || value > 1 {
		return 0, fmt.Errorf("invalid Pocket TTS time transform")
	}
	return value, nil // Inclusive saturation endpoints are preserved, not clamped.
}
func trainingTimeFromNormal(z float32) (float32, error) {
	logit, err := trainingLogit(z)
	if err != nil {
		return 0, err
	}
	return trainingTimeFromLogit(logit)
}
func trainingPairFromNormals(a, b float32) (s, t float32, err error) {
	lo, err := trainingLogit(a)
	if err != nil {
		return 0, 0, err
	}
	hi, err := trainingLogit(b)
	if err != nil {
		return 0, 0, err
	}
	if lo > hi {
		lo, hi = hi, lo
	}
	s, err = trainingTimeFromLogit(lo)
	if err == nil {
		t, err = trainingTimeFromLogit(hi)
	}
	return
}
