package pockettts

import (
	"fmt"

	"github.com/rcarmo/go-pherence/internal/checked"
)

// TrainingStepDraws holds explicit active-row inputs, without sample padding.
// Noise has validFrames*channels elements; each time slice has validFrames.
// The caller supplies the draws: this type defines no stochastic distribution,
// Gaussian scale, objective selection or PyTorch RNG-sequence equivalence.
type TrainingStepDraws struct {
	Noise              []float32
	DiagonalTime       []float32
	DistillS, DistillT []float32
}

// TrainingSampleLimits bounds physical rows and full-frame noise elements.
// Both ceilings must be positive. They bound work, not trainer memory or RSS.
type TrainingSampleLimits struct {
	MaxFrames, MaxNoiseElements int
}

// PrepareTrainingSamplesInto validates explicit draws before modifying any
// caller-owned destination. Destination lengths must match the full-frame
// TrainingStepSamples layout exactly; larger arenas must supply bounded views.
// Active values are copied unchanged, the mask becomes a valid prefix, and
// only sample padding is zeroed. It does not append an EOS frame or change
// latent targets, noise scale, loss selection or gradient semantics.
//
// Writable float views must not overlap each other or any input. Read-only
// inputs may overlap each other. The caller supplies normal typed bool/float
// storage: unsafe-forged cross-type aliases are unsupported. No concurrent
// mutation is supported. No slices are retained or destination storage allocated.
// This helper consumes no RNG and never prepares or commits a run-state row.
func PrepareTrainingSamplesInto(dst TrainingStepSamples, frames, validFrames, channels int, draws TrainingStepDraws, limits TrainingSampleLimits) error {
	if limits.MaxFrames <= 0 || limits.MaxNoiseElements <= 0 || frames <= 0 || frames > limits.MaxFrames || validFrames <= 0 || validFrames > frames || channels <= 0 {
		return fmt.Errorf("invalid Pocket TTS sample geometry or limits")
	}
	fullElements, ok := checked.MulInt(frames, channels)
	if !ok || fullElements > limits.MaxNoiseElements {
		return fmt.Errorf("Pocket TTS sample noise element bound exceeded")
	}
	// Positive channels and validFrames<=frames make every active/time view
	// no larger than the checked full-frame noise view.
	activeElements := validFrames * channels
	if _, ok := checked.MulInt(fullElements, 4); !ok {
		return fmt.Errorf("Pocket TTS sample byte range overflows")
	}
	if len(dst.Mask) != frames || len(dst.Noise) != fullElements || len(dst.DiagonalTime) != frames || len(dst.DistillS) != frames || len(dst.DistillT) != frames || len(draws.Noise) != activeElements || len(draws.DiagonalTime) != validFrames || len(draws.DistillS) != validFrames || len(draws.DistillT) != validFrames {
		return fmt.Errorf("invalid Pocket TTS explicit sample buffer shape")
	}
	outputs := [4][]float32{dst.Noise, dst.DiagonalTime, dst.DistillS, dst.DistillT}
	inputs := [4][]float32{draws.Noise, draws.DiagonalTime, draws.DistillS, draws.DistillT}
	for i, output := range outputs {
		for _, other := range outputs[i+1:] {
			if f32SlicesOverlap(output, other) {
				return fmt.Errorf("Pocket TTS sample destinations overlap")
			}
		}
		for _, input := range inputs {
			if f32SlicesOverlap(output, input) {
				return fmt.Errorf("Pocket TTS sample destination overlaps input")
			}
		}
	}
	for _, value := range draws.Noise {
		if !isFinite(value) {
			return fmt.Errorf("Pocket TTS explicit sample noise is non-finite")
		}
	}
	for row := 0; row < validFrames; row++ {
		diagonal, s, t := draws.DiagonalTime[row], draws.DistillS[row], draws.DistillT[row]
		if !isFinite(diagonal) || !isFinite(s) || !isFinite(t) || diagonal < 0 || diagonal > 1 || s < 0 || t > 1 || s > t {
			return fmt.Errorf("Pocket TTS explicit sample time is invalid")
		}
	}
	// All fallible checks precede mutation. Copying does not transform draws.
	copy(dst.Noise, draws.Noise)
	clear(dst.Noise[activeElements:])
	for i, output := range outputs[1:] {
		copy(output, inputs[i+1])
		clear(output[validFrames:])
	}
	for row := range dst.Mask {
		dst.Mask[row] = row < validFrames
	}
	return nil
}
