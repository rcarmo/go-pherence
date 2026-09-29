package nemotrondiarization

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

const diarizationHeadWidthProjection = 192
const diarizationSpeakers = 8
const diarizationUpsample = 8

// OfflineHead projects final-normalised audio-tower frames, performs the
// subpixel convolution, and classifies frames without speaker-cache updates.
type OfflineHead struct {
	projectionWeight, projectionBias      []float32
	convWeight, convDenseWeight, convBias []float32
	denseWeight, denseBias                []float32
	outputWeight, outputBias              []float32
}

func LoadOfflineHead(file *safetensors.File) (*OfflineHead, error) {
	if file == nil {
		return nil, fmt.Errorf("nil Nemotron diarization checkpoint")
	}
	load := func(name string, dims ...int) ([]float32, error) {
		values, shape, err := file.GetFloat32(name)
		if err != nil {
			return nil, err
		}
		if len(shape) != len(dims) {
			return nil, fmt.Errorf("invalid Nemotron diarization head %s shape %v", name, shape)
		}
		length := 1
		for i, dim := range dims {
			if shape[i] != dim {
				return nil, fmt.Errorf("invalid Nemotron diarization head %s shape %v", name, shape)
			}
			length *= dim
		}
		if len(values) != length {
			return nil, fmt.Errorf("invalid Nemotron diarization head %s length", name)
		}
		for _, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("non-finite Nemotron diarization head %s", name)
			}
		}
		return values, nil
	}
	m := &OfflineHead{}
	var err error
	if m.projectionWeight, err = load("model.proj.weight", diarizationHeadWidthProjection, projectedWidth); err != nil {
		return nil, err
	}
	if m.projectionBias, err = load("model.proj.bias", diarizationHeadWidthProjection); err != nil {
		return nil, err
	}
	if m.convWeight, err = load("model.upsampler.conv.weight", diarizationHeadWidthProjection*diarizationUpsample, diarizationHeadWidthProjection, 3); err != nil {
		return nil, err
	}
	if m.convBias, err = load("model.upsampler.conv.bias", diarizationHeadWidthProjection*diarizationUpsample); err != nil {
		return nil, err
	}
	// Conv1d stores [channel, feature, tap]. The dense window is
	// [tap, feature]; prepare weights once while loading the checkpoint.
	m.convDenseWeight = make([]float32, len(m.convWeight))
	for channel := 0; channel < diarizationHeadWidthProjection*diarizationUpsample; channel++ {
		for feature := 0; feature < diarizationHeadWidthProjection; feature++ {
			for tap := 0; tap < 3; tap++ {
				m.convDenseWeight[(channel*3+tap)*diarizationHeadWidthProjection+feature] = m.convWeight[(channel*diarizationHeadWidthProjection+feature)*3+tap]
			}
		}
	}
	if m.denseWeight, err = load("classifier.dense.weight", diarizationHeadWidthProjection, diarizationHeadWidthProjection); err != nil {
		return nil, err
	}
	if m.denseBias, err = load("classifier.dense.bias", diarizationHeadWidthProjection); err != nil {
		return nil, err
	}
	if m.outputWeight, err = load("classifier.out_proj.weight", diarizationSpeakers, diarizationHeadWidthProjection); err != nil {
		return nil, err
	}
	if m.outputBias, err = load("classifier.out_proj.bias", diarizationSpeakers); err != nil {
		return nil, err
	}
	return m, nil
}

// ForwardOffline accepts a complete [rows,512] final-normalised tower output.
// Its owned result has [rows*8,8] logits before any speaker-cache handling.
func (m *OfflineHead) ForwardOffline(input []float32, rows int) ([]float32, error) {
	_, _, _, logits, err := m.forwardStages(input, rows, false)
	return logits, err
}

// forwardStages retains the channel-major convolution and the separate
// pre-activation upsampled tensor only for pinned intermediate checks.
func (m *OfflineHead) forwardStages(input []float32, rows int, stages bool) (projected, convolved, upsampled, logits []float32, err error) {
	if m == nil || len(m.projectionWeight) != diarizationHeadWidthProjection*projectedWidth || len(m.projectionBias) != diarizationHeadWidthProjection || len(m.convDenseWeight) != diarizationHeadWidthProjection*diarizationUpsample*diarizationHeadWidthProjection*3 || len(m.convBias) != diarizationHeadWidthProjection*diarizationUpsample || len(m.denseWeight) != diarizationHeadWidthProjection*diarizationHeadWidthProjection || len(m.denseBias) != diarizationHeadWidthProjection || len(m.outputWeight) != diarizationSpeakers*diarizationHeadWidthProjection || len(m.outputBias) != diarizationSpeakers {
		return nil, nil, nil, nil, fmt.Errorf("invalid Nemotron diarization head weights")
	}
	if rows < 1 || rows > maxPreparedDiarizationRows || len(input) != rows*projectedWidth {
		return nil, nil, nil, nil, fmt.Errorf("invalid Nemotron diarization head input")
	}
	for _, value := range input {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, nil, nil, nil, fmt.Errorf("non-finite Nemotron diarization head input")
		}
	}
	projected = make([]float32, rows*diarizationHeadWidthProjection)
	if !simd.DenseNTTo(projected, input, m.projectionWeight, rows, diarizationHeadWidthProjection, projectedWidth, 1, projectedWidth, projectedWidth, diarizationHeadWidthProjection) {
		return nil, nil, nil, nil, fmt.Errorf("Nemotron diarization head projection rejected")
	}
	for i := range projected {
		projected[i] += m.projectionBias[i%diarizationHeadWidthProjection]
	}
	// Pack [row, tap, feature] so the checked dense kernel can consume
	// [channel, feature, tap] convolution weights after a one-time reorder.
	channels := diarizationHeadWidthProjection * diarizationUpsample
	windowWidth := diarizationHeadWidthProjection * 3
	windows := make([]float32, rows*windowWidth)
	for row := 0; row < rows; row++ {
		for tap := -1; tap <= 1; tap++ {
			position := row + tap
			if position >= 0 && position < rows {
				copy(windows[row*windowWidth+(tap+1)*diarizationHeadWidthProjection:row*windowWidth+(tap+2)*diarizationHeadWidthProjection], projected[position*diarizationHeadWidthProjection:(position+1)*diarizationHeadWidthProjection])
			}
		}
	}
	upsampled = make([]float32, channels*rows)
	if !simd.DenseNTTo(upsampled, windows, m.convDenseWeight, rows, channels, windowWidth, 1, windowWidth, windowWidth, channels) {
		return nil, nil, nil, nil, fmt.Errorf("Nemotron diarization subpixel convolution rejected")
	}
	for row := 0; row < rows; row++ {
		for channel := 0; channel < channels; channel++ {
			upsampled[row*channels+channel] += m.convBias[channel]
		}
	}
	if stages {
		// Reference Conv1d [channel,time] is diagnostic only.
		convolved = make([]float32, len(upsampled))
		for row := 0; row < rows; row++ {
			for channel := 0; channel < channels; channel++ {
				convolved[channel*rows+row] = upsampled[row*channels+channel]
			}
		}
	}
	frames := rows * diarizationUpsample
	activated := upsampled
	if stages {
		activated = append([]float32(nil), upsampled...)
	}
	for i, value := range activated {
		if value < 0 {
			activated[i] = 0
		}
	}
	hidden := make([]float32, len(upsampled))
	if !simd.DenseNTTo(hidden, activated, m.denseWeight, frames, diarizationHeadWidthProjection, diarizationHeadWidthProjection, 1, diarizationHeadWidthProjection, diarizationHeadWidthProjection, diarizationHeadWidthProjection) {
		return nil, nil, nil, nil, fmt.Errorf("Nemotron diarization head dense rejected")
	}
	for i, value := range hidden {
		hidden[i] = value + m.denseBias[i%diarizationHeadWidthProjection]
		if hidden[i] < 0 {
			hidden[i] = 0
		}
	}
	logits = make([]float32, frames*diarizationSpeakers)
	if !simd.DenseNTTo(logits, hidden, m.outputWeight, frames, diarizationSpeakers, diarizationHeadWidthProjection, 1, diarizationHeadWidthProjection, diarizationHeadWidthProjection, diarizationSpeakers) {
		return nil, nil, nil, nil, fmt.Errorf("Nemotron diarization classifier rejected")
	}
	for i := range logits {
		logits[i] += m.outputBias[i%diarizationSpeakers]
		if math.IsNaN(float64(logits[i])) || math.IsInf(float64(logits[i]), 0) {
			return nil, nil, nil, nil, fmt.Errorf("non-finite Nemotron diarization logits")
		}
	}
	return projected, convolved, upsampled, logits, nil
}
