// Copyright (c) 2026 Rui Carmo
// SPDX-License-Identifier: MIT
// PyanNet tensor topology follows the pinned source attributed in NOTICE.
package community1

import (
	"context"
	"fmt"
	"math"
	"sort"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// SegmentationLoadConfig is explicit checkpoint metadata, not inferred from
// parameter sizes/defaults. Only mono16k, fixed 80->60->60 SincNet and powerset
// PyanNet are supported. LSTM input must be60 and head input must equal the
// recurrent width. SplitLSTM selects ModuleList keys lstm.i.*_l0 instead of
// monolithic lstm.*_li. Dropout is disabled (evaluation), no projected LSTM.
type SegmentationLoadConfig struct {
	SincNetStride int
	LSTM          LSTMConfig
	Head          HeadConfig
	SplitLSTM     bool
}

// SegmentationTensorSource exposes one immutable metadata/offset address space
// and widened tensor values. The source owns header/file bounds; the caller
// closes it. GetFloat32 buffers can be reused by the next call. Synchronous
// reads are not interrupted. safetensors.File implements this interface.
type SegmentationTensorSource interface {
	TensorInfos() map[string]safetensors.TensorInfo
	GetFloat32(string) ([]float32, []int, error)
}

// SegmentationCheckpoint owns checked weights and exposes only feature input.
// ForwardFeatures runs recurrent/head components on explicit frame-major
// [frames,60] features. The separate ExperimentalSegmentation wrapper composes
// PCM inference with lowered filters, but retains strict boundary failures.
// Neither type alone qualifies global diarization or other trained checkpoints.
type SegmentationCheckpoint struct {
	cfg       SegmentationLoadConfig
	sincnet   SincNetWeights
	recurrent *LSTM
	head      *SegmentationHead
}

// Grid reports the nominal SincNet convolution geometry only. Whole-window
// instance normalization makes its receptive field unsuitable as a streaming
// dependency bound. It does not execute or qualify the frontend.
func (m *SegmentationCheckpoint) Grid(samples int) (SincNetGrid, error) {
	if m == nil {
		return SincNetGrid{}, fmt.Errorf("nil segmentation checkpoint")
	}
	return sincNetGrid(samples, m.cfg.SincNetStride)
}
func (m *SegmentationCheckpoint) ForwardFeatures(ctx context.Context, input []float32, frames int, lstmMode LSTMMode, headMode HeadMode) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if m == nil || m.recurrent == nil || m.head == nil {
		return nil, fmt.Errorf("invalid segmentation checkpoint")
	}
	if (lstmMode != LSTMScalar && lstmMode != LSTMSIMD) || (headMode != HeadScalar && headMode != HeadSIMD) {
		return nil, fmt.Errorf("invalid segmentation feature mode")
	}
	sequence, err := m.recurrent.Forward(ctx, input, frames, nil, nil, lstmMode)
	if err != nil {
		return nil, err
	}
	return m.head.Forward(ctx, sequence.Output, frames, headMode)
}

type segmentationBinding struct {
	name   string
	shape  []int
	target *[]float32
	fixed  string
}

func segmentationBindings(cfg SegmentationLoadConfig) (*SegmentationCheckpoint, []segmentationBinding, error) {
	if cfg.SincNetStride < 1 || cfg.SincNetStride > 10 || cfg.LSTM.InputSize != 60 {
		return nil, nil, fmt.Errorf("unsupported segmentation frontend geometry")
	}
	if err := checkLSTMConfig(cfg.LSTM); err != nil {
		return nil, nil, err
	}
	if cfg.Head.InputSize != cfg.LSTM.HiddenSize*lstmDirections(cfg.LSTM) {
		return nil, nil, fmt.Errorf("segmentation recurrent/head width mismatch")
	}
	classes, err := checkHeadConfig(cfg.Head)
	if err != nil {
		return nil, nil, err
	}
	m := &SegmentationCheckpoint{cfg: cfg, recurrent: &LSTM{cfg: cfg.LSTM, layers: make([]LSTMLayer, cfg.LSTM.NumLayers)}, head: &SegmentationHead{cfg: cfg.Head, classes: classes, layers: make([]HeadLinear, cfg.Head.NumLayers)}}
	var bindings []segmentationBinding
	add := func(name string, target *[]float32, shape ...int) {
		bindings = append(bindings, segmentationBinding{name, shape, target, ""})
	}
	add("sincnet.wav_norm1d.weight", &m.sincnet.WaveNorm.Weight, 1)
	add("sincnet.wav_norm1d.bias", &m.sincnet.WaveNorm.Bias, 1)
	const base = "sincnet.conv1d.0.filterbank."
	add(base+"low_hz_", &m.sincnet.LowHz, 40, 1)
	add(base+"band_hz_", &m.sincnet.BandHz, 40, 1)
	bindings = append(bindings, segmentationBinding{base + "window_", []int{125}, nil, "window"}, segmentationBinding{base + "n_", []int{1, 125}, nil, "time"})
	for i, in := range []int{80, 60} {
		add(fmt.Sprintf("sincnet.conv1d.%d.weight", i+1), &m.sincnet.Conv[i].Weight, 60, in, 5)
		add(fmt.Sprintf("sincnet.conv1d.%d.bias", i+1), &m.sincnet.Conv[i].Bias, 60)
	}
	for i, width := range []int{80, 60, 60} {
		add(fmt.Sprintf("sincnet.norm1d.%d.weight", i), &m.sincnet.Norm[i].Weight, width)
		add(fmt.Sprintf("sincnet.norm1d.%d.bias", i), &m.sincnet.Norm[i].Bias, width)
	}
	for i := 0; i < cfg.LSTM.NumLayers; i++ {
		width := 60
		if i > 0 {
			width = cfg.LSTM.HiddenSize * lstmDirections(cfg.LSTM)
		}
		for dir := 0; dir < lstmDirections(cfg.LSTM); dir++ {
			weights := &m.recurrent.layers[i].Forward
			suffix := ""
			if dir == 1 {
				weights = &m.recurrent.layers[i].Reverse
				suffix = "_reverse"
			}
			name := "lstm."
			layer := i
			if cfg.SplitLSTM {
				name = fmt.Sprintf("lstm.%d.", i)
				layer = 0
			}
			tail := fmt.Sprintf("_l%d%s", layer, suffix)
			h := cfg.LSTM.HiddenSize
			add(name+"weight_ih"+tail, &weights.WeightIH, 4*h, width)
			add(name+"weight_hh"+tail, &weights.WeightHH, 4*h, h)
			add(name+"bias_ih"+tail, &weights.BiasIH, 4*h)
			add(name+"bias_hh"+tail, &weights.BiasHH, 4*h)
		}
	}
	width := cfg.Head.InputSize
	for i := range m.head.layers {
		add(fmt.Sprintf("linear.%d.weight", i), &m.head.layers[i].Weight, cfg.Head.HiddenSize, width)
		add(fmt.Sprintf("linear.%d.bias", i), &m.head.layers[i].Bias, cfg.Head.HiddenSize)
		width = cfg.Head.HiddenSize
	}
	add("classifier.weight", &m.head.classifier.Weight, classes, width)
	add("classifier.bias", &m.head.classifier.Bias, classes)
	return m, bindings, nil
}

// LoadSegmentationSource validates the EXACT raw PyanNet state_dict inventory,
// shapes, byte extents and non-overlap BEFORE any payload read. No prefixes,
// training counters, extra buffers or ignored keys. Learned tensors accept
// F32/F16/BF16; fixed filterbank window_/n_ require F32 and values within one
// float32 ULP of the fixed16k/251-point definitions. They are verified and then
// discarded, not silently used as a different frontend. Narrowed fixed buffers
// are unsupported. All learned tensors are copied immediately, finite, and
// low/band cutoffs must define positive bandwidth below Nyquist.
//
// The weight-free private skeleton avoids constructor-wide second cloning.
// Metadata/file bounds belong to source; caller admits retained model memory.
// No unsafe checkpoint unpickling/config inference/training/GPU/PCM execution.
func LoadSegmentationSource(ctx context.Context, source SegmentationTensorSource, cfg SegmentationLoadConfig) (*SegmentationCheckpoint, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if source == nil {
		return nil, fmt.Errorf("nil segmentation tensor source")
	}
	model, bindings, err := segmentationBindings(cfg)
	if err != nil {
		return nil, err
	}
	infos := source.TensorInfos()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(infos) != len(bindings) {
		return nil, fmt.Errorf("unexpected segmentation tensor inventory")
	}
	type extent struct{ start, end int }
	extents := make([]extent, 0, len(bindings))
	for _, b := range bindings {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		info, ok := infos[b.name]
		if !ok || !weSpeakerSameShape(info.Shape, b.shape) {
			return nil, fmt.Errorf("missing/mis-shaped segmentation tensor %s", b.name)
		}
		width := 4
		switch info.DType {
		case "F32":
		case "F16", "BF16":
			width = 2
		default:
			return nil, fmt.Errorf("unsupported segmentation dtype %s for %s", info.DType, b.name)
		}
		if b.fixed != "" && info.DType != "F32" {
			return nil, fmt.Errorf("fixed segmentation buffer requires F32 %s", b.name)
		}
		count := 1
		for _, dim := range b.shape {
			count *= dim
		}
		start, end := info.DataOffsets[0], info.DataOffsets[1]
		if start < 0 || end < start || end-start != count*width {
			return nil, fmt.Errorf("invalid segmentation extent %s", b.name)
		}
		extents = append(extents, extent{start, end})
	}
	sort.Slice(extents, func(i, j int) bool { return extents[i].start < extents[j].start })
	for i := 1; i < len(extents); i++ {
		if extents[i].start < extents[i-1].end {
			return nil, fmt.Errorf("overlapping segmentation extents")
		}
	}
	for _, b := range bindings {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		values, shape, err := source.GetFloat32(b.name)
		if err != nil {
			return nil, fmt.Errorf("segmentation %s: %w", b.name, err)
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		count := 1
		for _, d := range b.shape {
			count *= d
		}
		if !weSpeakerSameShape(shape, b.shape) || len(values) != count {
			return nil, fmt.Errorf("changed segmentation tensor %s", b.name)
		}
		if err := finiteLSTM(ctx, values); err != nil {
			return nil, err
		}
		if b.fixed != "" {
			for i, v := range values {
				expected := float32(.54 - .46*math.Cos(2*math.Pi*float64(i)/250))
				if b.fixed == "time" {
					expected = float32(2*math.Pi) * float32(float32(i-125)/16000)
				}
				if v < math.Nextafter32(expected, float32(math.Inf(-1))) || v > math.Nextafter32(expected, float32(math.Inf(1))) {
					return nil, fmt.Errorf("noncanonical segmentation buffer %s", b.name)
				}
			}
			continue
		}
		owned := make([]float32, count)
		for start := 0; start < count; start += 4096 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			copy(owned[start:min(start+4096, count)], values[start:min(start+4096, count)])
		}
		*b.target = owned
	}
	for i, raw := range model.sincnet.LowHz {
		low := float32(50) + float32(math.Abs(float64(raw)))
		high := min(low+50+float32(math.Abs(float64(model.sincnet.BandHz[i]))), float32(8000))
		if high <= low {
			return nil, fmt.Errorf("degenerate segmentation sinc band %d", i)
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return model, nil
}
