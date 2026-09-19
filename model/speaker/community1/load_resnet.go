// Copyright (c) 2026 Rui Carmo
// SPDX-License-Identifier: Apache-2.0
// WeSpeaker tensor topology follows the reference attributed in NOTICE.
package community1

import (
	"context"
	"fmt"
	"math"
	"sort"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// WeSpeakerTensorSource exposes all metadata before payload access.
// safetensors.File implements this interface. Caller owns the source and its
// Close; it must remain immutable during loading. Byte extents must share ONE
// address space (sharded sources need normalised offsets). Source owns header/
// file bounds; synchronous GetFloat32 cannot be interrupted in progress.
// Source buffers may be reused by the next call: the loader copies immediately.
type WeSpeakerTensorSource interface {
	TensorInfos() map[string]safetensors.TensorInfo
	GetFloat32(name string) ([]float32, []int, error)
}

type weSpeakerTensorBinding struct {
	name     string
	shape    []int
	target   *[]float32
	variance bool
}

// Build a weight-free skeleton and exact inference bindings. Dimensions are
// used only after validated config bounds. No large weight arrays are allocated.
func weSpeakerResNetBindings(cfg WeSpeakerResNetConfig, prefix string) (*WeSpeakerResNet34, []weSpeakerTensorBinding, map[string]bool) {
	if prefix != "" {
		prefix += "."
	}
	m := &WeSpeakerResNet34{cfg: cfg}
	var bindings []weSpeakerTensorBinding
	counters := make(map[string]bool)
	add := func(name string, target *[]float32, shape ...int) {
		bindings = append(bindings, weSpeakerTensorBinding{name: prefix + name, shape: shape, target: target})
	}
	bn := func(name string, b *WeSpeakerBN, channels int) {
		add(name+".weight", &b.Weight, channels)
		add(name+".bias", &b.Bias, channels)
		add(name+".running_mean", &b.RunningMean, channels)
		add(name+".running_var", &b.RunningVariance, channels)
		bindings[len(bindings)-1].variance = true
		counters[prefix+name+".num_batches_tracked"] = true
	}
	add("conv1.weight", &m.stem, cfg.BaseChannels, 1, 3, 3)
	bn("bn1", &m.stemBN, cfg.BaseChannels)
	previous := cfg.BaseChannels
	for stage, count := range weSpeakerStageLengths {
		channels := cfg.BaseChannels * (1 << stage)
		m.stages[stage] = make([]*WeSpeakerBasicBlock, count)
		for index := 0; index < count; index++ {
			stride := 1
			if stage > 0 && index == 0 {
				stride = 2
			}
			block := &WeSpeakerBasicBlock{cfg: WeSpeakerBlockConfig{previous, channels, stride}}
			m.stages[stage][index] = block
			w := &block.weights
			name := fmt.Sprintf("layer%d.%d", stage+1, index)
			add(name+".conv1.weight", &w.Conv1, channels, previous, 3, 3)
			bn(name+".bn1", &w.BN1, channels)
			add(name+".conv2.weight", &w.Conv2, channels, channels, 3, 3)
			bn(name+".bn2", &w.BN2, channels)
			if blockNeedsShortcut(block.cfg) {
				add(name+".shortcut.0.weight", &w.Shortcut, channels, previous, 1, 1)
				bn(name+".shortcut.1", &w.ShortcutBN, channels)
			}
			previous = channels
		}
	}
	pooled := 2 * (cfg.MelBins / 8) * cfg.BaseChannels * 8
	add("seg_1.weight", &m.projection.Weight, cfg.EmbedDim, pooled)
	add("seg_1.bias", &m.projection.Bias, cfg.EmbedDim)
	return m, bindings, counters
}

// LoadWeSpeakerResNetSource loads the supported ResNet34/single-projection
// graph from unquantised tensors. prefix is explicitly "" (raw resnet.state_dict)
// or "resnet" (pyannote wrapper). No prefix auto-detection or ignored unknown
// tensor keys. Every required name/rank/dimension, F32/F16/BF16 dtype, byte extent
// and non-overlap is checked BEFORE the first payload read. Optional BatchNorm
// num_batches_tracked tensors must be I64 scalars with eight-byte extents; their
// training-only values are not read or used. All other integer tensors fail.
//
// Floating weights are finite, running variances nonnegative, and each tensor
// is copied into model-owned memory before reading the next. No partial model
// escapes an error/cancellation. This avoids a second constructor-wide clone;
// peak memory includes retained weights plus one source tensor and its copy.
// Caller owns model-memory admission and cfg/checkpoint provenance. The loader
// does not open files, download/unpickle checkpoints, parse model hyperparameters,
// start inference, prepack globals or initialise a GPU. Quantised/sharded layouts
// and two_emb_layer=true are unsupported. Synthetic schema/parity tests do not
// qualify a trained checkpoint or authorize production deployment.
func LoadWeSpeakerResNetSource(ctx context.Context, source WeSpeakerTensorSource, cfg WeSpeakerResNetConfig, prefix string) (*WeSpeakerResNet34, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if source == nil {
		return nil, fmt.Errorf("nil WeSpeaker tensor source")
	}
	if prefix != "" && prefix != "resnet" {
		return nil, fmt.Errorf("WeSpeaker prefix must be empty or resnet")
	}
	if err := validateWeSpeakerResNetConfig(cfg); err != nil {
		return nil, err
	}
	model, bindings, counters := weSpeakerResNetBindings(cfg, prefix)
	infos := source.TensorInfos()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(infos) < len(bindings) || len(infos) > len(bindings)+len(counters) {
		return nil, fmt.Errorf("unexpected WeSpeaker tensor inventory size")
	}
	expected := make(map[string]weSpeakerTensorBinding, len(bindings))
	for _, b := range bindings {
		expected[b.name] = b
	}
	names := make([]string, 0, len(infos))
	for name := range infos {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, b := range bindings {
		if _, ok := infos[b.name]; !ok {
			return nil, fmt.Errorf("missing WeSpeaker tensor %s", b.name)
		}
	}
	type extent struct {
		start, end int
		name       string
	}
	extents := make([]extent, 0, len(names))
	for _, name := range names {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		info := infos[name]
		bytes := 0
		elements := 1
		b, required := expected[name]
		if required {
			if !weSpeakerSameShape(info.Shape, b.shape) {
				return nil, fmt.Errorf("WeSpeaker tensor %s shape %v, want %v", name, info.Shape, b.shape)
			}
			for _, dim := range b.shape {
				elements *= dim
			}
			switch info.DType {
			case "F32":
				bytes = 4
			case "F16", "BF16":
				bytes = 2
			default:
				return nil, fmt.Errorf("unsupported WeSpeaker tensor %s dtype %s", name, info.DType)
			}
		} else {
			if !counters[name] {
				return nil, fmt.Errorf("unknown WeSpeaker tensor %s", name)
			}
			if len(info.Shape) != 0 || info.DType != "I64" {
				return nil, fmt.Errorf("invalid training-counter metadata %s", name)
			}
			bytes = 8
		}
		start, end := info.DataOffsets[0], info.DataOffsets[1]
		if start < 0 || end < start || end-start != elements*bytes {
			return nil, fmt.Errorf("invalid WeSpeaker tensor %s byte extent", name)
		}
		extents = append(extents, extent{start, end, name})
	}
	sort.Slice(extents, func(i, j int) bool { return extents[i].start < extents[j].start })
	for i := 1; i < len(extents); i++ {
		if extents[i].start < extents[i-1].end {
			return nil, fmt.Errorf("overlapping WeSpeaker tensors %s / %s", extents[i-1].name, extents[i].name)
		}
	}
	for _, b := range bindings {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		values, shape, err := source.GetFloat32(b.name)
		if err != nil {
			return nil, fmt.Errorf("load WeSpeaker tensor %s: %w", b.name, err)
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		count := 1
		for _, dim := range b.shape {
			count *= dim
		}
		if !weSpeakerSameShape(shape, b.shape) || len(values) != count {
			return nil, fmt.Errorf("WeSpeaker tensor %s changed shape/length during load", b.name)
		}
		owned := make([]float32, count)
		for start := 0; start < count; start += 4096 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			end := min(start+4096, count)
			for _, value := range values[start:end] {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || (b.variance && value < 0) {
					return nil, fmt.Errorf("invalid finite/variance values in WeSpeaker tensor %s", b.name)
				}
			}
			copy(owned[start:end], values[start:end])
		}
		*b.target = owned
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if _, err := model.FrameShape(1); err != nil {
		return nil, err
	}
	return model, nil
}

func weSpeakerSameShape(got, want []int) bool {
	if len(got) != len(want) {
		return false
	}
	for i, d := range got {
		if d != want[i] {
			return false
		}
	}
	return true
}
