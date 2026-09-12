// Copyright (c) 2021 Shuai Wang
// Copyright (c) 2022 Zhengyang Chen
// Copyright (c) 2023 Bing Han and CNRS
// Copyright (c) 2026 Rui Carmo (Go adaptation)
// SPDX-License-Identifier: Apache-2.0
// See NOTICE and LICENSE-APACHE-2.0.
package community1

import (
	"context"
	"fmt"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// WeSpeakerResNetConfig selects the ResNet34 BasicBlock [3,4,6,3] topology,
// TSTP pooling and one linear embedding projection (two_emb_layer=false).
// Smaller channel widths are allowed for synthetic testing. Production pyannote
// defaults are BaseChannels32, MelBins80, EmbedDim256; actual checkpoint metadata
// must confirm these settings. Alternate topologies/second embedding layers are
// not silently accepted. MelBins must be divisible by8 (reference stats layout).
type WeSpeakerResNetConfig struct{ BaseChannels, MelBins, EmbedDim int }

// WeSpeakerResNetWeights is the unquantised inference layout. Stage lengths must
// be exactly3,4,6,3. The stem is [base,1,3,3] and affine BatchNorm. Projection is
// [embed,2*(melBins/8)*base*8] plus [embed] bias. No source arrays are retained.
type WeSpeakerResNetWeights struct {
	Stem       []float32
	StemBN     WeSpeakerBN
	Stages     [4][]WeSpeakerBlockWeights
	Projection HeadLinear
}

// WeSpeakerResNet34 owns an immutable stem,16 BasicBlocks and embedding linear.
// It can be shared across calls, with call-local activations and no GPU/model
// globals. Loading a real checkpoint remains separate from this graph API.
type WeSpeakerResNet34 struct {
	cfg        WeSpeakerResNetConfig
	stem       []float32
	stemBN     WeSpeakerBN
	stages     [4][]*WeSpeakerBasicBlock
	projection HeadLinear
}

var weSpeakerStageLengths = [4]int{3, 4, 6, 3}

func validateWeSpeakerResNetConfig(c WeSpeakerResNetConfig) error {
	if c.BaseChannels < 1 || c.BaseChannels > 32 || c.MelBins < 8 || c.MelBins > 80 || c.MelBins%8 != 0 || c.EmbedDim < 1 || c.EmbedDim > 512 {
		return fmt.Errorf("unsupported WeSpeaker ResNet34 geometry")
	}
	return nil
}

// NewWeSpeakerResNet34 constructs only the supported topology. Cancellation
// discards partial construction. Per-block constructors validate shapes, finite
// weights and running statistics; the caller must keep all source slices
// immutable and enforce memory admission, tensor keys/dtypes/provenance first.
func NewWeSpeakerResNet34(ctx context.Context, cfg WeSpeakerResNetConfig, w WeSpeakerResNetWeights) (*WeSpeakerResNet34, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := validateWeSpeakerResNetConfig(cfg); err != nil {
		return nil, err
	}
	for i, n := range weSpeakerStageLengths {
		if len(w.Stages[i]) != n {
			return nil, fmt.Errorf("WeSpeaker stage%d requires%d blocks", i, n)
		}
	}
	pooled := 2 * (cfg.MelBins / 8) * (cfg.BaseChannels * 8)
	type binding struct {
		values   []float32
		size     int
		variance bool
	}
	arrays := []binding{{w.Stem, 9 * cfg.BaseChannels, false}, {w.StemBN.Weight, cfg.BaseChannels, false}, {w.StemBN.Bias, cfg.BaseChannels, false}, {w.StemBN.RunningMean, cfg.BaseChannels, false}, {w.StemBN.RunningVariance, cfg.BaseChannels, true}, {w.Projection.Weight, cfg.EmbedDim * pooled, false}, {w.Projection.Bias, cfg.EmbedDim, false}}
	for _, a := range arrays {
		if len(a.values) != a.size {
			return nil, fmt.Errorf("invalid WeSpeaker stem/projection length")
		}
		if err := finiteBlock(ctx, a.values); err != nil {
			return nil, err
		}
		if a.variance {
			for _, v := range a.values {
				if v < 0 {
					return nil, fmt.Errorf("negative stem running variance")
				}
			}
		}
	}
	m := &WeSpeakerResNet34{cfg: cfg}
	clone := func(src []float32) ([]float32, error) {
		dst := make([]float32, len(src))
		for i := 0; i < len(src); i += 4096 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			copy(dst[i:min(i+4096, len(src))], src[i:min(i+4096, len(src))])
		}
		return dst, nil
	}
	destinations := []*[]float32{&m.stem, &m.stemBN.Weight, &m.stemBN.Bias, &m.stemBN.RunningMean, &m.stemBN.RunningVariance, &m.projection.Weight, &m.projection.Bias}
	for i, a := range arrays {
		owned, err := clone(a.values)
		if err != nil {
			return nil, err
		}
		*destinations[i] = owned
	}
	previous := cfg.BaseChannels
	for stage, n := range weSpeakerStageLengths {
		channels := cfg.BaseChannels * (1 << stage)
		m.stages[stage] = make([]*WeSpeakerBasicBlock, n)
		for index := 0; index < n; index++ {
			stride := 1
			if stage > 0 && index == 0 {
				stride = 2
			}
			block, err := NewWeSpeakerBasicBlock(ctx, WeSpeakerBlockConfig{previous, channels, stride}, w.Stages[stage][index])
			if err != nil {
				return nil, fmt.Errorf("WeSpeaker stage%d block%d: %w", stage, index, err)
			}
			m.stages[stage][index] = block
			previous = channels
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return m, nil
}

// FrameShape validates every intermediate tensor bound before CNN allocation.
// Temporal resolution is ceil(fbankFrames/8), with nominal center frame0 and
// stride8 in Fbank index space (not PCM PTS or a causal dependency guarantee).
func (m *WeSpeakerResNet34) FrameShape(fbankFrames int) (CHWShape, error) {
	if m == nil {
		return CHWShape{}, fmt.Errorf("nil WeSpeaker ResNet")
	}
	if err := validateWeSpeakerResNetConfig(m.cfg); err != nil {
		return CHWShape{}, err
	}
	shape := CHWShape{m.cfg.BaseChannels, m.cfg.MelBins, fbankFrames}
	if _, err := chwElements(shape); err != nil {
		return CHWShape{}, err
	}
	if len(m.stem) != 9*m.cfg.BaseChannels {
		return CHWShape{}, fmt.Errorf("uninitialised WeSpeaker ResNet")
	}
	for stage, n := range weSpeakerStageLengths {
		if len(m.stages[stage]) != n {
			return CHWShape{}, fmt.Errorf("invalid WeSpeaker stage count")
		}
		for _, block := range m.stages[stage] {
			var err error
			shape, err = block.OutputShape(shape)
			if err != nil {
				return CHWShape{}, err
			}
		}
	}
	return shape, nil
}

// WeSpeakerResNetObserver exposes transient read-only CHW values at stem and
// all16 block exits. stage=-1,block=-1 denotes stem; others are zero-based.
// Copy before retention. A callback remains visible if a later stage fails.
type WeSpeakerResNetObserver func(stage, block int, shape CHWShape, values []float32)

// ForwardFrames evaluates the full stem/[3,4,6,3] trunk once on frame-major
// [frames,melBins] Fbank input and returns owned CHW [8*base,melBins/8,ceil(T/8)].
// It does not compute Fbank, identify masks, load weights or use SincNet. Input
// must stay immutable; independent calls can share the model. Tensor bounds
// apply to intermediates, not a total allocation/latency admission guarantee.
func (m *WeSpeakerResNet34) ForwardFrames(ctx context.Context, fbank []float32, frames int, mode WeSpeakerBlockMode) ([]float32, CHWShape, error) {
	return m.ForwardFramesObserved(ctx, fbank, frames, mode, nil)
}
func (m *WeSpeakerResNet34) ForwardFramesObserved(ctx context.Context, fbank []float32, frames int, mode WeSpeakerBlockMode, observe WeSpeakerResNetObserver) ([]float32, CHWShape, error) {
	fail := func(err error) ([]float32, CHWShape, error) { return nil, CHWShape{}, err }
	if err := ctx.Err(); err != nil {
		return fail(err)
	}
	final, err := m.FrameShape(frames)
	if err != nil {
		return fail(err)
	}
	if len(fbank) != frames*m.cfg.MelBins || (mode != WeSpeakerBlockScalar && mode != WeSpeakerBlockSIMD) {
		return fail(fmt.Errorf("invalid WeSpeaker Fbank layout/mode"))
	}
	if err := finiteBlock(ctx, fbank); err != nil {
		return fail(err)
	}
	input := make([]float32, len(fbank))
	for t := 0; t < frames; t++ {
		if err := ctx.Err(); err != nil {
			return fail(err)
		}
		for f := 0; f < m.cfg.MelBins; f++ {
			input[f*frames+t] = fbank[t*m.cfg.MelBins+f]
		}
	}
	shape := CHWShape{m.cfg.BaseChannels, m.cfg.MelBins, frames}
	out, err := weSpeakerBlockConv(ctx, input, CHWShape{1, m.cfg.MelBins, frames}, shape, m.stem, 3, 1, 1, mode)
	if err != nil {
		return fail(err)
	}
	if err := weSpeakerBlockBN(ctx, out, shape, m.stemBN); err != nil {
		return fail(err)
	}
	if err := blockReLU(ctx, out); err != nil {
		return fail(err)
	}
	if observe != nil {
		observe(-1, -1, shape, out)
	}
	if err := ctx.Err(); err != nil {
		return fail(err)
	}
	for stage, blocks := range m.stages {
		for index, block := range blocks {
			out, shape, err = block.Forward(ctx, out, shape, mode)
			if err != nil {
				return fail(fmt.Errorf("WeSpeaker stage%d block%d: %w", stage, index, err))
			}
			if observe != nil {
				observe(stage, index, shape, out)
			}
			if err := ctx.Err(); err != nil {
				return fail(err)
			}
		}
	}
	if shape != final {
		return fail(fmt.Errorf("WeSpeaker frame shape mismatch"))
	}
	return out, shape, nil
}

// WeSpeakerEmbeddingResult holds raw, unnormalised speaker-major [rows,dim]
// embeddings and mask support AFTER nearest resizing. Empty masks numerically
// yield the projection bias; this is NOT an admissible speaker embedding by
// default. Validity/minimum speech admission must be determined by the caller.
type WeSpeakerEmbeddingResult struct {
	Embeddings, WeightSum []float32
	NonzeroFrames         []int
}

// WeSpeakerEmbeddingObserver sees transient read-only pooled statistics then
// raw projected embeddings, [rows,features]. No batch normalization/second
// linear layer/L2 normalisation is applied in this supported one-projection path.
type WeSpeakerEmbeddingObserver func(stage string, rows, features int, values []float32)

func (m *WeSpeakerResNet34) validateEmbeddingInput(ctx context.Context, shape CHWShape, masks []float32, speakers, maskFrames int) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if m == nil {
		return fmt.Errorf("nil WeSpeaker ResNet")
	}
	if err := validateWeSpeakerResNetConfig(m.cfg); err != nil {
		return err
	}
	if _, err := chwElements(shape); err != nil {
		return err
	}
	if shape.Channels != m.cfg.BaseChannels*8 || shape.Frequency != m.cfg.MelBins/8 {
		return fmt.Errorf("invalid WeSpeaker embedding CHW layout")
	}
	pooled := 2 * shape.Channels * shape.Frequency
	if len(m.projection.Weight) != m.cfg.EmbedDim*pooled || len(m.projection.Bias) != m.cfg.EmbedDim {
		return fmt.Errorf("uninitialised WeSpeaker projection")
	}
	if masks == nil {
		if speakers != 0 || maskFrames != 0 || shape.Frames < 2 {
			return fmt.Errorf("unweighted embedding requires at least two CNN frames and no mask geometry")
		}
	} else if speakers < 1 || speakers > 8 || maskFrames < 1 || maskFrames > 4096 || len(masks) != speakers*maskFrames {
		return fmt.Errorf("invalid embedding masks")
	}
	return finitePool(ctx, masks, true)
}

// ForwardEmbedding reuses a previously computed CHW trunk for any mask set.
// It never recomputes the CNN. Frame source/model consistency is the caller's
// responsibility; changing model weights invalidates reusable trunk features.
func (m *WeSpeakerResNet34) ForwardEmbedding(ctx context.Context, features []float32, shape CHWShape, masks []float32, speakers, maskFrames int, mode WeSpeakerBlockMode) (*WeSpeakerEmbeddingResult, error) {
	return m.ForwardEmbeddingObserved(ctx, features, shape, masks, speakers, maskFrames, mode, nil)
}
func (m *WeSpeakerResNet34) ForwardEmbeddingObserved(ctx context.Context, features []float32, shape CHWShape, masks []float32, speakers, maskFrames int, mode WeSpeakerBlockMode, observe WeSpeakerEmbeddingObserver) (*WeSpeakerEmbeddingResult, error) {
	if err := m.validateEmbeddingInput(ctx, shape, masks, speakers, maskFrames); err != nil {
		return nil, err
	}
	if mode != WeSpeakerBlockScalar && mode != WeSpeakerBlockSIMD {
		return nil, fmt.Errorf("invalid WeSpeaker embedding mode")
	}
	// StatsPool validates exact feature length and finiteness before reductions
	// or allocation; do not repeat a full feature scan at this wrapper boundary.
	pooled, err := StatsPool(ctx, features, masks, StatsPoolConfig{Features: shape.Channels * shape.Frequency, Frames: shape.Frames, Speakers: speakers, MaskFrames: maskFrames})
	if err != nil {
		return nil, err
	}
	rows := max(1, speakers)
	cols := 2 * shape.Channels * shape.Frequency
	if observe != nil {
		observe("stats", rows, cols, pooled.Statistics)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	output := make([]float32, rows*m.cfg.EmbedDim)
	for row := 0; row < rows; row++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		dst := output[row*m.cfg.EmbedDim : (row+1)*m.cfg.EmbedDim]
		x := pooled.Statistics[row*cols : (row+1)*cols]
		if mode == WeSpeakerBlockSIMD {
			if !simd.GemvRows(dst, x, m.projection.Weight, m.cfg.EmbedDim, cols) {
				return nil, fmt.Errorf("WeSpeaker projection shape rejected")
			}
		} else {
			for out := range dst {
				if out%32 == 0 {
					if err := ctx.Err(); err != nil {
						return nil, err
					}
				}
				var sum float32
				for i, v := range x {
					sum += v * m.projection.Weight[out*cols+i]
				}
				dst[out] = sum
			}
		}
		for i := range dst {
			dst[i] += m.projection.Bias[i]
		}
	}
	if err := finiteBlock(ctx, output); err != nil {
		return nil, err
	}
	if observe != nil {
		observe("embedding", rows, m.cfg.EmbedDim, output)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return &WeSpeakerEmbeddingResult{Embeddings: output, WeightSum: pooled.WeightSum, NonzeroFrames: pooled.NonzeroFrames}, nil
}

// Forward combines one shared trunk pass with pooling/projection for all masks.
// Mask shape/value/admission errors are checked before expensive CNN evaluation.
// This entry accepts Fbank features only, not raw PCM. No SincNet dependency.
// No partial result escapes; a running GEMV/allocation/observer is uninterruptible.
func (m *WeSpeakerResNet34) Forward(ctx context.Context, fbank []float32, frames int, masks []float32, speakers, maskFrames int, mode WeSpeakerBlockMode) (*WeSpeakerEmbeddingResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	shape, err := m.FrameShape(frames)
	if err != nil {
		return nil, err
	}
	if err := m.validateEmbeddingInput(ctx, shape, masks, speakers, maskFrames); err != nil {
		return nil, err
	}
	features, shape, err := m.ForwardFrames(ctx, fbank, frames, mode)
	if err != nil {
		return nil, err
	}
	return m.ForwardEmbedding(ctx, features, shape, masks, speakers, maskFrames, mode)
}
