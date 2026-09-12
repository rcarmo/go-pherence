// Copyright (c) 2021 Shuai Wang
// Copyright (c) 2022 Zhengyang Chen
// Copyright (c) 2023 Bing Han and CNRS
// Copyright (c) 2026 Rui Carmo (Go adaptation)
// SPDX-License-Identifier: Apache-2.0
// See NOTICE and LICENSE-APACHE-2.0 for the pinned WeSpeaker reference.
package community1

import (
	"context"
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// WeSpeakerBN is inference BatchNorm2d with epsilon1e-5. All vectors have one
// value per channel. RunningMean/RunningVariance are checkpoint buffers, not
// statistics from the current window. No training updates/folding are performed.
type WeSpeakerBN struct{ Weight, Bias, RunningMean, RunningVariance []float32 }

// WeSpeakerBlockWeights follows the ResNet34 BasicBlock only: two bias-free
// convolutions [out,in,3,3], [out,out,3,3] and their BatchNorm parameters. The
// optional shortcut is [out,in,1,1]+BN, required for stride2 or changed channels.
// Shortcut arrays must be absent for identity blocks. Bottleneck/SE blocks,
// groups, dilation and alternate epsilon are not supported by this component.
type WeSpeakerBlockWeights struct {
	Conv1, Conv2 []float32
	BN1, BN2     WeSpeakerBN
	Shortcut     []float32
	ShortcutBN   WeSpeakerBN
}

type WeSpeakerBlockConfig struct{ InChannels, OutChannels, Stride int }

// WeSpeakerBasicBlock owns finite immutable copies of all parameters. The source
// checkpoint/config must identify this architecture; this constructor does not
// infer it from tensor lengths. Per-call scratch is independent and no global
// model state is changed. This is not a complete ResNet/embedding pipeline.
type WeSpeakerBasicBlock struct {
	cfg     WeSpeakerBlockConfig
	weights WeSpeakerBlockWeights
}

type WeSpeakerBlockMode uint8

const (
	WeSpeakerBlockScalar WeSpeakerBlockMode = iota
	// Reuses existing checked Plan9 GemvRows on model-specific convolution
	// patches. No new assembly or high-performance convolution claim.
	WeSpeakerBlockSIMD
)

// CHWShape describes one batch in channel-major [channels,frequency,time] order.
// Flatten frequency with channels (keeping time last) for StatsPool later.
// Limits here bound one block call, not a service memory/latency guarantee.
type CHWShape struct{ Channels, Frequency, Frames int }

func checkWeSpeakerBlock(c WeSpeakerBlockConfig) error {
	if c.InChannels < 1 || c.InChannels > 256 || c.OutChannels < 1 || c.OutChannels > 256 || (c.Stride != 1 && c.Stride != 2) {
		return fmt.Errorf("unsupported WeSpeaker basic block geometry")
	}
	return nil
}
func blockNeedsShortcut(c WeSpeakerBlockConfig) bool {
	return c.Stride != 1 || c.InChannels != c.OutChannels
}
func chwElements(s CHWShape) (int, error) {
	if s.Channels < 1 || s.Channels > 256 || s.Frequency < 1 || s.Frequency > 80 || s.Frames < 1 || s.Frames > 4096 {
		return 0, fmt.Errorf("invalid WeSpeaker CHW geometry")
	}
	count := s.Channels * s.Frequency * s.Frames
	if count > 4*1024*1024 {
		return 0, fmt.Errorf("WeSpeaker block tensor exceeds element bound")
	}
	return count, nil
}
func (b *WeSpeakerBasicBlock) OutputShape(s CHWShape) (CHWShape, error) {
	if b == nil {
		return CHWShape{}, fmt.Errorf("nil WeSpeaker block")
	}
	if err := checkWeSpeakerBlock(b.cfg); err != nil {
		return CHWShape{}, err
	}
	if _, err := chwElements(s); err != nil {
		return CHWShape{}, err
	}
	if s.Channels != b.cfg.InChannels {
		return CHWShape{}, fmt.Errorf("WeSpeaker input channel mismatch")
	}
	out := CHWShape{b.cfg.OutChannels, (s.Frequency + b.cfg.Stride - 1) / b.cfg.Stride, (s.Frames + b.cfg.Stride - 1) / b.cfg.Stride}
	if _, err := chwElements(out); err != nil {
		return CHWShape{}, err
	}
	return out, nil
}

// NewWeSpeakerBasicBlock validates every vector before copying. Input slices
// must stay immutable during construction. Copies ensure source closure/mutation
// cannot invalidate a returned block. The checkpoint loader still must verify
// tensor keys, true multidimensional shapes and dtype before calling this API.
func NewWeSpeakerBasicBlock(ctx context.Context, cfg WeSpeakerBlockConfig, w WeSpeakerBlockWeights) (*WeSpeakerBasicBlock, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := checkWeSpeakerBlock(cfg); err != nil {
		return nil, err
	}
	b := &WeSpeakerBasicBlock{cfg: cfg}
	type param struct {
		input    []float32
		count    int
		output   *[]float32
		variance bool
	}
	params := []param{{w.Conv1, cfg.OutChannels * cfg.InChannels * 9, &b.weights.Conv1, false}, {w.Conv2, cfg.OutChannels * cfg.OutChannels * 9, &b.weights.Conv2, false}}
	addBN := func(source WeSpeakerBN, dst *WeSpeakerBN) {
		params = append(params, param{source.Weight, cfg.OutChannels, &dst.Weight, false}, param{source.Bias, cfg.OutChannels, &dst.Bias, false}, param{source.RunningMean, cfg.OutChannels, &dst.RunningMean, false}, param{source.RunningVariance, cfg.OutChannels, &dst.RunningVariance, true})
	}
	addBN(w.BN1, &b.weights.BN1)
	addBN(w.BN2, &b.weights.BN2)
	if blockNeedsShortcut(cfg) {
		params = append(params, param{w.Shortcut, cfg.OutChannels * cfg.InChannels, &b.weights.Shortcut, false})
		addBN(w.ShortcutBN, &b.weights.ShortcutBN)
	} else {
		for _, v := range [][]float32{w.Shortcut, w.ShortcutBN.Weight, w.ShortcutBN.Bias, w.ShortcutBN.RunningMean, w.ShortcutBN.RunningVariance} {
			if len(v) != 0 {
				return nil, fmt.Errorf("unexpected shortcut tensors for identity block")
			}
		}
	}
	for _, p := range params {
		if len(p.input) != p.count {
			return nil, fmt.Errorf("invalid WeSpeaker block tensor length")
		}
		if err := finiteBlock(ctx, p.input); err != nil {
			return nil, err
		}
		if p.variance {
			for _, v := range p.input {
				if v < 0 {
					return nil, fmt.Errorf("negative running variance")
				}
			}
		}
	}
	for _, p := range params {
		*p.output = make([]float32, p.count)
		for start := 0; start < p.count; start += 4096 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			end := min(start+4096, p.count)
			copy((*p.output)[start:end], p.input[start:end])
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return b, nil
}

// WeSpeakerBlockObserver sees read-only transient CHW arrays at named operator
// boundaries. Copy to retain them; callers must not modify values. Completed
// observations remain visible on later cancellation. No goroutines are launched.
type WeSpeakerBlockObserver func(stage string, shape CHWShape, values []float32)

// Forward evaluates Conv3(stride,pad1)->BN->ReLU->Conv3(1,pad1)->BN, adds the
// identity or Conv1(stride)->BN shortcut, then ReLU. Inputs are CHW, batch1 and
// immutable during the call. Output is owned, including identity-shortcut cases.
// Cancellation returns no partial output and is checked per spatial row/patch
// and between operators. A running GEMV/callback/allocation is not interruptible.
func (b *WeSpeakerBasicBlock) Forward(ctx context.Context, input []float32, shape CHWShape, mode WeSpeakerBlockMode) ([]float32, CHWShape, error) {
	return b.ForwardObserved(ctx, input, shape, mode, nil)
}
func (b *WeSpeakerBasicBlock) ForwardObserved(ctx context.Context, input []float32, shape CHWShape, mode WeSpeakerBlockMode, observe WeSpeakerBlockObserver) ([]float32, CHWShape, error) {
	fail := func(err error) ([]float32, CHWShape, error) { return nil, CHWShape{}, err }
	if err := ctx.Err(); err != nil {
		return fail(err)
	}
	outShape, err := b.OutputShape(shape)
	if err != nil {
		return fail(err)
	}
	size, _ := chwElements(shape)
	if len(input) != size || (mode != WeSpeakerBlockScalar && mode != WeSpeakerBlockSIMD) {
		return fail(fmt.Errorf("invalid WeSpeaker block input/mode"))
	}
	if len(b.weights.Conv1) != b.cfg.OutChannels*b.cfg.InChannels*9 || len(b.weights.Conv2) != b.cfg.OutChannels*b.cfg.OutChannels*9 {
		return fail(fmt.Errorf("uninitialised WeSpeaker block weights"))
	}
	if err := finiteBlock(ctx, input); err != nil {
		return fail(err)
	}
	report := func(stage string, x []float32) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if observe != nil {
			observe(stage, outShape, x)
		}
		return ctx.Err()
	}
	x, err := weSpeakerBlockConv(ctx, input, shape, outShape, b.weights.Conv1, 3, b.cfg.Stride, 1, mode)
	if err != nil {
		return fail(err)
	}
	if err := report("conv1", x); err != nil {
		return fail(err)
	}
	if err := weSpeakerBlockBN(ctx, x, outShape, b.weights.BN1); err != nil {
		return fail(err)
	}
	if err := report("bn1", x); err != nil {
		return fail(err)
	}
	if err := blockReLU(ctx, x); err != nil {
		return fail(err)
	}
	if err := report("relu1", x); err != nil {
		return fail(err)
	}
	out, err := weSpeakerBlockConv(ctx, x, outShape, outShape, b.weights.Conv2, 3, 1, 1, mode)
	if err != nil {
		return fail(err)
	}
	if err := report("conv2", out); err != nil {
		return fail(err)
	}
	if err := weSpeakerBlockBN(ctx, out, outShape, b.weights.BN2); err != nil {
		return fail(err)
	}
	if err := report("bn2", out); err != nil {
		return fail(err)
	}
	shortcut := input
	if blockNeedsShortcut(b.cfg) {
		shortcut, err = weSpeakerBlockConv(ctx, input, shape, outShape, b.weights.Shortcut, 1, b.cfg.Stride, 0, mode)
		if err != nil {
			return fail(err)
		}
		if err := report("shortcut_conv", shortcut); err != nil {
			return fail(err)
		}
		if err := weSpeakerBlockBN(ctx, shortcut, outShape, b.weights.ShortcutBN); err != nil {
			return fail(err)
		}
	}
	if err := report("shortcut", shortcut); err != nil {
		return fail(err)
	}
	for i := range out {
		if i%4096 == 0 {
			if err := ctx.Err(); err != nil {
				return fail(err)
			}
		}
		out[i] += shortcut[i]
	}
	if err := finiteBlock(ctx, out); err != nil {
		return fail(err)
	}
	if err := report("residual", out); err != nil {
		return fail(err)
	}
	if err := blockReLU(ctx, out); err != nil {
		return fail(err)
	}
	if err := report("output", out); err != nil {
		return fail(err)
	}
	return out, outShape, nil
}

// Model-specific patch/GEMV composition for fixed WeSpeaker BasicBlock kernels.
// One patch and projection scratch are reused across positions: no full im2col
// tensor or per-position allocation. Generic kernel ownership remains in simd.
func weSpeakerBlockConv(ctx context.Context, x []float32, in, out CHWShape, w []float32, kernel, stride, padding int, mode WeSpeakerBlockMode) ([]float32, error) {
	n := in.Channels * kernel * kernel
	patch, projected := make([]float32, n), make([]float32, out.Channels)
	result := make([]float32, out.Channels*out.Frequency*out.Frames)
	for f := 0; f < out.Frequency; f++ {
		for t := 0; t < out.Frames; t++ {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			clear(patch)
			index := 0
			for c := 0; c < in.Channels; c++ {
				for kf := 0; kf < kernel; kf++ {
					for kt := 0; kt < kernel; kt++ {
						sf, st := f*stride+kf-padding, t*stride+kt-padding
						if sf >= 0 && sf < in.Frequency && st >= 0 && st < in.Frames {
							patch[index] = x[(c*in.Frequency+sf)*in.Frames+st]
						}
						index++
					}
				}
			}
			if mode == WeSpeakerBlockSIMD {
				if !simd.GemvRows(projected, patch, w, out.Channels, n) {
					return nil, fmt.Errorf("WeSpeaker checked block GEMV rejected shape")
				}
			} else {
				for c := range projected {
					if c%32 == 0 {
						if err := ctx.Err(); err != nil {
							return nil, err
						}
					}
					var sum float32
					for i, value := range patch {
						sum += value * w[c*n+i]
					}
					projected[c] = sum
				}
			}
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			for c, value := range projected {
				result[(c*out.Frequency+f)*out.Frames+t] = value
			}
		}
	}
	if err := finiteBlock(ctx, result); err != nil {
		return nil, err
	}
	return result, nil
}
func weSpeakerBlockBN(ctx context.Context, x []float32, shape CHWShape, bn WeSpeakerBN) error {
	count := shape.Frequency * shape.Frames
	for c := 0; c < shape.Channels; c++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		// Inference BatchNorm uses fixed running variance and epsilon1e-5. Apply
		// float32 affine scale/shift, with FMA; do not fold this into conv weights.
		inv := float32(1 / math.Sqrt(float64(bn.RunningVariance[c])+1e-5))
		scale := inv * bn.Weight[c]
		shift := bn.Bias[c] - bn.RunningMean[c]*scale
		if math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) || math.IsNaN(float64(shift)) || math.IsInf(float64(shift), 0) {
			return fmt.Errorf("non-finite WeSpeaker BatchNorm affine")
		}
		for i := 0; i < count; i++ {
			if i%4096 == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			pos := c*count + i
			x[pos] = float32(math.FMA(float64(x[pos]), float64(scale), float64(shift)))
		}
	}
	return finiteBlock(ctx, x)
}
func blockReLU(ctx context.Context, x []float32) error {
	for i, value := range x {
		if i%4096 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		x[i] = max(value, float32(0))
	}
	return ctx.Err()
}
func finiteBlock(ctx context.Context, x []float32) error {
	for i, value := range x {
		if i%4096 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("non-finite WeSpeaker block values")
		}
	}
	return ctx.Err()
}
