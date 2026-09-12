package community1

import (
	"context"
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// HeadConfig is checkpoint-supplied PyanNet segmentation-head geometry. There
// are NumLayers hidden Linear+LeakyReLU(0.01) layers, then one classifier and
// last-axis LogSoftmax over powerset classes. With zero hidden layers HiddenSize
// must be zero. No default speaker count or checkpoint identity is inferred.
type HeadConfig struct {
	InputSize, HiddenSize, NumLayers int
	Speakers, MaxActive              int
}

// HeadLinear is row-major [output,input] with a required [output] bias.
type HeadLinear struct{ Weight, Bias []float32 }

// SegmentationHead owns immutable finite copies of its parameters. Per-call
// output/scratch are independent; separate calls may share a constructed head.
// This consumes recurrent features, not PCM or SincNet output. Full segmentation
// is NOT integrated while the SincNet strict numerical gate is failing.
type SegmentationHead struct {
	cfg        HeadConfig
	classes    int
	layers     []HeadLinear
	classifier HeadLinear
}

type HeadMode uint8

const (
	HeadScalar HeadMode = iota
	// HeadSIMD reuses existing checked Plan9 GemvRows. Activations/logsoftmax
	// are scalar Go math; no new specialised SIMD kernel or speed claim.
	HeadSIMD
)

// HeadObserver receives transient read-only frame-major views. layer ranges
// [0,NumLayers) after hidden activation; layer=NumLayers is raw classifier
// logits. The final returned array is log probabilities, not this last view.
// Copy values to retain them. A callback must not mutate the view.
type HeadObserver func(layer, frames, features int, values []float32)

func checkHeadConfig(c HeadConfig) (int, error) {
	if c.InputSize < 1 || c.InputSize > 512 || c.NumLayers < 0 || c.NumLayers > 4 ||
		(c.NumLayers == 0 && c.HiddenSize != 0) || (c.NumLayers > 0 && (c.HiddenSize < 1 || c.HiddenSize > 512)) {
		return 0, fmt.Errorf("invalid segmentation head geometry")
	}
	p, err := NewPowerset(c.Speakers, c.MaxActive)
	if err != nil {
		return 0, err
	}
	return p.Classes(), nil
}

// NewSegmentationHead validates ALL weight shapes/finite values before copying.
// Caller arrays must remain immutable during construction. Tensor key/dtype and
// source shape validation remain a loader responsibility. No drivers or workers
// are created; cancellation never returns a partly constructed head.
func NewSegmentationHead(ctx context.Context, cfg HeadConfig, layers []HeadLinear, classifier HeadLinear) (*SegmentationHead, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	classes, err := checkHeadConfig(cfg)
	if err != nil {
		return nil, err
	}
	if len(layers) != cfg.NumLayers {
		return nil, fmt.Errorf("segmentation head layer count mismatch")
	}
	width := cfg.InputSize
	for index := 0; index <= cfg.NumLayers; index++ {
		weights, out := classifier, classes
		if index < cfg.NumLayers {
			weights, out = layers[index], cfg.HiddenSize
		}
		if len(weights.Weight) != out*width || len(weights.Bias) != out {
			return nil, fmt.Errorf("invalid segmentation linear%d shape", index)
		}
		if err := finiteHead(ctx, weights.Weight); err != nil {
			return nil, err
		}
		if err := finiteHead(ctx, weights.Bias); err != nil {
			return nil, err
		}
		width = out
	}
	clone := func(source HeadLinear) (HeadLinear, error) {
		out := HeadLinear{Weight: make([]float32, len(source.Weight)), Bias: make([]float32, len(source.Bias))}
		for _, pair := range []struct{ src, dst []float32 }{{source.Weight, out.Weight}, {source.Bias, out.Bias}} {
			for start := 0; start < len(pair.src); start += 4096 {
				if err := ctx.Err(); err != nil {
					return HeadLinear{}, err
				}
				end := min(start+4096, len(pair.src))
				copy(pair.dst[start:end], pair.src[start:end])
			}
		}
		return out, nil
	}
	head := &SegmentationHead{cfg: cfg, classes: classes, layers: make([]HeadLinear, len(layers))}
	for i, weights := range layers {
		head.layers[i], err = clone(weights)
		if err != nil {
			return nil, err
		}
	}
	head.classifier, err = clone(classifier)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return head, nil
}

// Classes reports the number of output powerset columns; nil/zero heads report0.
func (h *SegmentationHead) Classes() int {
	if h == nil {
		return 0
	}
	return h.classes
}

// Forward maps complete frame-major [frames,InputSize] recurrent output to an
// owned [frames,Classes] log-probability array. Frames must be1..4096. Results can
// feed Powerset.Decode with matching geometry; they are window-local speaker
// scores, not global IDs or exclusive diarization. Input must remain immutable.
// Cancellation is checked per row and after projections/callbacks. A running
// GEMV/callback is synchronous; no partial result escapes on error. Observers
// may already have seen completed layers. No per-frame allocation is performed.
func (h *SegmentationHead) Forward(ctx context.Context, input []float32, frames int, mode HeadMode) ([]float32, error) {
	return h.ForwardObserved(ctx, input, frames, mode, nil)
}
func (h *SegmentationHead) ForwardObserved(ctx context.Context, input []float32, frames int, mode HeadMode, observe HeadObserver) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if h == nil || h.classes < 1 {
		return nil, fmt.Errorf("nil/uninitialised segmentation head")
	}
	classes, err := checkHeadConfig(h.cfg)
	if err != nil {
		return nil, err
	}
	if classes != h.classes || len(h.layers) != h.cfg.NumLayers || frames < 1 || frames > MaxPowersetFrames || len(input) != frames*h.cfg.InputSize || (mode != HeadScalar && mode != HeadSIMD) {
		return nil, fmt.Errorf("invalid segmentation head input/mode")
	}
	if err := finiteHead(ctx, input); err != nil {
		return nil, err
	}
	// Two alternating buffers cover the hidden stack; the classifier has its
	// own result buffer and logsoftmax reuses it after the logits observer.
	var first, second []float32
	if h.cfg.NumLayers > 0 {
		first = make([]float32, frames*h.cfg.HiddenSize)
	}
	if h.cfg.NumLayers > 1 {
		second = make([]float32, len(first))
	}
	output := make([]float32, frames*h.classes)
	sequence, width := input, h.cfg.InputSize
	for index := 0; index <= h.cfg.NumLayers; index++ {
		weights, cols, dst := h.classifier, h.classes, output
		if index < h.cfg.NumLayers {
			weights, cols, dst = h.layers[index], h.cfg.HiddenSize, first
			if index%2 != 0 {
				dst = second
			}
		}
		for frame := 0; frame < frames; frame++ {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			row := dst[frame*cols : (frame+1)*cols]
			x := sequence[frame*width : (frame+1)*width]
			if mode == HeadSIMD {
				if !simd.GemvRows(row, x, weights.Weight, cols, width) {
					return nil, fmt.Errorf("checked head GEMV shape rejected")
				}
			} else {
				for out := range row {
					if out%32 == 0 {
						if err := ctx.Err(); err != nil {
							return nil, err
						}
					}
					var sum float32
					for in, value := range x {
						sum += value * weights.Weight[out*width+in]
					}
					row[out] = sum
				}
			}
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			for out := range row {
				value := row[out] + weights.Bias[out]
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					return nil, fmt.Errorf("non-finite segmentation affine output")
				}
				if index < h.cfg.NumLayers && value < 0 {
					value *= 0.01
				}
				row[out] = value
			}
		}
		if observe != nil {
			observe(index, frames, cols, dst)
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		sequence, width = dst, cols
	}
	if err := headLogSoftmax(ctx, output, frames, h.classes); err != nil {
		return nil, err
	}
	return output, nil
}

func finiteHead(ctx context.Context, values []float32) error {
	for i, value := range values {
		if i%4096 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fmt.Errorf("non-finite segmentation head input/weights")
		}
	}
	return ctx.Err()
}

// Shift before logsumexp to avoid overflow and loss of the normaliser for large
// common offsets. Finite extreme logits can yield -Inf on float32 conversion
// for impossible classes; at least the max class remains finite. Powerset.Decode
// explicitly accepts individual -Inf scores. No positive log probability or
// all-impossible row is introduced. Shapes and finite logits are validated above.
func headLogSoftmax(ctx context.Context, x []float32, frames, classes int) error {
	for frame := 0; frame < frames; frame++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		row := x[frame*classes : (frame+1)*classes]
		maximum := row[0]
		for _, value := range row[1:] {
			maximum = max(maximum, value)
		}
		sum := float64(0)
		for _, value := range row {
			sum += math.Exp(float64(value) - float64(maximum))
		}
		normalizer := math.Log(sum)
		for i, value := range row {
			row[i] = float32((float64(value) - float64(maximum)) - normalizer)
		}
	}
	return ctx.Err()
}
