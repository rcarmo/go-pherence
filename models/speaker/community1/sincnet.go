package community1

import (
	"context"
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

const MaxSincNetSamples = 160000

// SincNetWeights follows pyannote SincNet at 16kHz: affine waveform instance
// norm, ParamSincFB 40 learned band pairs (80 even/odd filters, kernel251),
// two learned convolutions [60,80,5]/[60,60,5], and three affine instance norms.
// Running means/variances are not used: normalisation spans the entire window.
type SincNetNorm struct{ Weight, Bias []float32 }
type SincNetConv struct{ Weight, Bias []float32 }
type SincNetWeights struct {
	WaveNorm      SincNetNorm
	LowHz, BandHz []float32
	Conv          [2]SincNetConv
	Norm          [3]SincNetNorm
}

// SincNet owns copied weights and precomputed filters. It has no mutable
// inference state. Experimental: narrow-band Torch float32 parity exceeds the
// 2e-4 gate; see TestSincNetStrictNarrowBandOracle and verification evidence.
// Not qualified for production/default selection. The first-convolution stride
// is explicit checkpoint metadata;
// accepted range1..10 includes pyannote defaults, not a detected model identity.
type SincNet struct {
	stride  int
	weights SincNetWeights
	filters []float32
}

// SincNetGrid describes the nominal convolution/pooling frame grid in samples
// of canonical16k PCM. Centers use zero-based sample indexes. InstanceNorm uses
// the WHOLE window, so ReceptiveField is not a strict dependency/streaming bound.
type SincNetGrid struct{ Frames, Step, FirstCenter, ReceptiveField int }

func sincNetGrid(samples, stride int) (SincNetGrid, error) {
	if stride < 1 || stride > 10 || samples < 1 || samples > MaxSincNetSamples {
		return SincNetGrid{}, fmt.Errorf("invalid SincNet stride/sample bounds")
	}
	n := samples
	for index, kernel := range []int{251, 3, 5, 3, 5, 3} {
		step := 1
		if index == 0 {
			step = stride
		} else if index%2 == 1 {
			step = 3
		}
		if n < kernel {
			return SincNetGrid{}, fmt.Errorf("SincNet window too short")
		}
		n = 1 + (n-kernel)/step
		// InstanceNorm1d without running stats requires more than one time element.
		if index%2 == 1 && n < 2 {
			return SincNetGrid{}, fmt.Errorf("SincNet instance norm needs at least two frames")
		}
	}
	if n > 4096 {
		return SincNetGrid{}, fmt.Errorf("SincNet output exceeds4096 frame bound")
	}
	return SincNetGrid{Frames: n, Step: 27 * stride, FirstCenter: 125 + 37*stride, ReceptiveField: 251 + 74*stride}, nil
}
func (m *SincNet) Grid(samples int) (SincNetGrid, error) {
	if m == nil {
		return SincNetGrid{}, fmt.Errorf("nil SincNet")
	}
	return sincNetGrid(samples, m.stride)
}

// NewSincNet validates and owns weights; learned cutoffs are transformed with
// abs/min-band/Nyquist clipping exactly as the reference. Degenerate/reversed
// bands are rejected rather than generating division-by-zero filters. A model
// loader must verify full checkpoint keys/shapes before calling this constructor.
func NewSincNet(ctx context.Context, stride int, w SincNetWeights) (*SincNet, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if stride < 1 || stride > 10 {
		return nil, fmt.Errorf("SincNet stride requires1..10")
	}
	type array struct {
		src  []float32
		size int
		dst  *[]float32
	}
	owned := SincNetWeights{}
	arrays := []array{{w.WaveNorm.Weight, 1, &owned.WaveNorm.Weight}, {w.WaveNorm.Bias, 1, &owned.WaveNorm.Bias}, {w.LowHz, 40, &owned.LowHz}, {w.BandHz, 40, &owned.BandHz}}
	channels := []int{80, 60, 60}
	for i := range w.Norm {
		arrays = append(arrays, array{w.Norm[i].Weight, channels[i], &owned.Norm[i].Weight}, array{w.Norm[i].Bias, channels[i], &owned.Norm[i].Bias})
	}
	for i, in := range []int{80, 60} {
		arrays = append(arrays, array{w.Conv[i].Weight, 60 * in * 5, &owned.Conv[i].Weight}, array{w.Conv[i].Bias, 60, &owned.Conv[i].Bias})
	}
	for _, a := range arrays {
		if len(a.src) != a.size {
			return nil, fmt.Errorf("invalid SincNet tensor length")
		}
		if err := finiteLSTM(ctx, a.src); err != nil {
			return nil, err
		}
	}
	for _, a := range arrays {
		*a.dst = make([]float32, a.size)
		for start := 0; start < a.size; start += 4096 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			end := min(start+4096, a.size)
			copy((*a.dst)[start:end], a.src[start:end])
		}
	}
	filters, err := sincNetFilters(ctx, owned.LowHz, owned.BandHz)
	if err != nil {
		return nil, err
	}
	return &SincNet{stride: stride, weights: owned, filters: filters}, nil
}

func sincNetFilters(ctx context.Context, lowHz, bandHz []float32) ([]float32, error) {
	out := make([]float32, 80*251)
	for band := 0; band < 40; band++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		low := float32(50) + float32(math.Abs(float64(lowHz[band])))
		high := min(low+50+float32(math.Abs(float64(bandHz[band]))), float32(8000))
		width := high - low
		if width <= 0 || math.IsInf(float64(width), 0) || math.IsNaN(float64(width)) {
			return nil, fmt.Errorf("degenerate SincNet frequency band%d", band)
		}
		denominator := 2 * width
		even, odd := out[band*251:(band+1)*251], out[(40+band)*251:(41+band)*251]
		even[125] = 1
		odd[125] = 0
		for i := 0; i < 125; i++ {
			window := float32(0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/250))
			n := float32(2*math.Pi) * float32(float32(i-125)/16000)
			lo, hi := low*n, high*n
			// Keep float32 rounding at each Torch tensor operation. exp/trig are Go
			// scalar math here; independent fixture tolerances cover remaining error.
			e := ((float32(math.Sin(float64(hi))) - float32(math.Sin(float64(lo)))) / (n / 2)) * window
			o := ((float32(math.Cos(float64(lo))) - float32(math.Cos(float64(hi)))) / (n / 2)) * window
			even[i] = e / denominator
			even[250-i] = even[i]
			odd[i] = o / denominator
			odd[250-i] = -odd[i]
		}
	}
	if err := finiteLSTM(ctx, out); err != nil {
		return nil, err
	}
	return out, nil
}

// SincNetMode selects scalar accumulation or existing Plan9 SIMD dots plus
// checked AVX2/FMA affine normalization where supported. Statistics reductions
// are unchanged. Neither mode uses CGo/BLAS, GPU or worker pools.
type SincNetMode uint8

const (
	SincNetScalar SincNetMode = iota
	SincNetSIMD
)

// SincNetObserver receives transient channel-major boundary views. stage=-1
// is waveform normalisation; stages0..2 are pooled/normalised/activated blocks.
// The observer must not mutate/retain the view; copy if needed. Calls synchronise.
type SincNetObserver func(stage, channels, frames int, values []float32)

// Forward returns owned FRAME-major [grid.Frames,60] input for LSTM. It neither
// pads, truncates, rescales nor resamples. PCM is mono16k; minimum duration is
// determined by Grid, maximum160000samples/4096featureframes. Whole-window
// normalisation makes arbitrary chunking semantically different from one call.
func (m *SincNet) Forward(ctx context.Context, pcm []float32, mode SincNetMode) ([]float32, SincNetGrid, error) {
	return m.ForwardObserved(ctx, pcm, mode, nil)
}
func (m *SincNet) ForwardObserved(ctx context.Context, pcm []float32, mode SincNetMode, observe SincNetObserver) ([]float32, SincNetGrid, error) {
	fail := func(err error) ([]float32, SincNetGrid, error) { return nil, SincNetGrid{}, err }
	if err := ctx.Err(); err != nil {
		return fail(err)
	}
	grid, err := m.Grid(len(pcm))
	if err != nil {
		return fail(err)
	}
	if len(m.filters) != 80*251 || (mode != SincNetScalar && mode != SincNetSIMD) {
		return fail(fmt.Errorf("invalid SincNet model/mode"))
	}
	if err := finiteLSTM(ctx, pcm); err != nil {
		return fail(err)
	}
	x := append([]float32(nil), pcm...)
	n := len(pcm)
	channels := 1
	if err := sincNetNormMode(ctx, x, channels, n, m.weights.WaveNorm, mode); err != nil {
		return fail(err)
	}
	if observe != nil {
		observe(-1, channels, n, x)
	}
	for stage := 0; stage < 3; stage++ {
		if err := ctx.Err(); err != nil {
			return fail(err)
		}
		weight := m.filters
		var bias []float32
		outChannels, kernel, stride := 80, 251, m.stride
		if stage > 0 {
			weight = m.weights.Conv[stage-1].Weight
			bias = m.weights.Conv[stage-1].Bias
			outChannels, kernel, stride = 60, 5, 1
		}
		var conv []float32
		conv, n, err = sincNetConvolve(ctx, x, channels, n, weight, bias, outChannels, kernel, stride, mode)
		if err != nil {
			return fail(err)
		}
		if stage == 0 {
			for i, value := range conv {
				if i%4096 == 0 {
					if err := ctx.Err(); err != nil {
						return fail(err)
					}
				}
				conv[i] = float32(math.Abs(float64(value)))
			}
		}
		x, n, err = sincNetPool(ctx, conv, outChannels, n)
		if err != nil {
			return fail(err)
		}
		if err := sincNetNormMode(ctx, x, outChannels, n, m.weights.Norm[stage], mode); err != nil {
			return fail(err)
		}
		for i, value := range x {
			if i%4096 == 0 {
				if err := ctx.Err(); err != nil {
					return fail(err)
				}
			}
			if value < 0 {
				x[i] = value * 0.01
			}
		}
		channels = outChannels
		if observe != nil {
			observe(stage, channels, n, x)
		}
	}
	if err := ctx.Err(); err != nil {
		return fail(err)
	}
	if n != grid.Frames {
		return fail(fmt.Errorf("SincNet frame-grid mismatch"))
	}
	output := make([]float32, len(x))
	for frame := 0; frame < n; frame++ {
		if err := ctx.Err(); err != nil {
			return fail(err)
		}
		for channel := 0; channel < 60; channel++ {
			output[frame*60+channel] = x[channel*n+frame]
		}
	}
	if err := ctx.Err(); err != nil {
		return fail(err)
	}
	return output, grid, nil
}

// Population (biased) variance across time independently per channel, epsilon
// 1e-5, affine parameters. Pinned Torch contiguous BatchNorm statistics use a
// float64 sum, rounded float32 mean, float32 centred squares accumulated into
// float64, then rounded float32 variance sum/division. Its affine beta is an
// FMA, as is output scale/shift. See testdata/sincnet-norm-reference.json.gz.
// Full SincNet convolution/filter parity remains an open strict gate.
func sincNetNorm(ctx context.Context, x []float32, channels, frames int, norm SincNetNorm) error {
	return sincNetNormMode(ctx, x, channels, frames, norm, SincNetScalar)
}

func sincNetNormMode(ctx context.Context, x []float32, channels, frames int, norm SincNetNorm, mode SincNetMode) error {
	for channel := 0; channel < channels; channel++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		row := x[channel*frames : (channel+1)*frames]
		mean := float64(0)
		for i, value := range row {
			if i%4096 == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			mean += float64(value)
		}
		mean = float64(float32(mean / float64(frames)))
		variance := float64(0)
		for i, value := range row {
			if i%4096 == 0 {
				if err := ctx.Err(); err != nil {
					return err
				}
			}
			delta := value - float32(mean)
			variance += float64(delta * delta)
		}
		variance = float64(float32(variance) / float32(frames))
		// Torch's CPU instance norm applies one affine scale/shift after
		// computing float32 mean/inverse standard deviation.
		inverse := float32(1 / math.Sqrt(float64(float32(variance))+1e-5))
		scale := inverse * norm.Weight[channel]
		shift := sincNetFMA32(-float32(mean), scale, norm.Bias[channel])
		for start := 0; start < frames; start += 4096 {
			if err := ctx.Err(); err != nil {
				return err
			}
			block := row[start:min(start+4096, frames)]
			if mode == SincNetSIMD {
				if !simd.AffineF32InPlaceChecked(block, scale, shift) {
					return fmt.Errorf("unsupported/nonfinite SincNet affine input or FP environment")
				}
			} else {
				for i, value := range block {
					block[i] = sincNetFMA32(value, scale, shift)
				}
			}
		}
	}
	return finiteLSTM(ctx, x)
}

// Shared exact-rounding fallback lives with the reusable affine kernel.
func sincNetFMA32(a, b, c float32) float32 { return simd.FMA32Scalar(a, b, c) }

func sincNetConvolve(ctx context.Context, x []float32, in, n int, weight, bias []float32, out, kernel, stride int, mode SincNetMode) ([]float32, int, error) {
	frames := 1 + (n-kernel)/stride
	result := make([]float32, out*frames)
	for channel := 0; channel < out; channel++ {
		for frame := 0; frame < frames; frame++ {
			if frame%32 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, 0, err
				}
			}
			sum := float32(0)
			for input := 0; input < in; input++ {
				a := x[input*n+frame*stride : input*n+frame*stride+kernel]
				b := weight[(channel*in+input)*kernel : (channel*in+input+1)*kernel]
				if mode == SincNetSIMD {
					sum += simd.Sdot(a, b)
				} else {
					for i, value := range a {
						sum += value * b[i]
					}
				}
			}
			if bias != nil {
				sum += bias[channel]
			}
			result[channel*frames+frame] = sum
		}
	}
	if err := finiteLSTM(ctx, result); err != nil {
		return nil, 0, err
	}
	return result, frames, nil
}
func sincNetPool(ctx context.Context, x []float32, channels, n int) ([]float32, int, error) {
	frames := n / 3
	out := make([]float32, channels*frames)
	for channel := 0; channel < channels; channel++ {
		for frame := 0; frame < frames; frame++ {
			if frame%4096 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, 0, err
				}
			}
			start := channel*n + 3*frame
			out[channel*frames+frame] = max(x[start], x[start+1], x[start+2])
		}
	}
	return out, frames, nil
}
