package omnivoice

import (
	"context"
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
)

// CodecDecoder is an initial decode-only HiggsAudioV2 RVQ/DAC implementation.
// Convolutions use SIMD GEMM with bounded 64-position im2col tiles. Activation
// buffers are currently allocated per operator; optimization follows parity.
// Snake sine and transposed-convolution scatter are scalar. Not concurrent safe.
type CodecDecoder struct{ weights *loader.CodecWeights }

func NewCodecDecoder(w *loader.CodecWeights) (*CodecDecoder, error) {
	if w == nil || len(w.Tensors["fc2.weight"]) == 0 || len(w.Tensors["acoustic_decoder.conv2.weight"]) == 0 {
		return nil, fmt.Errorf("omnivoice: missing codec decoder tensors")
	}
	if err := validateCodecWeights(w); err != nil {
		return nil, err
	}
	return &CodecDecoder{weights: w}, nil
}

type signal struct {
	data             []float32
	channels, frames int
}

func (d *CodecDecoder) Decode(ctx context.Context, codes []int, books, frames int) ([]float32, error) {
	if ctx == nil || books < 1 || books > d.weights.Quantizers || frames < 1 || frames > 250 || len(codes) != books*frames {
		return nil, fmt.Errorf("omnivoice: invalid codec input")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	for _, code := range codes {
		if code < 0 || code >= d.weights.CodebookSize {
			return nil, fmt.Errorf("omnivoice: code out of range")
		}
	}
	shape := d.weights.Shapes["fc2.weight"]
	if len(shape) != 2 {
		return nil, fmt.Errorf("omnivoice: invalid codec projection")
	}
	hidden := shape[1]
	sum := signal{make([]float32, hidden*frames), hidden, frames}
	for book := 0; book < books; book++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		prefix := fmt.Sprintf("quantizer.quantizers.%d.", book)
		embedding := d.weights.Tensors[prefix+"codebook.embed"]
		es := d.weights.Shapes[prefix+"codebook.embed"]
		if len(es) != 2 {
			return nil, fmt.Errorf("omnivoice: codebook missing")
		}
		x := signal{make([]float32, es[1]*frames), es[1], frames}
		for t := 0; t < frames; t++ {
			code := codes[book*frames+t]
			for c := 0; c < es[1]; c++ {
				x.data[c*frames+t] = embedding[code*es[1]+c]
			}
		}
		y, err := d.conv(x, prefix+"project_out", 1, 0, 1)
		if err != nil {
			return nil, err
		}
		simd.VecAdd(sum.data, sum.data, y.data)
	}
	x, err := d.conv(sum, "fc2", 1, 0, 1)
	if err != nil {
		return nil, err
	}
	x, err = d.conv(x, "acoustic_decoder.conv1", 1, 3, 1)
	if err != nil {
		return nil, err
	}
	for i, stride := range d.weights.Rates {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		prefix := fmt.Sprintf("acoustic_decoder.block.%d.", i)
		if err = d.snake(x, prefix+"snake1.alpha"); err != nil {
			return nil, err
		}
		x, err = d.transpose(x, prefix+"conv_t1", stride, (stride+1)/2, stride%2)
		if err != nil {
			return nil, err
		}
		for j, dilation := range []int{1, 3, 9} {
			res := x
			rp := fmt.Sprintf("%sres_unit%d.", prefix, j+1)
			work := signal{append([]float32(nil), x.data...), x.channels, x.frames}
			if err = d.snake(work, rp+"snake1.alpha"); err != nil {
				return nil, err
			}
			work, err = d.conv(work, rp+"conv1", 1, 3*dilation, dilation)
			if err != nil {
				return nil, err
			}
			if err = d.snake(work, rp+"snake2.alpha"); err != nil {
				return nil, err
			}
			work, err = d.conv(work, rp+"conv2", 1, 0, 1)
			if err != nil {
				return nil, err
			}
			if len(work.data) != len(res.data) {
				return nil, fmt.Errorf("omnivoice: residual dimensions differ")
			}
			simd.VecAdd(work.data, work.data, res.data)
			x = work
		}
	}
	if err = d.snake(x, "acoustic_decoder.snake1.alpha"); err != nil {
		return nil, err
	}
	x, err = d.conv(x, "acoustic_decoder.conv2", 1, 3, 1)
	if err != nil {
		return nil, err
	}
	// Higgs replaces DAC's final tanh with identity. Do not clip here.
	return x.data, nil
}
func (d *CodecDecoder) snake(x signal, name string) error {
	alpha := d.weights.Tensors[name]
	if len(alpha) != x.channels {
		return fmt.Errorf("omnivoice: invalid snake %s", name)
	}
	for c, a := range alpha {
		for t := 0; t < x.frames; t++ {
			i := c*x.frames + t
			phase := a * x.data[i]
			s := float32(math.Sin(float64(phase)))
			x.data[i] += (float32(1) / (a + 1e-9)) * (s * s)
		}
	}
	return nil
}
func (d *CodecDecoder) conv(x signal, name string, stride, padding, dilation int) (signal, error) {
	weight, shape := d.weights.Tensors[name+".weight"], d.weights.Shapes[name+".weight"]
	bias := d.weights.Tensors[name+".bias"]
	if len(shape) < 2 || len(shape) > 3 || shape[1] != x.channels {
		return signal{}, fmt.Errorf("omnivoice: invalid conv %s", name)
	}
	out, kernel := shape[0], 1
	if len(shape) == 3 {
		kernel = shape[2]
	}
	length := (x.frames+2*padding-dilation*(kernel-1)-1)/stride + 1
	if length <= 0 {
		return signal{}, fmt.Errorf("omnivoice: short conv input")
	}
	y := signal{make([]float32, out*length), out, length}
	const tile = 64
	k := x.channels * kernel
	packed := make([]float32, k*tile)
	result := make([]float32, out*tile)
	for start := 0; start < length; start += tile {
		n := min(tile, length-start)
		p := packed[:k*n]
		clear(p)
		for c := 0; c < x.channels; c++ {
			for j := 0; j < kernel; j++ {
				for t := 0; t < n; t++ {
					source := (start+t)*stride - padding + j*dilation
					if source >= 0 && source < x.frames {
						p[(c*kernel+j)*n+t] = x.data[c*x.frames+source]
					}
				}
			}
		}
		r := result[:out*n]
		clear(r)
		if !simd.SgemmNNTo(r, weight, p, out, n, k, 1, k, n, n) {
			return signal{}, fmt.Errorf("omnivoice: conv GEMM shape")
		}
		for c := 0; c < out; c++ {
			b := float32(0)
			if len(bias) > 0 {
				if len(bias) != out {
					return signal{}, fmt.Errorf("omnivoice: invalid bias")
				}
				b = bias[c]
			}
			for t := 0; t < n; t++ {
				y.data[c*length+start+t] = r[c*n+t] + b
			}
		}
	}
	return y, nil
}
func (d *CodecDecoder) transpose(x signal, name string, stride, padding, outputPadding int) (signal, error) {
	w, s := d.weights.Tensors[name+".weight"], d.weights.Shapes[name+".weight"]
	bias := d.weights.Tensors[name+".bias"]
	if len(s) != 3 || s[0] != x.channels {
		return signal{}, fmt.Errorf("omnivoice: invalid transposed conv")
	}
	out, kernel := s[1], s[2]
	length := (x.frames-1)*stride - 2*padding + kernel + outputPadding
	y := signal{make([]float32, out*length), out, length}
	const tile = 32
	ncols := out * kernel
	input := make([]float32, tile*x.channels)
	projected := make([]float32, tile*ncols)
	for start := 0; start < x.frames; start += tile {
		n := min(tile, x.frames-start)
		a := input[:n*x.channels]
		for t := 0; t < n; t++ {
			for c := 0; c < x.channels; c++ {
				a[t*x.channels+c] = x.data[c*x.frames+start+t]
			}
		}
		p := projected[:n*ncols]
		clear(p)
		if !simd.SgemmNNTo(p, a, w, n, ncols, x.channels, 1, x.channels, ncols, ncols) {
			return signal{}, fmt.Errorf("omnivoice: transposed GEMM shape")
		}
		for t := 0; t < n; t++ {
			for c := 0; c < out; c++ {
				for j := 0; j < kernel; j++ {
					dest := (start+t)*stride - padding + j
					if dest >= 0 && dest < length {
						y.data[c*length+dest] += p[t*ncols+c*kernel+j]
					}
				}
			}
		}
	}
	if len(bias) != out {
		return signal{}, fmt.Errorf("omnivoice: invalid transpose bias")
	}
	for c, b := range bias {
		for t := 0; t < length; t++ {
			y.data[c*length+t] += b
		}
	}
	return y, nil
}
