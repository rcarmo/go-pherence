package omnivoice

import (
	"context"
	"fmt"
	"strings"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
)

// CodecDecoder implements HiggsAudioV2 RVQ/DAC decoding. Convolutions use
// SIMD GEMM with bounded 64-position im2col tiles. Prepare reserves reusable
// activation slots for allocation-free DecodeInto calls. Snake uses bounded
// SIMD sine where supported; transposed-convolution scatter is scalar.
// Not concurrent safe.
type CodecDecoder struct {
	ops            map[string]codecOperator
	names          map[string]map[string]string
	weights        *loader.CodecWeights
	scratch        *codecScratch
	quantizerNames [8]string
	blockNames     [5]string
	residualNames  [5][3]string
}

func NewCodecDecoder(w *loader.CodecWeights) (*CodecDecoder, error) {
	if w == nil || len(w.Tensors["fc2.weight"]) == 0 || len(w.Tensors["acoustic_decoder.conv2.weight"]) == 0 {
		return nil, fmt.Errorf("omnivoice: missing codec decoder tensors")
	}
	if err := validateCodecWeights(w); err != nil {
		return nil, err
	}
	d := &CodecDecoder{weights: w, names: map[string]map[string]string{}, ops: map[string]codecOperator{}}
	for i := range d.quantizerNames {
		d.quantizerNames[i] = fmt.Sprintf("quantizer.quantizers.%d.", i)
	}
	for i := range d.blockNames {
		d.blockNames[i] = fmt.Sprintf("acoustic_decoder.block.%d.", i)
		for j := range d.residualNames[i] {
			d.residualNames[i][j] = fmt.Sprintf("%sres_unit%d.", d.blockNames[i], j+1)
		}
	}
	prefixes := append([]string(nil), d.quantizerNames[:]...)
	prefixes = append(prefixes, d.blockNames[:]...)
	for _, rows := range d.residualNames {
		prefixes = append(prefixes, rows[:]...)
	}
	for _, p := range prefixes {
		d.names[p] = map[string]string{}
		for _, suffix := range []string{"codebook.embed", "project_out", "snake1.alpha", "snake2.alpha", "conv1", "conv2", "conv_t1"} {
			d.names[p][suffix] = p + suffix
		}
	}
	for name, shape := range w.Shapes {
		if strings.HasSuffix(name, ".weight") {
			base := strings.TrimSuffix(name, ".weight")
			d.ops[base] = codecOperator{w.Tensors[name], w.Tensors[base+".bias"], shape}
		}
	}
	return d, nil
}

type codecOperator struct {
	weight, bias []float32
	shape        []int
}

type signal struct {
	data             []float32
	channels, frames int
}

func (d *CodecDecoder) Decode(ctx context.Context, codes []int, books, frames int) ([]float32, error) {
	if err := d.Prepare(frames); err != nil {
		return nil, err
	}
	out := make([]float32, frames*960)
	if err := d.DecodeInto(ctx, out, codes, books, frames); err != nil {
		return nil, err
	}
	return out, nil
}
func (d *CodecDecoder) decode(ctx context.Context, codes []int, books, frames int) ([]float32, error) {
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
	sum := signal{d.buffer(hidden * frames), hidden, frames}
	for book := 0; book < books; book++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		prefix := d.quantizerNames[book]
		embedding := d.weights.Tensors[d.names[prefix]["codebook.embed"]]
		es := d.weights.Shapes[d.names[prefix]["codebook.embed"]]
		if len(es) != 2 {
			return nil, fmt.Errorf("omnivoice: codebook missing")
		}
		x := signal{d.buffer(es[1] * frames), es[1], frames}
		for t := 0; t < frames; t++ {
			code := codes[book*frames+t]
			for c := 0; c < es[1]; c++ {
				x.data[c*frames+t] = embedding[code*es[1]+c]
			}
		}
		y, err := d.conv(x, d.names[prefix]["project_out"], 1, 0, 1)
		if err != nil {
			return nil, err
		}
		simd.VecAdd(sum.data, sum.data, y.data)
		d.release(x)
		d.release(y)
	}
	x, err := d.conv(sum, "fc2", 1, 0, 1)
	if err != nil {
		return nil, err
	}
	d.release(sum)
	previous := x
	x, err = d.conv(x, "acoustic_decoder.conv1", 1, 3, 1)
	d.release(previous)
	if err != nil {
		return nil, err
	}
	for i, stride := range d.weights.Rates {
		if err = ctx.Err(); err != nil {
			return nil, err
		}
		prefix := d.blockNames[i]
		if err = d.snake(x, d.names[prefix]["snake1.alpha"]); err != nil {
			return nil, err
		}
		previous = x
		x, err = d.transpose(x, d.names[prefix]["conv_t1"], stride, (stride+1)/2, stride%2)
		d.release(previous)
		if err != nil {
			return nil, err
		}
		for j, dilation := range []int{1, 3, 9} {
			res := x
			rp := d.residualNames[i][j]
			work := signal{d.buffer(len(x.data)), x.channels, x.frames}
			copy(work.data, x.data)
			if err = d.snake(work, d.names[rp]["snake1.alpha"]); err != nil {
				return nil, err
			}
			previous = work
			work, err = d.conv(work, d.names[rp]["conv1"], 1, 3*dilation, dilation)
			d.release(previous)
			if err != nil {
				return nil, err
			}
			if err = d.snake(work, d.names[rp]["snake2.alpha"]); err != nil {
				return nil, err
			}
			previous = work
			work, err = d.conv(work, d.names[rp]["conv2"], 1, 0, 1)
			d.release(previous)
			if err != nil {
				return nil, err
			}
			if len(work.data) != len(res.data) {
				return nil, fmt.Errorf("omnivoice: residual dimensions differ")
			}
			simd.VecAdd(work.data, work.data, res.data)
			d.release(res)
			x = work
		}
	}
	if err = d.snake(x, "acoustic_decoder.snake1.alpha"); err != nil {
		return nil, err
	}
	previous = x
	x, err = d.conv(x, "acoustic_decoder.conv2", 1, 3, 1)
	d.release(previous)
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
		snakeChannel(x.data[c*x.frames:(c+1)*x.frames], a)
	}
	return nil
}
func (d *CodecDecoder) conv(x signal, name string, stride, padding, dilation int) (signal, error) {
	op, ok := d.ops[name]
	if !ok {
		op = codecOperator{d.weights.Tensors[name+".weight"], d.weights.Tensors[name+".bias"], d.weights.Shapes[name+".weight"]}
	}
	weight, shape, bias := op.weight, op.shape, op.bias
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
	y := signal{d.buffer(out * length), out, length}
	const tile = 64
	k := x.channels * kernel
	var packed, result []float32
	if d.scratch != nil {
		packed = d.scratch.packed[:k*tile]
		result = d.scratch.result[:out*tile]
	} else {
		packed = make([]float32, k*tile)
		result = make([]float32, out*tile)
	}
	for start := 0; start < length; start += tile {
		n := min(tile, length-start)
		p := packed[:k*n]
		clear(p)
		for c := 0; c < x.channels; c++ {
			for j := 0; j < kernel; j++ {
				if stride == 1 {
					// Dilation shifts each tap origin; positions within a tap
					// remain contiguous. Cleared scratch supplies the padding.
					source := start - padding + j*dilation
					lo, hi := max(0, -source), min(n, x.frames-source)
					if lo < hi {
						copy(p[(c*kernel+j)*n+lo:(c*kernel+j)*n+hi], x.data[c*x.frames+source+lo:c*x.frames+source+hi])
					}
				} else {
					for t := 0; t < n; t++ {
						source := (start+t)*stride - padding + j*dilation
						if source >= 0 && source < x.frames {
							p[(c*kernel+j)*n+t] = x.data[c*x.frames+source]
						}
					}
				}
			}
		}
		r := result[:out*n]
		if !d.gemm(r, weight, p, out, n, k) {
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

// Prepared decode reuses a bounded panel for full-tile SIMD GEMM. The
// unprepared test/helper path retains the existing allocation behavior.
func (d *CodecDecoder) gemm(c, a, b []float32, m, n, k int) bool {
	if d.scratch != nil {
		return simd.SgemmNNPackedOverwriteTo(c, a, b, d.scratch.gemmPanel, m, n, k, k, n, n)
	}
	// The unprepared NN path accumulates; prepared packed GEMM clears itself.
	clear(c)
	return simd.SgemmNNTo(c, a, b, m, n, k, 1, k, n, n)
}

func (d *CodecDecoder) transpose(x signal, name string, stride, padding, outputPadding int) (signal, error) {
	op, ok := d.ops[name]
	if !ok {
		op = codecOperator{d.weights.Tensors[name+".weight"], d.weights.Tensors[name+".bias"], d.weights.Shapes[name+".weight"]}
	}
	w, s, bias := op.weight, op.shape, op.bias
	if len(s) != 3 || s[0] != x.channels {
		return signal{}, fmt.Errorf("omnivoice: invalid transposed conv")
	}
	out, kernel := s[1], s[2]
	length := (x.frames-1)*stride - 2*padding + kernel + outputPadding
	y := signal{d.buffer(out * length), out, length}
	tile := 32
	if d.scratch != nil && simd.HasSgemmAsm {
		tile = codecPackedTransposeTile
	}
	ncols := out * kernel
	var input, projected []float32
	if d.scratch != nil {
		input = d.scratch.input[:tile*x.channels]
		projected = d.scratch.projected[:tile*ncols]
	} else {
		input = make([]float32, tile*x.channels)
		projected = make([]float32, tile*ncols)
	}
	for start := 0; start < x.frames; start += tile {
		n := min(tile, x.frames-start)
		a := input[:n*x.channels]
		for t := 0; t < n; t++ {
			for c := 0; c < x.channels; c++ {
				a[t*x.channels+c] = x.data[c*x.frames+start+t]
			}
		}
		p := projected[:n*ncols]
		if !d.gemm(p, a, w, n, ncols, x.channels) {
			return signal{}, fmt.Errorf("omnivoice: transposed GEMM shape")
		}
		for t := 0; t < n; t++ {
			origin := (start+t)*stride - padding
			lo, hi := max(0, -origin), min(kernel, length-origin)
			if lo >= hi {
				continue
			}
			for c := 0; c < out; c++ {
				dst := y.data[c*length+origin+lo : c*length+origin+hi]
				src := p[t*ncols+c*kernel+lo : t*ncols+c*kernel+hi]
				for j, v := range src {
					dst[j] += v
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
