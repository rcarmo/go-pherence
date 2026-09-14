package omnivoice

import (
	"fmt"
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
	"math"
	"testing"
)

func (d *CodecDecoder) transposeReference(x signal, name string, stride, padding, outputPadding int) (signal, error) {
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
	const tile = 32
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
		clear(p)
		if !d.gemm(p, a, w, n, ncols, x.channels) {
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

func TestCodecTransposeBoundedScatterExact(t *testing.T) {
	for _, frames := range []int{1, 2, 7, 31, 32, 33, 65} {
		for _, kernel := range []int{1, 3, 7, 16} {
			for _, stride := range []int{1, 2, 3, 8} {
				for _, padding := range []int{0, 1, 4, 16} {
					for _, extra := range []int{0, stride - 1} {
						length := (frames-1)*stride - 2*padding + kernel + extra
						if length <= 0 {
							continue
						}
						const channels, out = 3, 5
						w := &loader.CodecWeights{Tensors: map[string][]float32{"c.weight": make([]float32, channels*out*kernel), "c.bias": {.1, .2, .3, .4, .5}}, Shapes: map[string][]int{"c.weight": {channels, out, kernel}}}
						for i := range w.Tensors["c.weight"] {
							w.Tensors["c.weight"][i] = float32(math.Sin(float64(i) * .2))
						}
						d := &CodecDecoder{weights: w}
						x := signal{make([]float32, channels*frames), channels, frames}
						for i := range x.data {
							x.data[i] = float32(math.Cos(float64(i) * .4))
						}
						want, err := d.transposeReference(x, "c", stride, padding, extra)
						if err != nil {
							t.Fatal(err)
						}
						got, err := d.transpose(x, "c", stride, padding, extra)
						if err != nil {
							t.Fatal(err)
						}
						assertFloat32Exact(t, got.data, want.data)
					}
				}
			}
		}
	}
}
