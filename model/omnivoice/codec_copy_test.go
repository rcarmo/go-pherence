package omnivoice

import (
	"fmt"
	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
	"math"
	"testing"
)

func (d *CodecDecoder) scalarConvReference(x signal, name string, stride, padding, dilation int) (signal, error) {
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

func TestCodecConvContiguousCopyExact(t *testing.T) {
	for _, frames := range []int{1, 7, 63, 64, 65, 129} {
		for _, kernel := range []int{1, 3, 7} {
			for _, stride := range []int{1, 2} {
				for _, dilation := range []int{1, 3, 9} {
					const channels, out = 3, 5
					padding := dilation * (kernel - 1) / 2
					weights := make([]float32, out*channels*kernel)
					for i := range weights {
						weights[i] = float32(math.Sin(float64(i)))
					}
					bias := []float32{.1, .2, .3, .4, .5}
					w := &loader.CodecWeights{Tensors: map[string][]float32{"c.weight": weights, "c.bias": bias}, Shapes: map[string][]int{"c.weight": {out, channels, kernel}}}
					d := &CodecDecoder{weights: w}
					x := signal{make([]float32, channels*frames), channels, frames}
					for i := range x.data {
						x.data[i] = float32(math.Cos(float64(i)))
					}
					want, err := d.scalarConvReference(x, "c", stride, padding, dilation)
					if err != nil {
						t.Fatal(err)
					}
					got, err := d.conv(x, "c", stride, padding, dilation)
					if err != nil {
						t.Fatal(err)
					}
					assertFloat32Exact(t, got.data, want.data)
				}
			}
		}
	}
}
