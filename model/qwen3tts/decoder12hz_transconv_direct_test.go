package qwen3tts

import (
	"math"
	"math/rand"
	"slices"
	"testing"
)

// The prior full-raw-buffer implementation is an independent reduction-order
// oracle for the trimmed direct-output path. The tail must never affect output.
func referenceDecoderTransConv(c decoderTransConv1D, input []float32, length int) []float32 {
	rawLen := (length-1)*c.stride + c.k
	raw := make([]float32, c.outChannels*rawLen)
	for oc := 0; oc < c.outChannels; oc++ {
		for t := 0; t < rawLen; t++ {
			raw[oc*rawLen+t] = c.bias[oc]
		}
	}
	for ic := 0; ic < c.inChannels; ic++ {
		for t := 0; t < length; t++ {
			x := input[ic*length+t]
			for oc := 0; oc < c.outChannels; oc++ {
				w := c.weight[(ic*c.outChannels+oc)*c.k : (ic*c.outChannels+oc+1)*c.k]
				for tap := 0; tap < c.k; tap++ {
					raw[oc*rawLen+t*c.stride+tap] += x * w[tap]
				}
			}
		}
	}
	outLen := length * c.stride
	out := make([]float32, c.outChannels*outLen)
	for oc := 0; oc < c.outChannels; oc++ {
		copy(out[oc*outLen:(oc+1)*outLen], raw[oc*rawLen:oc*rawLen+outLen])
	}
	return out
}

func TestDecoderTransConvDirectMatchesTrimmedReference(t *testing.T) {
	rng := rand.New(rand.NewSource(1029))
	for _, shape := range []struct{ in, out, k, stride, length int }{
		{1, 1, 1, 1, 1}, {1, 1, 4, 2, 2}, {2, 3, 7, 2, 3}, {4, 2, 9, 3, 7}, {3, 3, 8, 8, 1}, {8, 4, 17, 5, 11},
	} {
		c := decoderTransConv1D{inChannels: shape.in, outChannels: shape.out, k: shape.k, stride: shape.stride}
		c.weight = make([]float32, shape.in*shape.out*shape.k)
		c.bias = make([]float32, shape.out)
		input := make([]float32, shape.in*shape.length)
		for i := range input {
			input[i] = rng.Float32()*2 - 1
		}
		for i := range c.weight {
			c.weight[i] = rng.Float32()*2 - 1
		}
		for i := range c.bias {
			c.bias[i] = rng.Float32()*2 - 1
		}
		original := slices.Clone(input)
		want := referenceDecoderTransConv(c, input, shape.length)
		got, n, err := c.forward(input, shape.length)
		if err != nil || n != shape.length*shape.stride || len(got) != len(want) {
			t.Fatalf("shape=%+v gotlen=%d wantlen=%d n=%d err=%v", shape, len(got), len(want), n, err)
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("shape=%+v sample=%d got=%g want=%g", shape, i, got[i], want[i])
			}
		}
		if !slices.Equal(original, input) {
			t.Fatalf("shape=%+v input mutated", shape)
		}
		if _, _, err := c.forward(input[:len(input)-1], shape.length); err == nil {
			t.Fatalf("shape=%+v accepted short input", shape)
		}
		if _, _, err := c.forward(input, 0); err == nil {
			t.Fatalf("shape=%+v accepted zero length", shape)
		}
	}
	valid := decoderTransConv1D{inChannels: 1, outChannels: 1, k: 2, stride: 1, weight: []float32{1, 2}, bias: []float32{0}}
	for _, mutate := range []func(*decoderTransConv1D){
		func(c *decoderTransConv1D) { c.inChannels = 0 },
		func(c *decoderTransConv1D) { c.outChannels = 0 },
		func(c *decoderTransConv1D) { c.stride = 0 },
		func(c *decoderTransConv1D) { c.k = 0 },
		func(c *decoderTransConv1D) { c.k = 1; c.stride = 2 },
		func(c *decoderTransConv1D) { c.weight = nil },
		func(c *decoderTransConv1D) { c.bias = nil },
	} {
		c := valid
		mutate(&c)
		if _, _, err := c.forward([]float32{1}, 1); err == nil {
			t.Fatalf("accepted malformed geometry %+v", c)
		}
	}
	if _, _, err := valid.forward([]float32{1}, int(^uint(0)>>1)); err == nil {
		t.Fatal("accepted overflowing length")
	}
}
