package qwen3tts

import (
	"math"
	"math/rand"
	"slices"
	"testing"
)

// Keep the original scalar reduction order as the oracle for indexed views.
func referenceDecoderConv1D(c decoderConv1D, input []float32, length int) []float32 {
	out := make([]float32, c.outChannels*length)
	groups := c.groups()
	for oc := 0; oc < c.outChannels; oc++ {
		group := oc * groups / c.outChannels
		for t := 0; t < length; t++ {
			sum := c.bias[oc]
			for icg := 0; icg < c.inChannels; icg++ {
				ic := group*c.inChannels + icg
				for tap := 0; tap < c.k; tap++ {
					src := t - c.dilation*(c.k-1-tap)
					if src >= 0 {
						sum += input[ic*length+src] * c.weight[(oc*c.inChannels+icg)*c.k+tap]
					}
				}
			}
			out[oc*length+t] = sum
		}
	}
	return out
}

func TestDecoderConvIndexedMatchesScalarReduction(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	for _, shape := range []struct{ in, out, kernel, dilation, length int }{
		{1, 1, 1, 1, 1}, {1, 3, 1, 3, 7}, {1, 3, 3, 1, 7}, // depthwise groups
		{3, 4, 7, 3, 17}, {2, 2, 5, 9, 4}, // boundary with most taps padded
		{8, 4, 1, 1, 33}, {16, 8, 1, 3, 31}, // one-tap dense, including dilation
		{8, 4, 3, 1, 33}, {16, 8, 7, 3, 31},
	} {
		c := decoderConv1D{inChannels: shape.in, outChannels: shape.out, k: shape.kernel, dilation: shape.dilation}
		c.weight = make([]float32, shape.in*shape.out*shape.kernel)
		c.bias = make([]float32, shape.out)
		input := make([]float32, shape.in*c.groups()*shape.length)
		for i := range c.weight {
			c.weight[i] = rng.Float32()*2 - 1
		}
		for i := range c.bias {
			c.bias[i] = rng.Float32()*2 - 1
		}
		for i := range input {
			input[i] = rng.Float32()*2 - 1
		}
		original := slices.Clone(input)
		want := referenceDecoderConv1D(c, input, shape.length)
		got, n, err := c.forward(input, shape.length)
		if err != nil || n != shape.length || len(got) != len(want) {
			t.Fatalf("shape=%+v len=%d/%d n=%d err=%v", shape, len(got), len(want), n, err)
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("shape=%+v sample=%d got=%g want=%g", shape, i, got[i], want[i])
			}
		}
		if !slices.Equal(input, original) {
			t.Fatalf("shape=%+v input mutated", shape)
		}
		if _, _, err := c.forward(input[:len(input)-1], shape.length); err == nil {
			t.Fatalf("shape=%+v accepted short input", shape)
		}
		if _, _, err := c.forward(input, 0); err == nil {
			t.Fatalf("shape=%+v accepted zero length", shape)
		}
	}
}
