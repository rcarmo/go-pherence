package pockettts

import (
	"fmt"
	"math"
	"testing"
)

func TestCausalConvNarrowLargeKMatchesScalar(t *testing.T) {
	const in, out, kernel, stride = 512, 3, 32, 16
	weight := make([]float32, out*in*kernel)
	for i := range weight {
		weight[i] = float32((i*7)%23-11) * .001
	}
	conv := CausalConv1D{Weight: weight, In: in, Out: out, Kernel: kernel, Stride: stride, Dilation: 1}
	for width := 1; width < 16; width++ {
		t.Run(fmt.Sprintf("n%d", width), func(t *testing.T) {
			length := width * stride
			input := make([]float32, in*length)
			for i := range input {
				input[i] = float32((i*5)%17-8) * .02
			}
			got, gotLen, err := coldConvForward(conv, input, length)
			if err != nil {
				t.Fatal(err)
			}
			if gotLen != width || len(got) != out*width {
				t.Fatalf("shape %d %d", gotLen, len(got))
			}
			prev := kernel - stride
			for o := 0; o < out; o++ {
				for x := 0; x < width; x++ {
					want := float32(0)
					for ic := 0; ic < in; ic++ {
						for tap := 0; tap < kernel; tap++ {
							source := x*stride + tap - prev
							if source >= 0 && source < length {
								want += weight[(o*in+ic)*kernel+tap] * input[ic*length+source]
							}
						}
					}
					if math.Abs(float64(got[o*width+x]-want)) > 2e-5 {
						t.Fatalf("out[%d,%d]=%g want=%g", o, x, got[o*width+x], want)
					}
				}
			}
		})
	}
}
