package community1

import (
	"context"
	"math"
	"testing"
)

func TestWeSpeakerTiledConvolution(t *testing.T) {
	for _, shape := range []CHWShape{{1, 1, 1}, {1, 3, 33}, {3, 5, 7}, {4, 8, 17}, {7, 9, 65}} {
		for _, kernel := range []int{1, 3} {
			for _, stride := range []int{1, 2} {
				out := CHWShape{5, (shape.Frequency + stride - 1) / stride, (shape.Frames + stride - 1) / stride}
				pad := kernel / 2
				k := shape.Channels * kernel * kernel
				x := make([]float32, shape.Channels*shape.Frequency*shape.Frames)
				w := make([]float32, out.Channels*k)
				for i := range x {
					x[i] = float32(math.Sin(float64(i) * .17))
				}
				for i := range w {
					w[i] = float32(math.Cos(float64(i) * .13))
				}
				got, e := weSpeakerBlockConv(context.Background(), x, shape, out, w, kernel, stride, pad, WeSpeakerBlockGEMM)
				if e != nil {
					t.Fatal(e)
				}
				for c := 0; c < out.Channels; c++ {
					for f := 0; f < out.Frequency; f++ {
						for frame := 0; frame < out.Frames; frame++ {
							var sum float32
							for in := 0; in < shape.Channels; in++ {
								for kf := 0; kf < kernel; kf++ {
									for kt := 0; kt < kernel; kt++ {
										sf, st := f*stride+kf-pad, frame*stride+kt-pad
										var v float32
										if sf >= 0 && sf < shape.Frequency && st >= 0 && st < shape.Frames {
											v = x[(in*shape.Frequency+sf)*shape.Frames+st]
										}
										sum = sincNetFMA32(w[c*k+(in*kernel+kf)*kernel+kt], v, sum)
									}
								}
							}
							at := (c*out.Frequency+f)*out.Frames + frame
							if math.Float32bits(got[at]) != math.Float32bits(sum) {
								t.Fatal("tile order/geometry", shape, kernel, stride, at, got[at], sum)
							}
						}
					}
				}
			}
		}
	}
}
func TestWeSpeakerTiledFullDepthOracle(t *testing.T) {
	for _, c := range loadResNetFixtures(t) {
		m, e := NewWeSpeakerResNet34(context.Background(), c.Config, c.Weights)
		if e != nil {
			t.Fatal(e)
		}
		seen := 0
		x, s, e := m.ForwardFramesObserved(context.Background(), c.Input, c.Frames, WeSpeakerBlockGEMM, func(stage, block int, shape CHWShape, v []float32) {
			b := c.Boundaries[seen]
			seen++
			if b.Stage != stage || b.Block != block || b.Shape != shape {
				t.Fatal("geometry")
			}
			closePool(t, v, b.Values)
		})
		if e != nil || seen != 17 {
			t.Fatal(e, seen)
		}
		for _, mask := range c.MaskCases {
			r, e := m.ForwardEmbedding(context.Background(), x, s, mask.Masks, mask.Speakers, mask.MaskFrames, WeSpeakerBlockGEMM)
			if e != nil {
				t.Fatal(e)
			}
			checkEmbedding(t, r, mask)
		}
	}
}
