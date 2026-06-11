//go:build riscv64

package ideogram4

import (
	"fmt"
	"math"

	simdruntime "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func k3GroupNorm(in FeatureMap, groups int, gamma, beta []float32, eps float32) (FeatureMap, bool, error) {
	if !k3Enabled() {
		return FeatureMap{}, false, nil
	}
	if err := in.validate(); err != nil {
		return FeatureMap{}, true, err
	}
	if groups <= 0 || in.C%groups != 0 {
		return FeatureMap{}, true, fmt.Errorf("ideogram4 K3 groupnorm channels=%d groups=%d", in.C, groups)
	}
	if len(gamma) != in.C || len(beta) != in.C {
		return FeatureMap{}, true, fmt.Errorf("ideogram4 K3 groupnorm affine len gamma=%d beta=%d want=%d", len(gamma), len(beta), in.C)
	}
	chPerGroup := in.C / groups
	hw := in.H * in.W
	out := FeatureMap{C: in.C, H: in.H, W: in.W, Data: make([]float32, len(in.Data))}
	for g := 0; g < groups; g++ {
		c0 := g * chPerGroup
		n := chPerGroup * hw
		var mean float64
		for c := c0; c < c0+chPerGroup; c++ {
			for i := 0; i < hw; i++ {
				mean += float64(in.Data[c*hw+i])
			}
		}
		mean /= float64(n)
		var variance float64
		for c := c0; c < c0+chPerGroup; c++ {
			for i := 0; i < hw; i++ {
				d := float64(in.Data[c*hw+i]) - mean
				variance += d * d
			}
		}
		variance /= float64(n)
		inv := 1 / math.Sqrt(variance+float64(eps))
		for c := c0; c < c0+chPerGroup; c++ {
			base := c * hw
			dst := out.Data[base : base+hw]
			for i := 0; i < hw; i++ {
				dst[i] = in.Data[base+i] - float32(mean)
			}
			simdruntime.VecScale(dst, dst, float32(inv)*gamma[c])
			be := beta[c]
			for i := range dst {
				dst[i] += be
			}
		}
	}
	return out, true, nil
}

func k3UpsampleNearest(in FeatureMap, factor int) (FeatureMap, bool, error) {
	if !k3Enabled() {
		return FeatureMap{}, false, nil
	}
	if err := in.validate(); err != nil {
		return FeatureMap{}, true, err
	}
	if factor <= 0 {
		return FeatureMap{}, true, fmt.Errorf("ideogram4 K3 upsample factor=%d", factor)
	}
	H, W := in.H*factor, in.W*factor
	out := FeatureMap{C: in.C, H: H, W: W, Data: make([]float32, in.C*H*W)}
	for c := 0; c < in.C; c++ {
		for y := 0; y < H; y++ {
			sy := y / factor
			for x := 0; x < W; x++ {
				sx := x / factor
				out.Data[(c*H+y)*W+x] = in.at(c, sy, sx)
			}
		}
	}
	return out, true, nil
}

func k3RGB(f FeatureMap) (Image, bool) {
	if !k3Enabled() || f.C != 3 || f.H <= 0 || f.W <= 0 || len(f.Data) < 3*f.H*f.W {
		return Image{}, false
	}
	HW := f.H * f.W
	rgb := make([]byte, HW*3)
	scaled := make([]float32, 3*HW)
	// RVV-backed scaling where available: x in [-1,1] -> x*127.5, then scalar
	// offset/clamp/interleave. A future fused RVV kernel should combine all of it.
	simdruntime.VecScale(scaled, f.Data[:3*HW], 127.5)
	for p := 0; p < HW; p++ {
		for c := 0; c < 3; c++ {
			v := scaled[c*HW+p] + 127.5
			if v < 0 {
				v = 0
			}
			if v > 255 {
				v = 255
			}
			rgb[p*3+c] = byte(v + 0.5)
		}
	}
	return Image{Width: f.W, Height: f.H, RGB: rgb}, true
}
