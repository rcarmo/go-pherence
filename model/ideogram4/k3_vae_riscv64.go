//go:build riscv64

package ideogram4

import (
	"fmt"
	"math"
	"sync"

	simdruntime "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

func k3GroupNorm(in FeatureMap, groups int, gamma, beta []float32, eps float32) (FeatureMap, bool, error) {
	if !k3Enabled() || in.C <= 0 || groups <= 0 || in.C%groups != 0 {
		return FeatureMap{}, false, nil
	}
	chPerGroup := in.C / groups
	HW := in.H * in.W
	out := FeatureMap{C: in.C, H: in.H, W: in.W, Data: make([]float32, len(in.Data))}
	nw := k3Threads()
	if nw > groups {
		nw = groups
	}
	if nw <= 1 {
		k3GroupNormWork(in.Data, out.Data, gamma, beta, groups, chPerGroup, HW, eps, 0, groups)
	} else {
		var wg sync.WaitGroup
		wg.Add(nw)
		for wid := 0; wid < nw; wid++ {
			g0 := wid * groups / nw
			g1 := (wid + 1) * groups / nw
			go func() {
				defer wg.Done()
				k3GroupNormWork(in.Data, out.Data, gamma, beta, groups, chPerGroup, HW, eps, g0, g1)
			}()
		}
		wg.Wait()
	}
	return out, true, nil
}

func k3GroupNormWork(inData, outData, gamma, beta []float32, groups, chPerGroup, HW int, eps float32, g0, g1 int) {
	for g := g0; g < g1; g++ {
		c0 := g * chPerGroup
		c1 := c0 + chPerGroup
		n := chPerGroup * HW
		var sum, sq float64
		for c := c0; c < c1; c++ {
			off := c * HW
			for hw := 0; hw < HW; hw++ {
				v := float64(inData[off+hw])
				sum += v
				sq += v * v
			}
		}
		mean := sum / float64(n)
		variance := sq/float64(n) - mean*mean
		invStd := float32(1.0 / math.Sqrt(float64(eps)+variance))
		meanF := float32(mean)
		for c := c0; c < c1; c++ {
			g := gamma[c]
			b := beta[c]
			off := c * HW
			for hw := 0; hw < HW; hw++ {
				outData[off+hw] = (inData[off+hw]-meanF)*invStd*g + b
			}
		}
	}
}

func k3VaeRGBConvert(in FeatureMap) ([]byte, bool) {
	if !k3Enabled() || in.C != 3 || in.H <= 0 || in.W <= 0 {
		return nil, false
	}
	HW := in.H * in.W
	rgb := make([]byte, 3*HW)
	nw := k3Threads()
	if nw > in.H {
		nw = in.H
	}
	if nw <= 1 {
		k3RGBWork(in.Data, rgb, HW, in.W, 0, in.H)
	} else {
		var wg sync.WaitGroup
		wg.Add(nw)
		for wid := 0; wid < nw; wid++ {
			y0 := wid * in.H / nw
			y1 := (wid + 1) * in.H / nw
			go func() {
				defer wg.Done()
				k3RGBWork(in.Data, rgb, HW, in.W, y0, y1)
			}()
		}
		wg.Wait()
	}
	return rgb, true
}

func k3RGBWork(data []float32, rgb []byte, HW, W, y0, y1 int) {
	for y := y0; y < y1; y++ {
		for x := 0; x < W; x++ {
			hw := y*W + x
			for c := 0; c < 3; c++ {
				v := data[c*HW+hw]
				v = (v + 1) * 0.5 * 255
				if v < 0 {
					v = 0
				}
				if v > 255 {
					v = 255
				}
				rgb[hw*3+c] = byte(v)
			}
		}
	}
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
