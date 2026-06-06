package ideogram4

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// FeatureMap is an NCHW=1 (single image) feature tensor: Channels x H x W,
// row-major as Data[c*H*W + y*W + x].
type FeatureMap struct {
	C    int
	H    int
	W    int
	Data []float32
}

func (f FeatureMap) at(c, y, x int) float32 { return f.Data[(c*f.H+y)*f.W+x] }

func (f FeatureMap) validate() error {
	if f.C <= 0 || f.H <= 0 || f.W <= 0 {
		return fmt.Errorf("ideogram4 vae feature shape %dx%dx%d", f.C, f.H, f.W)
	}
	if len(f.Data) != f.C*f.H*f.W {
		return fmt.Errorf("ideogram4 vae feature data=%d want=%d", len(f.Data), f.C*f.H*f.W)
	}
	return nil
}

// UnpatchifyLatents converts denoised DiT tokens [imageTokens, in_channels]
// back into a VAE latent feature map. The DiT operates on patchified latents
// where in_channels = latentChannels * patch_h * patch_w; this reverses the
// patch packing onto a (latentChannels, gridH*patchH, gridW*patchW) map.
//
// Patch packing order matches the reference reshape
// z.view(grid_h, grid_w, patch_h, patch_w, ae_channels): within a token the 128
// channels are laid out (patch_h, patch_w, ae_channels) with ae_channels
// innermost, i.e. index = (py*patchW + px)*latentChannels + ch.
func UnpatchifyLatents(tokens []float32, gridH, gridW, inChannels, latentChannels, patchH, patchW int) (FeatureMap, error) {
	if latentChannels*patchH*patchW != inChannels {
		return FeatureMap{}, fmt.Errorf("ideogram4 unpatchify: latent=%d patch=%dx%d != in=%d", latentChannels, patchH, patchW, inChannels)
	}
	imgTokens := gridH * gridW
	if len(tokens) != imgTokens*inChannels {
		return FeatureMap{}, fmt.Errorf("ideogram4 unpatchify: tokens=%d want %d*%d", len(tokens), imgTokens, inChannels)
	}
	if gpuVAEEnabled() {
		if out, err := unpatchifyLatentsGPU(tokens, gridH, gridW, inChannels, latentChannels, patchH, patchW); err == nil {
			return out, nil
		} else if gpuVAEStrict() {
			return FeatureMap{}, err
		}
	}
	H := gridH * patchH
	W := gridW * patchW
	out := FeatureMap{C: latentChannels, H: H, W: W, Data: make([]float32, latentChannels*H*W)}
	for r := 0; r < gridH; r++ {
		for c := 0; c < gridW; c++ {
			tok := r*gridW + c
			base := tok * inChannels
			for py := 0; py < patchH; py++ {
				for px := 0; px < patchW; px++ {
					for ch := 0; ch < latentChannels; ch++ {
						idx := base + (py*patchW+px)*latentChannels + ch
						y := r*patchH + py
						x := c*patchW + px
						out.Data[(ch*H+y)*W+x] = tokens[idx]
					}
				}
			}
		}
	}
	return out, nil
}

// Conv2DWeights holds an OIHW convolution kernel plus optional bias.
type Conv2DWeights struct {
	OutC   int
	InC    int
	KH     int
	KW     int
	Weight []float32 // OutC*InC*KH*KW
	Bias   []float32 // OutC or nil
}

func (w Conv2DWeights) validate() error {
	if w.OutC <= 0 || w.InC <= 0 || w.KH <= 0 || w.KW <= 0 {
		return fmt.Errorf("ideogram4 conv shape %dx%dx%dx%d", w.OutC, w.InC, w.KH, w.KW)
	}
	if len(w.Weight) != w.OutC*w.InC*w.KH*w.KW {
		return fmt.Errorf("ideogram4 conv weight=%d want=%d", len(w.Weight), w.OutC*w.InC*w.KH*w.KW)
	}
	if w.Bias != nil && len(w.Bias) != w.OutC {
		return fmt.Errorf("ideogram4 conv bias=%d want=%d", len(w.Bias), w.OutC)
	}
	return nil
}

// Conv2D applies a stride-1 same-padding (zero-pad) convolution using an
// im2col transform plus a SIMD GEMM, falling back to a scalar loop when the
// accelerated GEMM is unavailable.
func Conv2D(in FeatureMap, w Conv2DWeights) (FeatureMap, error) {
	if err := in.validate(); err != nil {
		return FeatureMap{}, err
	}
	if err := w.validate(); err != nil {
		return FeatureMap{}, err
	}
	if w.InC != in.C {
		return FeatureMap{}, fmt.Errorf("ideogram4 conv in_c=%d feature_c=%d", w.InC, in.C)
	}
	padY := (w.KH - 1) / 2
	padX := (w.KW - 1) / 2
	H, W := in.H, in.W
	HW := H * W
	K := in.C * w.KH * w.KW
	out := FeatureMap{C: w.OutC, H: H, W: W, Data: make([]float32, w.OutC*HW)}

	// im2col: col[hw, k] with k = ic*KH*KW + ky*KW + kx (matches OIHW weight).
	col := make([]float32, HW*K)
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			hw := y*W + x
			base := hw * K
			for ic := 0; ic < in.C; ic++ {
				for ky := 0; ky < w.KH; ky++ {
					iy := y + ky - padY
					if iy < 0 || iy >= H {
						continue
					}
					row := (ic*H + iy) * W
					kbase := base + (ic*w.KH+ky)*w.KW
					for kx := 0; kx < w.KW; kx++ {
						ix := x + kx - padX
						if ix < 0 || ix >= W {
							continue
						}
						col[kbase+kx] = in.Data[row+ix]
					}
				}
			}
		}
	}

	// outT[hw, oc] = col[hw, :] . weight[oc, :]
	outT := make([]float32, HW*w.OutC)
	if !simd.GemmRows(outT, col, w.Weight, HW, w.OutC, K) {
		for hw := 0; hw < HW; hw++ {
			cb := hw * K
			for oc := 0; oc < w.OutC; oc++ {
				wb := oc * K
				var acc float32
				for k := 0; k < K; k++ {
					acc += col[cb+k] * w.Weight[wb+k]
				}
				outT[hw*w.OutC+oc] = acc
			}
		}
	}

	// transpose outT[hw, oc] -> out[oc, hw] and add bias.
	for oc := 0; oc < w.OutC; oc++ {
		var bias float32
		if w.Bias != nil {
			bias = w.Bias[oc]
		}
		dst := out.Data[oc*HW : (oc+1)*HW]
		for hw := 0; hw < HW; hw++ {
			dst[hw] = outT[hw*w.OutC+oc] + bias
		}
	}
	return out, nil
}

// GroupNorm applies group normalization with per-channel affine (gamma/beta),
// matching diffusers VAE GroupNorm(num_groups, channels).
func GroupNorm(in FeatureMap, groups int, gamma, beta []float32, eps float32) (FeatureMap, error) {
	if err := in.validate(); err != nil {
		return FeatureMap{}, err
	}
	if groups <= 0 || in.C%groups != 0 {
		return FeatureMap{}, fmt.Errorf("ideogram4 groupnorm channels=%d groups=%d", in.C, groups)
	}
	if len(gamma) != in.C || len(beta) != in.C {
		return FeatureMap{}, fmt.Errorf("ideogram4 groupnorm affine len gamma=%d beta=%d want=%d", len(gamma), len(beta), in.C)
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
			ga, be := float64(gamma[c]), float64(beta[c])
			for i := 0; i < hw; i++ {
				norm := (float64(in.Data[c*hw+i]) - mean) * inv
				out.Data[c*hw+i] = float32(norm*ga + be)
			}
		}
	}
	return out, nil
}

// SiLUMap applies SiLU activation in place.
func (f FeatureMap) SiLUMap() {
	if gpuVAEEnabled() && len(f.Data) > 0 {
		out := make([]float32, len(f.Data))
		if err := siluGPU(out, f.Data); err == nil {
			copy(f.Data, out)
			return
		} else if gpuVAEStrict() {
			panic(err)
		}
	}
	for i := range f.Data {
		f.Data[i] = siluScalar(f.Data[i])
	}
}

// UpsampleNearest doubles spatial resolution by nearest-neighbour replication,
// as used by diffusers VAE upsample blocks (followed by a conv).
func UpsampleNearest(in FeatureMap, factor int) (FeatureMap, error) {
	if err := in.validate(); err != nil {
		return FeatureMap{}, err
	}
	if factor <= 0 {
		return FeatureMap{}, fmt.Errorf("ideogram4 upsample factor=%d", factor)
	}
	if gpuVAEEnabled() {
		if out, err := upsampleNearestGPU(in, factor); err == nil {
			return out, nil
		} else if gpuVAEStrict() {
			return FeatureMap{}, err
		}
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
	return out, nil
}

// AddResidual adds b into a element-wise (shapes must match).
func (f FeatureMap) AddResidual(b FeatureMap) error {
	if f.C != b.C || f.H != b.H || f.W != b.W {
		return fmt.Errorf("ideogram4 vae residual shape mismatch %dx%dx%d vs %dx%dx%d", f.C, f.H, f.W, b.C, b.H, b.W)
	}
	for i := range f.Data {
		f.Data[i] += b.Data[i]
	}
	return nil
}
