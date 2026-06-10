package ideogram4

import (
	"fmt"
	"math"
	"os"
	"time"
)

// F32TensorSource provides float32 tensors by name (satisfied by
// *safetensors.File and *safetensors.ShardedFile via GetFloat32).
type F32TensorSource interface {
	GetFloat32(name string) ([]float32, []int, error)
}

// VAEDecoder is a native AutoencoderKLFlux2 decoder graph loaded from VAE
// safetensors weights. It maps latent feature maps to RGB images.
type VAEDecoder struct {
	src              F32TensorSource
	groups           int
	eps              float32
	blockOutChannels []int
	layersPerBlock   int
	latentChannels   int
	scalingFactor    float32
	shiftFactor      float32
	usePostQuantConv bool
	midAddAttention  bool
}

// VAEDecoderOptions configures decoder construction from the VAE config.
type VAEDecoderOptions struct {
	BlockOutChannels []int
	LayersPerBlock   int
	LatentChannels   int
	NormNumGroups    int
	ScalingFactor    float32
	ShiftFactor      float32
	UsePostQuantConv bool
	MidAddAttention  bool
}

// NewVAEDecoder builds a decoder bound to a weight source.
func NewVAEDecoder(src F32TensorSource, opt VAEDecoderOptions) (*VAEDecoder, error) {
	if src == nil {
		return nil, fmt.Errorf("ideogram4 vae: nil weight source")
	}
	if len(opt.BlockOutChannels) == 0 || opt.LayersPerBlock <= 0 || opt.LatentChannels <= 0 {
		return nil, fmt.Errorf("ideogram4 vae: invalid options %+v", opt)
	}
	groups := opt.NormNumGroups
	if groups <= 0 {
		groups = 32
	}
	scale := opt.ScalingFactor
	if scale == 0 {
		scale = 1
	}
	return &VAEDecoder{
		src:              src,
		groups:           groups,
		eps:              1e-6,
		blockOutChannels: append([]int(nil), opt.BlockOutChannels...),
		layersPerBlock:   opt.LayersPerBlock,
		latentChannels:   opt.LatentChannels,
		scalingFactor:    scale,
		shiftFactor:      opt.ShiftFactor,
		usePostQuantConv: opt.UsePostQuantConv,
		midAddAttention:  opt.MidAddAttention,
	}, nil
}

func (d *VAEDecoder) loadConv(prefix string) (Conv2DWeights, error) {
	w, ws, err := d.src.GetFloat32(prefix + ".weight")
	if err != nil {
		return Conv2DWeights{}, fmt.Errorf("ideogram4 vae conv %q: %w", prefix, err)
	}
	if len(ws) != 4 {
		return Conv2DWeights{}, fmt.Errorf("ideogram4 vae conv %q shape=%v want OIHW", prefix, ws)
	}
	cw := Conv2DWeights{OutC: ws[0], InC: ws[1], KH: ws[2], KW: ws[3], Weight: w}
	if b, _, err := d.src.GetFloat32(prefix + ".bias"); err == nil {
		cw.Bias = b
	}
	return cw, nil
}

func (d *VAEDecoder) loadAffine(prefix string, channels int) (gamma, beta []float32, err error) {
	gamma, _, err = d.src.GetFloat32(prefix + ".weight")
	if err != nil {
		return nil, nil, fmt.Errorf("ideogram4 vae norm %q: %w", prefix, err)
	}
	beta, _, err = d.src.GetFloat32(prefix + ".bias")
	if err != nil {
		return nil, nil, fmt.Errorf("ideogram4 vae norm %q bias: %w", prefix, err)
	}
	if len(gamma) != channels || len(beta) != channels {
		return nil, nil, fmt.Errorf("ideogram4 vae norm %q len gamma=%d beta=%d want=%d", prefix, len(gamma), len(beta), channels)
	}
	return gamma, beta, nil
}

func (d *VAEDecoder) conv(in FeatureMap, prefix string) (FeatureMap, error) {
	w, err := d.loadConv(prefix)
	if err != nil {
		return FeatureMap{}, err
	}
	return Conv2D(in, w)
}

func (d *VAEDecoder) groupNorm(in FeatureMap, prefix string) (FeatureMap, error) {
	gamma, beta, err := d.loadAffine(prefix, in.C)
	if err != nil {
		return FeatureMap{}, err
	}
	return GroupNorm(in, d.groups, gamma, beta, d.eps)
}

// resnet applies a diffusers ResnetBlock2D: norm1->silu->conv1->norm2->silu->
// conv2 plus (optional 1x1) shortcut.
func (d *VAEDecoder) resnet(in FeatureMap, prefix string) (FeatureMap, error) {
	h, err := d.groupNorm(in, prefix+".norm1")
	if err != nil {
		return FeatureMap{}, err
	}
	h.SiLUMap()
	if h, err = d.conv(h, prefix+".conv1"); err != nil {
		return FeatureMap{}, err
	}
	if h, err = d.groupNorm(h, prefix+".norm2"); err != nil {
		return FeatureMap{}, err
	}
	h.SiLUMap()
	if h, err = d.conv(h, prefix+".conv2"); err != nil {
		return FeatureMap{}, err
	}
	shortcut := in
	if sw, _, scErr := d.src.GetFloat32(prefix + ".conv_shortcut.weight"); scErr == nil {
		cw := Conv2DWeights{OutC: h.C, InC: in.C, KH: 1, KW: 1, Weight: sw}
		if b, _, bErr := d.src.GetFloat32(prefix + ".conv_shortcut.bias"); bErr == nil {
			cw.Bias = b
		}
		if shortcut, err = Conv2D(in, cw); err != nil {
			return FeatureMap{}, err
		}
	}
	if err := h.AddResidual(shortcut); err != nil {
		return FeatureMap{}, err
	}
	return h, nil
}

// attention applies the diffusers VAE mid-block self-attention (single head)
// over spatial positions, with a group-norm pre-norm and residual.
func (d *VAEDecoder) attention(in FeatureMap, prefix string) (FeatureMap, error) {
	h, err := d.groupNorm(in, prefix+".group_norm")
	if err != nil {
		return FeatureMap{}, err
	}
	C := in.C
	// linear projections are stored as [C,C] (.weight) + [C] (.bias).
	q, err := d.spatialLinear(h, prefix+".to_q")
	if err != nil {
		return FeatureMap{}, err
	}
	k, err := d.spatialLinear(h, prefix+".to_k")
	if err != nil {
		return FeatureMap{}, err
	}
	v, err := d.spatialLinear(h, prefix+".to_v")
	if err != nil {
		return FeatureMap{}, err
	}
	scale := float32(1 / math.Sqrt(float64(C)))
	attnOut, err := vaeSpatialAttention(q, k, v, scale)
	if err != nil {
		return FeatureMap{}, err
	}
	out, err := d.spatialLinear(attnOut, prefix+".to_out.0")
	if err != nil {
		return FeatureMap{}, err
	}
	if err := out.AddResidual(in); err != nil {
		return FeatureMap{}, err
	}
	return out, nil
}

func vaeSpatialAttention(q, k, v FeatureMap, scale float32) (FeatureMap, error) {
	C, HW := q.C, q.H*q.W
	if gpuVAEEnabled() {
		if out, err := vaeSpatialAttentionGPU(q, k, v, scale); err == nil {
			return out, nil
		} else if gpuVAEStrict() {
			return FeatureMap{}, err
		}
	}
	attnOut := FeatureMap{C: C, H: q.H, W: q.W, Data: make([]float32, C*HW)}
	scores := make([]float32, HW)
	for i := 0; i < HW; i++ {
		for j := 0; j < HW; j++ {
			var dot float32
			for c := 0; c < C; c++ {
				dot += q.Data[c*HW+i] * k.Data[c*HW+j]
			}
			scores[j] = dot * scale
		}
		softmaxFallback(scores)
		for c := 0; c < C; c++ {
			var acc float32
			for j := 0; j < HW; j++ {
				acc += scores[j] * v.Data[c*HW+j]
			}
			attnOut.Data[c*HW+i] = acc
		}
	}
	return attnOut, nil
}

// spatialLinear applies a [outC,inC] weight + bias to every spatial position.
func (d *VAEDecoder) spatialLinear(in FeatureMap, prefix string) (FeatureMap, error) {
	w, ws, err := d.src.GetFloat32(prefix + ".weight")
	if err != nil {
		return FeatureMap{}, fmt.Errorf("ideogram4 vae linear %q: %w", prefix, err)
	}
	if len(ws) != 2 || ws[1] != in.C {
		return FeatureMap{}, fmt.Errorf("ideogram4 vae linear %q shape=%v in_c=%d", prefix, ws, in.C)
	}
	outC := ws[0]
	bias, _, _ := d.src.GetFloat32(prefix + ".bias")
	HW := in.H * in.W
	out := FeatureMap{C: outC, H: in.H, W: in.W, Data: make([]float32, outC*HW)}
	for oc := 0; oc < outC; oc++ {
		var b float32
		if bias != nil && oc < len(bias) {
			b = bias[oc]
		}
		for p := 0; p < HW; p++ {
			var acc float32
			for ic := 0; ic < in.C; ic++ {
				acc += w[oc*in.C+ic] * in.Data[ic*HW+p]
			}
			out.Data[oc*HW+p] = acc + b
		}
	}
	return out, nil
}

// Decode converts a latent feature map (latentChannels x H x W) into an RGB
// image. Latents are de-scaled (latent/scale + shift) before the decoder.
func (d *VAEDecoder) Decode(latents FeatureMap) (Image, error) {
	if d == nil {
		return Image{}, ErrRuntimeNotImplemented
	}
	if err := latents.validate(); err != nil {
		return Image{}, err
	}
	if latents.C != d.latentChannels {
		return Image{}, fmt.Errorf("ideogram4 vae decode latent_c=%d want=%d", latents.C, d.latentChannels)
	}
	trace := os.Getenv("GO_PHERENCE_IDEOGRAM4_VAE_TIMING") == "1"
	traceStart := time.Now()
	last := traceStart
	mark := func(name string, f FeatureMap) {
		if trace {
			fmt.Fprintf(os.Stderr, "vae_timing %s=%s total=%s shape=%dx%dx%d\n", name, time.Since(last), time.Since(traceStart), f.C, f.H, f.W)
			last = time.Now()
		}
	}
	// Latents are expected already denormalized (see DenormalizeLatents); the
	// KL VAE decoder applies no additional scaling.
	z := FeatureMap{C: latents.C, H: latents.H, W: latents.W, Data: append([]float32(nil), latents.Data...)}
	mark("copy_latents", z)
	var err error
	if d.usePostQuantConv {
		if z, err = d.conv(z, "post_quant_conv"); err != nil {
			return Image{}, err
		}
		mark("post_quant_conv", z)
	}
	h, err := d.conv(z, "decoder.conv_in")
	if err != nil {
		return Image{}, err
	}
	mark("conv_in", h)
	// mid block: resnet, [attention], resnet.
	if h, err = d.resnet(h, "decoder.mid_block.resnets.0"); err != nil {
		return Image{}, err
	}
	mark("mid_resnet_0", h)
	if d.midAddAttention {
		if h, err = d.attention(h, "decoder.mid_block.attentions.0"); err != nil {
			return Image{}, err
		}
		mark("mid_attention", h)
	}
	if h, err = d.resnet(h, "decoder.mid_block.resnets.1"); err != nil {
		return Image{}, err
	}
	mark("mid_resnet_1", h)
	// up blocks (reversed channel order). diffusers names them 0..N-1 in the
	// reversed order, each with layers_per_block+1 resnets and an upsampler
	// (except the last block).
	numBlocks := len(d.blockOutChannels)
	for b := 0; b < numBlocks; b++ {
		for r := 0; r <= d.layersPerBlock; r++ {
			prefix := fmt.Sprintf("decoder.up_blocks.%d.resnets.%d", b, r)
			if h, err = d.resnet(h, prefix); err != nil {
				return Image{}, err
			}
			mark(fmt.Sprintf("up%d_resnet_%d", b, r), h)
		}
		if b < numBlocks-1 {
			if h, err = UpsampleNearest(h, 2); err != nil {
				return Image{}, err
			}
			mark(fmt.Sprintf("up%d_upsample", b), h)
			if h, err = d.conv(h, fmt.Sprintf("decoder.up_blocks.%d.upsamplers.0.conv", b)); err != nil {
				return Image{}, err
			}
			mark(fmt.Sprintf("up%d_upsample_conv", b), h)
		}
	}
	if h, err = d.groupNorm(h, "decoder.conv_norm_out"); err != nil {
		return Image{}, err
	}
	mark("conv_norm_out", h)
	h.SiLUMap()
	mark("silu_out", h)
	if h, err = d.conv(h, "decoder.conv_out"); err != nil {
		return Image{}, err
	}
	mark("conv_out", h)
	if h.C != 3 {
		return Image{}, fmt.Errorf("ideogram4 vae decode out channels=%d want 3", h.C)
	}
	img := featureMapToImage(h)
	if trace {
		fmt.Fprintf(os.Stderr, "vae_timing rgb=%s total=%s shape=%dx%d\n", time.Since(last), time.Since(traceStart), img.Width, img.Height)
	}
	return img, nil
}

// featureMapToImage converts a [3,H,W] float map in [-1,1] to 8-bit RGB.
func featureMapToImage(f FeatureMap) Image {
	HW := f.H * f.W
	rgb := make([]byte, HW*3)
	if gpuVAEEnabled() {
		if vals, err := rgbClampGPU(f); err == nil {
			for i, v := range vals {
				rgb[i] = byte(v + 0.5)
			}
			return Image{Width: f.W, Height: f.H, RGB: rgb}
		} else if gpuVAEStrict() {
			panic(err)
		}
	}
	for p := 0; p < HW; p++ {
		for c := 0; c < 3; c++ {
			v := (f.Data[c*HW+p]*0.5 + 0.5) * 255
			if v < 0 {
				v = 0
			}
			if v > 255 {
				v = 255
			}
			rgb[p*3+c] = byte(v + 0.5)
		}
	}
	return Image{Width: f.W, Height: f.H, RGB: rgb}
}
