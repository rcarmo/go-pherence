package zimage

import (
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/internal/checked"
	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
)

// LatentShape describes the NCHW tensor passed to the Z-Image transformer.
// It does not allocate random latents or execute the transformer or VAE.
type LatentShape struct {
	Batch, Channels, Height, Width int
}

// DefaultLatentShape implements the pinned pipeline's image admission and
// prepare_latents shape rule. With four VAE blocks, image dimensions must be
// divisible by 16 and the latent spatial dimensions are image dimensions / 8.
func DefaultLatentShape(vae loaderconfig.ZImageVAEConfig, transformer loaderconfig.ZImageTransformerConfig, batch, imageHeight, imageWidth int) (LatentShape, error) {
	if !pinnedVAE(vae) || transformer.InChannels != vae.LatentChannels || batch <= 0 || imageHeight <= 0 || imageWidth <= 0 || imageHeight%16 != 0 || imageWidth%16 != 0 {
		return LatentShape{}, fmt.Errorf("zimage: unsupported latent geometry or VAE")
	}
	shape := LatentShape{Batch: batch, Channels: vae.LatentChannels, Height: imageHeight / 8, Width: imageWidth / 8}
	if shape.Numel() == 0 {
		return LatentShape{}, fmt.Errorf("zimage: latent shape overflow")
	}
	return shape, nil
}

// Numel returns zero for malformed or overflowing latent shapes.
func (s LatentShape) Numel() int {
	n := 1
	for _, d := range [...]int{s.Batch, s.Channels, s.Height, s.Width} {
		if d <= 0 {
			return 0
		}
		var ok bool
		n, ok = checked.MulInt(n, d)
		if !ok {
			return 0
		}
	}
	return n
}

// PrepareVAEDecodeInputInto applies the pinned Z-Image decode boundary:
// latent / scaling_factor + shift_factor. It does not decode an RGB image.
// Shapes, values, and buffer overlap are checked before writing dst. Exact
// dst==latents is allowed; other overlap is rejected.
func PrepareVAEDecodeInputInto(dst, latents []float32, shape LatentShape, vae loaderconfig.ZImageVAEConfig) error {
	if !pinnedVAE(vae) || shape.Channels != vae.LatentChannels || shape.Numel() == 0 || len(dst) != shape.Numel() || len(latents) != len(dst) {
		return fmt.Errorf("zimage: invalid VAE decode input shape or config")
	}
	if overlap(dst, latents) && &dst[0] != &latents[0] {
		return fmt.Errorf("zimage: partial VAE decode input overlap")
	}
	scale, shift := float32(vae.ScalingFactor), float32(vae.ShiftFactor)
	for _, x := range latents {
		if !finite(x) || !finite(x/scale+shift) {
			return fmt.Errorf("zimage: non-finite VAE decode input or result")
		}
	}
	for i, x := range latents {
		dst[i] = x/scale + shift
	}
	return nil
}

func pinnedVAE(c loaderconfig.ZImageVAEConfig) bool {
	if c.ClassName != "AutoencoderKL" || c.LatentChannels != 16 || c.ScalingFactor != 0.3611 || c.ShiftFactor != 0.1159 || len(c.BlockOutChannels) != 4 || math.IsInf(c.ScalingFactor, 0) || math.IsNaN(c.ShiftFactor) {
		return false
	}
	for i, n := range [...]int{128, 256, 512, 512} {
		if c.BlockOutChannels[i] != n {
			return false
		}
	}
	return true
}
