package hunyuan3d

import (
	"fmt"
	"image"
	"image/color"
	"math"
)

// ImagePreprocessConfig captures the ImageProcessorV2 knobs used by the
// Hunyuan3D shape conditioner. The implementation is intentionally small and
// deterministic; it is a Go parity target for scripts/hunyuan3d_image_fixture.py,
// not a vision encoder.
type ImagePreprocessConfig struct {
	Size        int
	BorderRatio float64
}

// ImagePreprocessResult contains BCHW float32 tensors in [-1, 1]. Image has
// shape [1,3,size,size], Mask has shape [1,1,size,size].
type ImagePreprocessResult struct {
	Image []float32
	Mask  []float32
	Size  int
}

func DefaultImagePreprocessConfig() ImagePreprocessConfig {
	return ImagePreprocessConfig{Size: 512, BorderRatio: 0.15}
}

// PreprocessImageV2 mirrors the upstream Hunyuan3D ImageProcessorV2 structure:
// alpha-mask object recentering, white-background compositing, final square
// resize, and BCHW normalization to [-1, 1].
func PreprocessImageV2(src image.Image, cfg ImagePreprocessConfig) (ImagePreprocessResult, error) {
	if src == nil {
		return ImagePreprocessResult{}, fmt.Errorf("hunyuan3d image preprocess: nil image")
	}
	if cfg.Size <= 0 {
		return ImagePreprocessResult{}, fmt.Errorf("hunyuan3d image preprocess: invalid size %d", cfg.Size)
	}
	if cfg.BorderRatio < 0 || cfg.BorderRatio >= 1 {
		return ImagePreprocessResult{}, fmt.Errorf("hunyuan3d image preprocess: invalid border ratio %g", cfg.BorderRatio)
	}
	rgba := toNRGBA(src)
	recentered, mask, err := recenterObject(rgba, cfg.BorderRatio)
	if err != nil {
		return ImagePreprocessResult{}, err
	}
	resized := resizeBilinearNRGBA(recentered, cfg.Size, cfg.Size)
	resizedMask := resizeNearestGray(mask, cfg.Size, cfg.Size)
	return bchwFromImageAndMask(resized, resizedMask), nil
}

func toNRGBA(src image.Image) *image.NRGBA {
	b := src.Bounds()
	out := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			out.Set(x, y, src.At(b.Min.X+x, b.Min.Y+y))
		}
	}
	return out
}

func recenterObject(src *image.NRGBA, borderRatio float64) (*image.NRGBA, *image.Gray, error) {
	w, h := src.Bounds().Dx(), src.Bounds().Dy()
	size := w
	if h > size {
		size = h
	}
	xMin, yMin, xMax, yMax := w, h, -1, -1
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			if src.NRGBAAt(x, y).A != 0 {
				if x < xMin {
					xMin = x
				}
				if x > xMax {
					xMax = x
				}
				if y < yMin {
					yMin = y
				}
				if y > yMax {
					yMax = y
				}
			}
		}
	}
	// Match the Python fixture's exclusive max slice convention.
	objW, objH := xMax-xMin, yMax-yMin
	if objW <= 0 || objH <= 0 {
		return nil, nil, fmt.Errorf("hunyuan3d image preprocess: input image is empty")
	}
	desired := int(float64(size) * (1 - borderRatio))
	scale := float64(desired) / float64(max(objW, objH))
	w2, h2 := int(float64(objW)*scale), int(float64(objH)*scale)
	if w2 <= 0 || h2 <= 0 {
		return nil, nil, fmt.Errorf("hunyuan3d image preprocess: object resized to empty shape")
	}
	crop := image.NewNRGBA(image.Rect(0, 0, objW, objH))
	for y := 0; y < objH; y++ {
		for x := 0; x < objW; x++ {
			crop.SetNRGBA(x, y, src.NRGBAAt(xMin+x, yMin+y))
		}
	}
	obj := resizeBilinearNRGBA(crop, w2, h2)
	canvas := image.NewNRGBA(image.Rect(0, 0, size, size))
	mask := image.NewGray(image.Rect(0, 0, size, size))
	dx, dy := (size-w2)/2, (size-h2)/2
	for y := 0; y < h2; y++ {
		for x := 0; x < w2; x++ {
			p := obj.NRGBAAt(x, y)
			a := float64(p.A) / 255
			canvas.SetNRGBA(dx+x, dy+y, color.NRGBA{
				R: uint8(clamp255(float64(p.R)*a + 255*(1-a))),
				G: uint8(clamp255(float64(p.G)*a + 255*(1-a))),
				B: uint8(clamp255(float64(p.B)*a + 255*(1-a))),
				A: 255,
			})
			mask.SetGray(dx+x, dy+y, color.Gray{Y: p.A})
		}
	}
	return canvas, mask, nil
}

func resizeNearestGray(src *image.Gray, w, h int) *image.Gray {
	out := image.NewGray(image.Rect(0, 0, w, h))
	sw, sh := src.Bounds().Dx(), src.Bounds().Dy()
	for y := 0; y < h; y++ {
		sy := min(sh-1, int(float64(y)*float64(sh)/float64(h)))
		for x := 0; x < w; x++ {
			sx := min(sw-1, int(float64(x)*float64(sw)/float64(w)))
			out.SetGray(x, y, src.GrayAt(sx, sy))
		}
	}
	return out
}

func resizeBilinearNRGBA(src *image.NRGBA, w, h int) *image.NRGBA {
	out := image.NewNRGBA(image.Rect(0, 0, w, h))
	sw, sh := src.Bounds().Dx(), src.Bounds().Dy()
	if sw == 1 || sh == 1 {
		return out
	}
	for y := 0; y < h; y++ {
		fy := (float64(y)+0.5)*float64(sh)/float64(h) - 0.5
		y0 := int(math.Floor(fy))
		wy := fy - float64(y0)
		if y0 < 0 {
			y0, wy = 0, 0
		}
		if y0 >= sh-1 {
			y0, wy = sh-2, 1
		}
		for x := 0; x < w; x++ {
			fx := (float64(x)+0.5)*float64(sw)/float64(w) - 0.5
			x0 := int(math.Floor(fx))
			wx := fx - float64(x0)
			if x0 < 0 {
				x0, wx = 0, 0
			}
			if x0 >= sw-1 {
				x0, wx = sw-2, 1
			}
			out.SetNRGBA(x, y, lerpNRGBA(src.NRGBAAt(x0, y0), src.NRGBAAt(x0+1, y0), src.NRGBAAt(x0, y0+1), src.NRGBAAt(x0+1, y0+1), wx, wy))
		}
	}
	return out
}

func lerpNRGBA(a, b, c, d color.NRGBA, wx, wy float64) color.NRGBA {
	mix := func(aa, bb, cc, dd uint8) uint8 {
		return uint8(clamp255((float64(aa)*(1-wx)+float64(bb)*wx)*(1-wy) + (float64(cc)*(1-wx)+float64(dd)*wx)*wy))
	}
	return color.NRGBA{R: mix(a.R, b.R, c.R, d.R), G: mix(a.G, b.G, c.G, d.G), B: mix(a.B, b.B, c.B, d.B), A: mix(a.A, b.A, c.A, d.A)}
}

func bchwFromImageAndMask(img *image.NRGBA, mask *image.Gray) ImagePreprocessResult {
	size := img.Bounds().Dx()
	pixels := size * size
	imageTensor := make([]float32, 3*pixels)
	maskTensor := make([]float32, pixels)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			i := y*size + x
			p := img.NRGBAAt(x, y)
			imageTensor[i] = normU8(p.R)
			imageTensor[pixels+i] = normU8(p.G)
			imageTensor[2*pixels+i] = normU8(p.B)
			maskTensor[i] = normU8(mask.GrayAt(x, y).Y)
		}
	}
	return ImagePreprocessResult{Image: imageTensor, Mask: maskTensor, Size: size}
}

func normU8(v uint8) float32 { return float32(v)/255*2 - 1 }
func clamp255(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return math.Round(v)
}
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
