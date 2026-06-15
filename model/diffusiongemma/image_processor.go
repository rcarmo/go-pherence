package diffusiongemma

import (
	"fmt"
	"image"
	"image/color"
	"math"
)

// Gemma4ImagePreprocessConfig captures the executable image-processor slice used
// before the full DiffusionGemma vision encoder: RGB conversion, optional square
// resize, rescale, normalize, and BCHW float32 layout.
type Gemma4ImagePreprocessConfig struct {
	Size          int
	DoConvertRGB  bool
	DoResize      bool
	DoRescale     bool
	DoNormalize   bool
	RescaleFactor float32
	ImageMean     [3]float32
	ImageStd      [3]float32
}

type Gemma4ImagePreprocessResult struct {
	PixelValues []float32 `json:"-"`
	Shape       [4]int    `json:"shape"` // [1, 3, height, width]
	Width       int       `json:"width"`
	Height      int       `json:"height"`
}

func DefaultGemma4ImagePreprocessConfig(size int) Gemma4ImagePreprocessConfig {
	return Gemma4ImagePreprocessConfig{
		Size:          size,
		DoConvertRGB:  true,
		DoResize:      size > 0,
		DoRescale:     true,
		DoNormalize:   false,
		RescaleFactor: 1.0 / 255.0,
		ImageMean:     [3]float32{0, 0, 0},
		ImageStd:      [3]float32{1, 1, 1},
	}
}

func Gemma4ImagePreprocessConfigFromProcessor(proc *ProcessorMetadata, size int) Gemma4ImagePreprocessConfig {
	cfg := DefaultGemma4ImagePreprocessConfig(size)
	if proc == nil {
		return cfg
	}
	cfg.DoConvertRGB = proc.DoConvertRGB
	cfg.DoResize = proc.DoResize
	cfg.DoRescale = proc.DoRescale
	cfg.DoNormalize = proc.DoNormalize
	if proc.RescaleFactor != 0 {
		cfg.RescaleFactor = proc.RescaleFactor
	}
	if len(proc.ImageMean) >= 3 {
		copy(cfg.ImageMean[:], proc.ImageMean[:3])
	}
	if len(proc.ImageStd) >= 3 {
		copy(cfg.ImageStd[:], proc.ImageStd[:3])
	}
	return cfg
}

// PreprocessGemma4Image returns BCHW pixel values. Patch projection/image-token
// insertion plus guarded streaming tower entrypoints live in vision_embeddings.go;
// full image-sequence vision reference validation remains the outstanding step.
func PreprocessGemma4Image(src image.Image, cfg Gemma4ImagePreprocessConfig) (Gemma4ImagePreprocessResult, error) {
	if src == nil {
		return Gemma4ImagePreprocessResult{}, fmt.Errorf("DiffusionGemma image preprocess: nil image")
	}
	b := src.Bounds()
	if b.Empty() {
		return Gemma4ImagePreprocessResult{}, fmt.Errorf("DiffusionGemma image preprocess: empty image")
	}
	if cfg.DoResize && cfg.Size <= 0 {
		return Gemma4ImagePreprocessResult{}, fmt.Errorf("DiffusionGemma image preprocess: invalid resize size %d", cfg.Size)
	}
	for i, std := range cfg.ImageStd {
		if cfg.DoNormalize && std == 0 {
			return Gemma4ImagePreprocessResult{}, fmt.Errorf("DiffusionGemma image preprocess: image_std[%d] is zero", i)
		}
	}
	img := gemma4ToNRGBA(src)
	if cfg.DoResize {
		img = gemma4ResizeBilinearNRGBA(img, cfg.Size, cfg.Size)
	}
	return gemma4BCHW(img, cfg), nil
}

func gemma4ToNRGBA(src image.Image) *image.NRGBA {
	b := src.Bounds()
	out := image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	for y := 0; y < b.Dy(); y++ {
		for x := 0; x < b.Dx(); x++ {
			c := color.NRGBAModel.Convert(src.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA)
			out.SetNRGBA(x, y, c)
		}
	}
	return out
}

func gemma4ResizeBilinearNRGBA(src *image.NRGBA, w, h int) *image.NRGBA {
	out := image.NewNRGBA(image.Rect(0, 0, w, h))
	sb := src.Bounds()
	sw, sh := sb.Dx(), sb.Dy()
	if sw == w && sh == h {
		copy(out.Pix, src.Pix)
		return out
	}
	for y := 0; y < h; y++ {
		var fy float64
		if h > 1 {
			fy = float64(y) * float64(sh-1) / float64(h-1)
		}
		y0 := int(math.Floor(fy))
		y1 := min(y0+1, sh-1)
		wy := fy - float64(y0)
		for x := 0; x < w; x++ {
			var fx float64
			if w > 1 {
				fx = float64(x) * float64(sw-1) / float64(w-1)
			}
			x0 := int(math.Floor(fx))
			x1 := min(x0+1, sw-1)
			wx := fx - float64(x0)
			for c := 0; c < 4; c++ {
				p00 := float64(src.Pix[src.PixOffset(x0, y0)+c])
				p10 := float64(src.Pix[src.PixOffset(x1, y0)+c])
				p01 := float64(src.Pix[src.PixOffset(x0, y1)+c])
				p11 := float64(src.Pix[src.PixOffset(x1, y1)+c])
				v0 := p00*(1-wx) + p10*wx
				v1 := p01*(1-wx) + p11*wx
				out.Pix[out.PixOffset(x, y)+c] = uint8(math.Round(v0*(1-wy) + v1*wy))
			}
		}
	}
	return out
}

func gemma4BCHW(img *image.NRGBA, cfg Gemma4ImagePreprocessConfig) Gemma4ImagePreprocessResult {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	pixels := w * h
	out := make([]float32, 3*pixels)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			off := img.PixOffset(x, y)
			vals := [3]float32{float32(img.Pix[off]), float32(img.Pix[off+1]), float32(img.Pix[off+2])}
			i := y*w + x
			for c := 0; c < 3; c++ {
				v := vals[c]
				if cfg.DoRescale {
					v *= cfg.RescaleFactor
				}
				if cfg.DoNormalize {
					v = (v - cfg.ImageMean[c]) / cfg.ImageStd[c]
				}
				out[c*pixels+i] = v
			}
		}
	}
	return Gemma4ImagePreprocessResult{PixelValues: out, Shape: [4]int{1, 3, h, w}, Width: w, Height: h}
}
