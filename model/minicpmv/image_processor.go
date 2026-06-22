package minicpmv

import (
	"fmt"
	"image"
	"image/color"
	"math"
)

// ImagePreprocessConfig captures the executable image-processing slice needed
// before MiniCPM-V vision-tower execution. It intentionally mirrors common
// Hugging Face CLIP/SigLIP processor fields while staying independent of Python.
type ImagePreprocessConfig struct {
	Size          int
	PatchSize     int
	DoConvertRGB  bool
	DoResize      bool
	DoRescale     bool
	DoNormalize   bool
	RescaleFactor float32
	ImageMean     [3]float32
	ImageStd      [3]float32
}

type ImagePreprocessResult struct {
	PixelValues []float32 `json:"-"`
	Shape       [4]int    `json:"shape"` // [1, 3, height, width]
	Width       int       `json:"width"`
	Height      int       `json:"height"`
	PatchGrid   [2]int    `json:"patch_grid"` // [height/patch, width/patch]
	PatchCount  int       `json:"patch_count"`
}

type ImageTokenLayout struct {
	ImageSize      int `json:"image_size"`
	PatchSize      int `json:"patch_size"`
	PatchGrid      int `json:"patch_grid"`
	VisionTokens   int `json:"vision_tokens"`
	ResamplerQuery int `json:"resampler_query"`
}

func DefaultImagePreprocessConfig(size, patchSize int) ImagePreprocessConfig {
	return ImagePreprocessConfig{
		Size:          size,
		PatchSize:     patchSize,
		DoConvertRGB:  true,
		DoResize:      size > 0,
		DoRescale:     true,
		DoNormalize:   true,
		RescaleFactor: 1.0 / 255.0,
		ImageMean:     [3]float32{0.5, 0.5, 0.5},
		ImageStd:      [3]float32{0.5, 0.5, 0.5},
	}
}

func NewImageTokenLayout(imageSize, patchSize, numQuery int) (ImageTokenLayout, error) {
	if imageSize <= 0 || patchSize <= 0 {
		return ImageTokenLayout{}, fmt.Errorf("MiniCPM-V invalid image layout image_size=%d patch_size=%d", imageSize, patchSize)
	}
	if imageSize%patchSize != 0 {
		return ImageTokenLayout{}, fmt.Errorf("MiniCPM-V image_size=%d is not divisible by patch_size=%d", imageSize, patchSize)
	}
	if numQuery <= 0 {
		return ImageTokenLayout{}, fmt.Errorf("MiniCPM-V invalid resampler query count %d", numQuery)
	}
	grid := imageSize / patchSize
	return ImageTokenLayout{ImageSize: imageSize, PatchSize: patchSize, PatchGrid: grid, VisionTokens: grid * grid, ResamplerQuery: numQuery}, nil
}

// PreprocessImage returns BCHW float32 pixel values for the MiniCPM-V vision
// tower. Full vision-tower tensor execution is separate; this function locks the
// processor contract and patch grid before weights are wired in.
func PreprocessImage(src image.Image, cfg ImagePreprocessConfig) (ImagePreprocessResult, error) {
	if src == nil {
		return ImagePreprocessResult{}, fmt.Errorf("MiniCPM-V image preprocess: nil image")
	}
	b := src.Bounds()
	if b.Empty() {
		return ImagePreprocessResult{}, fmt.Errorf("MiniCPM-V image preprocess: empty image")
	}
	if cfg.DoResize && cfg.Size <= 0 {
		return ImagePreprocessResult{}, fmt.Errorf("MiniCPM-V image preprocess: invalid resize size %d", cfg.Size)
	}
	if cfg.PatchSize < 0 {
		return ImagePreprocessResult{}, fmt.Errorf("MiniCPM-V image preprocess: invalid patch size %d", cfg.PatchSize)
	}
	for i, std := range cfg.ImageStd {
		if cfg.DoNormalize && std == 0 {
			return ImagePreprocessResult{}, fmt.Errorf("MiniCPM-V image preprocess: image_std[%d] is zero", i)
		}
	}
	img := toNRGBA(src)
	if cfg.DoResize {
		img = resizeBilinearNRGBA(img, cfg.Size, cfg.Size)
	}
	return bchw(img, cfg)
}

func toNRGBA(src image.Image) *image.NRGBA {
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

func resizeBilinearNRGBA(src *image.NRGBA, w, h int) *image.NRGBA {
	out := image.NewNRGBA(image.Rect(0, 0, w, h))
	sw, sh := src.Bounds().Dx(), src.Bounds().Dy()
	if sw == w && sh == h {
		copy(out.Pix, src.Pix)
		return out
	}
	for y := 0; y < h; y++ {
		fy := 0.0
		if h > 1 {
			fy = float64(y) * float64(sh-1) / float64(h-1)
		}
		y0 := int(math.Floor(fy))
		y1 := min(y0+1, sh-1)
		wy := fy - float64(y0)
		for x := 0; x < w; x++ {
			fx := 0.0
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

func bchw(img *image.NRGBA, cfg ImagePreprocessConfig) (ImagePreprocessResult, error) {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	if cfg.PatchSize > 0 && (w%cfg.PatchSize != 0 || h%cfg.PatchSize != 0) {
		return ImagePreprocessResult{}, fmt.Errorf("MiniCPM-V image preprocess: size %dx%d is not divisible by patch_size=%d", w, h, cfg.PatchSize)
	}
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
	gridH, gridW, count := 0, 0, 0
	if cfg.PatchSize > 0 {
		gridH, gridW = h/cfg.PatchSize, w/cfg.PatchSize
		count = gridH * gridW
	}
	return ImagePreprocessResult{PixelValues: out, Shape: [4]int{1, 3, h, w}, Width: w, Height: h, PatchGrid: [2]int{gridH, gridW}, PatchCount: count}, nil
}
