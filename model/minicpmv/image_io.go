package minicpmv

import (
	"image"
	_ "image/jpeg"
	_ "image/png"
	"os"
)

type ImageFilePreprocessResult struct {
	Path   string                `json:"path"`
	Format string                `json:"format"`
	Result ImagePreprocessResult `json:"result"`
}

func PreprocessImageFile(path string, cfg ImagePreprocessConfig) (ImageFilePreprocessResult, error) {
	f, err := os.Open(path)
	if err != nil {
		return ImageFilePreprocessResult{}, err
	}
	defer f.Close()
	img, format, err := image.Decode(f)
	if err != nil {
		return ImageFilePreprocessResult{}, err
	}
	res, err := PreprocessImage(img, cfg)
	if err != nil {
		return ImageFilePreprocessResult{}, err
	}
	return ImageFilePreprocessResult{Path: path, Format: format, Result: res}, nil
}
