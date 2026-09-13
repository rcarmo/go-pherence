package omnivoice

import "fmt"

// MatrixRowsInto reads rows from an on-disk float matrix into caller-owned
// storage without materializing the full embedding/head tensor.
func (w *Weights) MatrixRowsInto(dst []float32, name string, start, rows int) error {
	raw, dtype, shape, err := w.file.GetRaw(name)
	if err != nil {
		return err
	}
	if len(shape) != 2 || start < 0 || rows <= 0 || start > shape[0]-rows {
		return fmt.Errorf("omnivoice: matrix row range invalid: %s", name)
	}
	width := shape[1]
	if width <= 0 || rows > int(^uint(0)>>1)/width || len(dst) != rows*width {
		return fmt.Errorf("omnivoice: matrix row output size mismatch")
	}
	size := 2
	if dtype == "F32" {
		size = 4
	}
	// safetensors has already validated the complete payload shape and byte range.
	begin, end := start*width*size, (start+rows)*width*size
	if begin < 0 || end < begin || end > len(raw) {
		return fmt.Errorf("omnivoice: matrix byte range invalid")
	}
	return convertInto(dst, raw[begin:end], dtype)
}

// CheckCodebookOffsets verifies the trained additive-embedding convention.
// These I64 values are small nonnegative multiples of AudioVocabSize.
func (w *Weights) CheckCodebookOffsets() error {
	raw, dtype, shape, err := w.file.GetRaw("codebook_layer_offsets")
	if err != nil {
		return err
	}
	if dtype != "I64" || len(shape) != 1 || shape[0] != w.Config.NumAudioCodebook {
		return fmt.Errorf("omnivoice: invalid codebook offsets")
	}
	for i := 0; i < shape[0]; i++ {
		var value uint64
		for j := 0; j < 8; j++ {
			value |= uint64(raw[i*8+j]) << uint(8*j)
		}
		if value != uint64(i*w.Config.AudioVocabSize) {
			return fmt.Errorf("omnivoice: unsupported codebook offset at %d", i)
		}
	}
	return nil
}
