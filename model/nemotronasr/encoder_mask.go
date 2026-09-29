package nemotronasr

import "fmt"

// EncoderMaskRows applies the pinned encoder's offline causal length formula
// to the input mel mask. This is distinct from the streaming subsampler's
// internal floor(valid/2) per-stage masks. For example, 25/26 and 24/32
// mel chunks each project four rows and mark all four visible to attention.
// Returns projected rows and the mask's length threshold. It is deliberately
// not clamped: the pinned mask formula can yield a threshold above projected
// rows; comparison with each row index makes such rows visible. The first
// streaming chunk has an extra initial Conv2D pad row at each stage.
func EncoderMaskRows(frames, valid int, first bool) (rows, visible int, err error) {
	if frames < 1 || frames > 128 || valid < 0 || valid > frames {
		return 0, 0, fmt.Errorf("invalid Nemotron ASR encoder mel mask")
	}
	rows, visible = frames, valid
	for i := 0; i < 3; i++ {
		if first {
			rows = (rows + 1) / 2
		} else {
			rows /= 2
		}
		visible = visible/2 + 1
	}
	return rows, visible, nil
}
