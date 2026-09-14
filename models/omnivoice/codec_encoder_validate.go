package omnivoice

import (
	"fmt"
	"math"

	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
)

func validateCodecEncoderWeights(w *loader.CodecWeights) error {
	expect := func(name string, shape ...int) error {
		got := w.Shapes[name]
		if len(got) != len(shape) {
			return fmt.Errorf("omnivoice: missing/malformed codec encoder %s", name)
		}
		size := 1
		for i, n := range shape {
			if got[i] != n || n <= 0 || size > int(^uint(0)>>1)/n {
				return fmt.Errorf("omnivoice: codec encoder shape %s", name)
			}
			size *= n
		}
		values := w.Tensors[name]
		if len(values) != size {
			return fmt.Errorf("omnivoice: codec encoder data length %s", name)
		}
		for _, v := range values {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("omnivoice: nonfinite codec encoder weight %s", name)
			}
		}
		return nil
	}
	conv := func(name string, out, in, kernel int) error {
		if err := expect(name+".weight", out, in, kernel); err != nil {
			return err
		}
		return expect(name+".bias", out)
	}
	if w.SampleRate != 24000 || w.CodebookSize != 1024 || w.Quantizers != 8 || len(w.Rates) != 5 {
		return fmt.Errorf("omnivoice: unsupported codec encoder variant")
	}
	for i, rate := range []int{8, 5, 4, 2, 3} {
		if w.Rates[i] != rate {
			return fmt.Errorf("omnivoice: unsupported encoder rates")
		}
	}
	if err := expect("fc.weight", 1024, 1024); err != nil {
		return err
	}
	if err := expect("fc.bias", 1024); err != nil {
		return err
	}
	if err := conv("acoustic_encoder.conv1", 64, 1, 7); err != nil {
		return err
	}
	in := 64
	for i, rate := range w.Rates {
		p := fmt.Sprintf("acoustic_encoder.block.%d.", i)
		if err := expect(p+"snake1.alpha", 1, in, 1); err != nil {
			return err
		}
		for j := 1; j <= 3; j++ {
			rp := fmt.Sprintf("%sres_unit%d.", p, j)
			for _, n := range []string{"snake1.alpha", "snake2.alpha"} {
				if err := expect(rp+n, 1, in, 1); err != nil {
					return err
				}
			}
			if err := conv(rp+"conv1", in, in, 7); err != nil {
				return err
			}
			if err := conv(rp+"conv2", in, in, 1); err != nil {
				return err
			}
		}
		out := in * 2
		if err := conv(p+"conv1", out, in, 2*rate); err != nil {
			return err
		}
		in = out
	}
	if err := expect("acoustic_encoder.snake1.alpha", 1, 2048, 1); err != nil {
		return err
	}
	if err := conv("acoustic_encoder.conv2", 256, 2048, 3); err != nil {
		return err
	}
	if err := expect("encoder_semantic.conv.weight", 768, 768, 3); err != nil {
		return err
	}
	for i := 0; i < 2; i++ {
		p := fmt.Sprintf("encoder_semantic.conv_blocks.%d.", i)
		for j := 0; j < 2; j++ {
			rp := fmt.Sprintf("%sres_units.%d.", p, j)
			if err := expect(rp+"conv1.weight", 768, 768, 3); err != nil {
				return err
			}
			if err := expect(rp+"conv2.weight", 768, 768, 1); err != nil {
				return err
			}
		}
		if err := expect(p+"conv.weight", 768, 768, 3); err != nil {
			return err
		}
		if err := expect(p+"conv.bias", 768); err != nil {
			return err
		}
	}
	for i := 0; i < 8; i++ {
		p := fmt.Sprintf("quantizer.quantizers.%d.", i)
		for _, spec := range []struct {
			name  string
			shape []int
		}{{p + "codebook.embed", []int{1024, 64}}, {p + "project_in.weight", []int{64, 1024}}, {p + "project_in.bias", []int{64}}, {p + "project_out.weight", []int{1024, 64}}, {p + "project_out.bias", []int{1024}}} {
			if err := expect(spec.name, spec.shape...); err != nil {
				return err
			}
		}
	}
	return nil
}
