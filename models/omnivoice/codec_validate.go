package omnivoice

import (
	"fmt"
	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
	"math"
)

// Validate the supported fixed Higgs/DAC graph before any numeric indexing.
func validateCodecWeights(w *loader.CodecWeights) error {
	expect := func(name string, shape ...int) error {
		got := w.Shapes[name]
		if len(got) != len(shape) {
			return fmt.Errorf("omnivoice: missing/malformed codec %s", name)
		}
		size := 1
		for i, n := range shape {
			if got[i] != n || n <= 0 || size > int(^uint(0)>>1)/n {
				return fmt.Errorf("omnivoice: codec shape %s", name)
			}
			size *= n
		}
		values := w.Tensors[name]
		if len(values) != size {
			return fmt.Errorf("omnivoice: codec data length %s", name)
		}
		for _, v := range values {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("omnivoice: nonfinite codec weight %s", name)
			}
		}
		return nil
	}
	if w.SampleRate != 24000 || w.CodebookSize != 1024 || w.Quantizers != 8 || len(w.Rates) != 5 {
		return fmt.Errorf("omnivoice: unsupported codec variant")
	}
	for i, rate := range []int{8, 5, 4, 2, 3} {
		if w.Rates[i] != rate {
			return fmt.Errorf("omnivoice: unsupported decoder rates")
		}
	}
	for i := 0; i < 8; i++ {
		p := fmt.Sprintf("quantizer.quantizers.%d.", i)
		for _, spec := range []struct {
			name  string
			shape []int
		}{{p + "codebook.embed", []int{1024, 64}}, {p + "project_out.weight", []int{1024, 64}}, {p + "project_out.bias", []int{1024}}} {
			if err := expect(spec.name, spec.shape...); err != nil {
				return err
			}
		}
	}
	conv := func(name string, out, in, kernel int) error {
		if err := expect(name+".weight", out, in, kernel); err != nil {
			return err
		}
		return expect(name+".bias", out)
	}
	if err := expect("fc2.weight", 256, 1024); err != nil {
		return err
	}
	if err := expect("fc2.bias", 256); err != nil {
		return err
	}
	if err := conv("acoustic_decoder.conv1", 1024, 256, 7); err != nil {
		return err
	}
	in := 1024
	for i, rate := range w.Rates {
		out := in / 2
		p := fmt.Sprintf("acoustic_decoder.block.%d.", i)
		if err := expect(p+"snake1.alpha", 1, in, 1); err != nil {
			return err
		}
		if err := expect(p+"conv_t1.weight", in, out, 2*rate); err != nil {
			return err
		}
		if err := expect(p+"conv_t1.bias", out); err != nil {
			return err
		}
		for j := 1; j <= 3; j++ {
			rp := fmt.Sprintf("%sres_unit%d.", p, j)
			for _, n := range []string{"snake1.alpha", "snake2.alpha"} {
				if err := expect(rp+n, 1, out, 1); err != nil {
					return err
				}
			}
			if err := conv(rp+"conv1", out, out, 7); err != nil {
				return err
			}
			if err := conv(rp+"conv2", out, out, 1); err != nil {
				return err
			}
		}
		in = out
	}
	if err := expect("acoustic_decoder.snake1.alpha", 1, 32, 1); err != nil {
		return err
	}
	return conv("acoustic_decoder.conv2", 1, 32, 7)
}
