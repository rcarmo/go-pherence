package omnivoice

import (
	"fmt"
	"math"

	loader "github.com/rcarmo/go-pherence/loader/omnivoice"
)

func validateHubertWeights(w *loader.HubertWeights) error {
	expect := func(name string, shape ...int) error {
		got := w.Shapes[name]
		if len(got) != len(shape) {
			return fmt.Errorf("omnivoice: missing/malformed hubert %s", name)
		}
		size := 1
		for i, n := range shape {
			if got[i] != n || n <= 0 || size > int(^uint(0)>>1)/n {
				return fmt.Errorf("omnivoice: hubert shape %s", name)
			}
			size *= n
		}
		values := w.Tensors[name]
		if len(values) != size {
			return fmt.Errorf("omnivoice: hubert data length %s", name)
		}
		for _, v := range values {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("omnivoice: nonfinite hubert weight %s", name)
			}
		}
		return nil
	}
	conv := func(name string, out, in, kernel int) error {
		return expect(name+".weight", out, in, kernel)
	}
	linear := func(name string, out, in int) error {
		if err := expect(name+".weight", out, in); err != nil {
			return err
		}
		return expect(name+".bias", out)
	}
	layerNorm := func(name string, width int) error {
		if err := expect(name+".weight", width); err != nil {
			return err
		}
		return expect(name+".bias", width)
	}
	if w.SampleRate != 16000 {
		return fmt.Errorf("omnivoice: unsupported hubert sample rate")
	}
	if err := conv("semantic_model.feature_extractor.conv_layers.0.conv", 512, 1, 10); err != nil {
		return err
	}
	if err := layerNorm("semantic_model.feature_extractor.conv_layers.0.layer_norm", 512); err != nil {
		return err
	}
	for i, kernel := range []int{3, 3, 3, 3, 2, 2} {
		if err := conv(fmt.Sprintf("semantic_model.feature_extractor.conv_layers.%d.conv", i+1), 512, 512, kernel); err != nil {
			return err
		}
	}
	if err := layerNorm("semantic_model.feature_projection.layer_norm", 512); err != nil {
		return err
	}
	if err := linear("semantic_model.feature_projection.projection", 768, 512); err != nil {
		return err
	}
	if err := expect("semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original0", 1, 1, 128); err != nil {
		return err
	}
	if err := expect("semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1", 768, 48, 128); err != nil {
		return err
	}
	if err := expect("semantic_model.encoder.pos_conv_embed.conv.bias", 768); err != nil {
		return err
	}
	if err := layerNorm("semantic_model.encoder.layer_norm", 768); err != nil {
		return err
	}
	for i := 0; i < 12; i++ {
		p := fmt.Sprintf("semantic_model.encoder.layers.%d.", i)
		for _, name := range []string{"q_proj", "k_proj", "v_proj", "out_proj"} {
			if err := linear(p+"attention."+name, 768, 768); err != nil {
				return err
			}
		}
		if err := linear(p+"feed_forward.intermediate_dense", 3072, 768); err != nil {
			return err
		}
		if err := linear(p+"feed_forward.output_dense", 768, 3072); err != nil {
			return err
		}
		if err := layerNorm(p+"layer_norm", 768); err != nil {
			return err
		}
		if err := layerNorm(p+"final_layer_norm", 768); err != nil {
			return err
		}
	}
	return nil
}
