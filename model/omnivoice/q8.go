package omnivoice

import "fmt"

// EnableDirectQ8 selects the experimental single-caller direct Q8_0 projection
// kernel. Enable before creating CFG siblings. It cannot combine with resident
// float32/prepacked layers; worker pools still accelerate the float audio head.
// Encoded weights remain borrowed from Weights. The streamed float arena remains
// reserved but quantised projection conversion is skipped. No concurrent setup.
func (b *Backbone) EnableDirectQ8() error {
	if b == nil || b.weights == nil {
		return fmt.Errorf("omnivoice: nil backbone")
	}
	if b.resident != nil {
		return fmt.Errorf("omnivoice: direct Q8 is incompatible with resident layers")
	}
	if b.directQ8 != nil {
		return nil
	}
	layers := make([]map[string][]byte, b.weights.Config.LLMConfig.NumHiddenLayers)
	count := 0
	for i := range layers {
		layers[i] = make(map[string][]byte)
		for _, suffix := range blockPrepackedSuffixes {
			if raw, ok := b.weights.QuantizedProjection(fmt.Sprintf("llm.layers.%d.%s", i, suffix)); ok {
				layers[i][suffix] = raw
				count++
			}
		}
	}
	if count == 0 {
		return fmt.Errorf("omnivoice: no Q8_0 projections in checkpoint")
	}
	b.directQ8 = layers
	return nil
}
