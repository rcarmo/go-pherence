package diffusiongemma

type ResidencyEstimate struct {
	Globals      bool  `json:"globals"`
	Layers       int   `json:"layers"`
	TensorCount  int   `json:"tensor_count"`
	Elements     int64 `json:"elements"`
	Float32Bytes int64 `json:"float32_bytes"`
}

func EstimateResidency(plan *TextTensorPlan, globals bool, layers int) ResidencyEstimate {
	out := ResidencyEstimate{Globals: globals, Layers: layers}
	if plan == nil {
		return out
	}
	add := func(h TensorHandle) {
		// Shape metadata requires opened shards, so fall back to known coarse
		// published-shape estimates where available in future. This function is
		// intentionally conservative until shape metadata is passed in.
		_ = h
		out.TensorCount++
	}
	if globals {
		for _, h := range plan.Globals {
			add(h)
		}
	}
	if layers > len(plan.Layers) {
		layers = len(plan.Layers)
	}
	if layers < 0 {
		layers = 0
	}
	out.Layers = layers
	for i := 0; i < layers; i++ {
		for _, h := range plan.Layers[i].Handles {
			add(h)
		}
	}
	return out
}

func EstimateResidencyFromWeights(weights *TextWeights, globals bool, layers int) ResidencyEstimate {
	out := ResidencyEstimate{Globals: globals, Layers: layers}
	if weights == nil {
		return out
	}
	add := func(b TensorBinding) {
		out.TensorCount++
		n := int64(1)
		for _, dim := range b.Shape {
			if dim <= 0 {
				return
			}
			n *= int64(dim)
		}
		out.Elements += n
		out.Float32Bytes += n * 4
	}
	if globals {
		for _, b := range weights.Globals {
			add(b)
		}
	}
	if layers > len(weights.Layers) {
		layers = len(weights.Layers)
	}
	if layers < 0 {
		layers = 0
	}
	out.Layers = layers
	for i := 0; i < layers; i++ {
		for _, b := range weights.Layers[i].Bindings {
			add(b)
		}
	}
	return out
}
