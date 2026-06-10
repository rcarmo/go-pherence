package diffusiongemma

type ResidencyEstimate struct {
	Globals      bool                     `json:"globals"`
	Layers       int                      `json:"layers"`
	TensorCount  int                      `json:"tensor_count"`
	Elements     int64                    `json:"elements"`
	Float32Bytes int64                    `json:"float32_bytes"`
	GlobalBytes  int64                    `json:"global_bytes"`
	LayerBytes   []LayerResidencyEstimate `json:"layer_bytes,omitempty"`
}

type LayerResidencyEstimate struct {
	Layer        int   `json:"layer"`
	TensorCount  int   `json:"tensor_count"`
	Elements     int64 `json:"elements"`
	Float32Bytes int64 `json:"float32_bytes"`
}

type ResidencyBudget struct {
	BudgetBytes       int64 `json:"budget_bytes"`
	Globals           bool  `json:"globals"`
	GlobalBytes       int64 `json:"global_bytes"`
	ResidentLayers    int   `json:"resident_layers"`
	ResidentBytes     int64 `json:"resident_bytes"`
	RemainingBytes    int64 `json:"remaining_bytes"`
	TotalLayers       int   `json:"total_layers"`
	AllLayersResident bool  `json:"all_layers_resident"`
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
	addBinding := func(b TensorBinding) (int64, int64) {
		n := int64(1)
		for _, dim := range b.Shape {
			if dim <= 0 {
				return 0, 0
			}
			n *= int64(dim)
		}
		return n, n * 4
	}
	add := func(b TensorBinding) {
		out.TensorCount++
		elements, bytes := addBinding(b)
		out.Elements += elements
		out.Float32Bytes += bytes
	}
	if globals {
		for _, b := range weights.Globals {
			elements, bytes := addBinding(b)
			out.TensorCount++
			out.Elements += elements
			out.Float32Bytes += bytes
			out.GlobalBytes += bytes
		}
	}
	if layers > len(weights.Layers) {
		layers = len(weights.Layers)
	}
	if layers < 0 {
		layers = 0
	}
	out.Layers = layers
	out.LayerBytes = make([]LayerResidencyEstimate, 0, layers)
	for i := 0; i < layers; i++ {
		layer := LayerResidencyEstimate{Layer: i}
		for _, b := range weights.Layers[i].Bindings {
			elements, bytes := addBinding(b)
			layer.TensorCount++
			layer.Elements += elements
			layer.Float32Bytes += bytes
			add(b)
		}
		out.LayerBytes = append(out.LayerBytes, layer)
	}
	return out
}

func EstimateResidencyBudgetFromWeights(weights *TextWeights, globals bool, budgetBytes int64) ResidencyBudget {
	out := ResidencyBudget{BudgetBytes: budgetBytes, Globals: globals}
	if weights == nil {
		return out
	}
	out.TotalLayers = len(weights.Layers)
	if globals {
		globalEstimate := EstimateResidencyFromWeights(weights, true, 0)
		out.GlobalBytes = globalEstimate.GlobalBytes
		out.ResidentBytes = out.GlobalBytes
	}
	if budgetBytes > 0 && out.ResidentBytes > budgetBytes {
		out.RemainingBytes = budgetBytes - out.ResidentBytes
		return out
	}
	for i := 0; i < len(weights.Layers); i++ {
		layerEstimate := EstimateResidencyFromWeights(weights, false, i+1)
		var layerBytes int64
		if len(layerEstimate.LayerBytes) > 0 {
			layerBytes = layerEstimate.LayerBytes[len(layerEstimate.LayerBytes)-1].Float32Bytes
		}
		if budgetBytes > 0 && out.ResidentBytes+layerBytes > budgetBytes {
			break
		}
		out.ResidentLayers++
		out.ResidentBytes += layerBytes
	}
	if budgetBytes > 0 {
		out.RemainingBytes = budgetBytes - out.ResidentBytes
	}
	out.AllLayersResident = out.ResidentLayers == len(weights.Layers)
	return out
}
