package diffusiongemma

import "fmt"

func LoadVisionLayerF32(weights *VisionWeights, layer int) (VisionLayerF32, error) {
	if weights == nil {
		return VisionLayerF32{}, fmt.Errorf("DiffusionGemma vision layer load missing weights")
	}
	fp := weights.ForwardPlan()
	if !fp.Ready {
		return VisionLayerF32{}, fmt.Errorf("DiffusionGemma vision forward bindings incomplete: %v", fp.Missing)
	}
	if layer < 0 || layer >= len(fp.Layers) {
		return VisionLayerF32{}, fmt.Errorf("DiffusionGemma vision layer %d outside [0,%d)", layer, len(fp.Layers))
	}
	lb := fp.Layers[layer]
	inputNorm, err := loadVisionFloatVector(weights, lb.InputLayerNorm)
	if err != nil {
		return VisionLayerF32{}, err
	}
	postAttnNorm, err := loadVisionFloatVector(weights, lb.PostAttentionLayerNorm)
	if err != nil {
		return VisionLayerF32{}, err
	}
	preFFNNorm, err := loadVisionFloatVector(weights, lb.PreFFNLayerNorm)
	if err != nil {
		return VisionLayerF32{}, err
	}
	postFFNNorm, err := loadVisionFloatVector(weights, lb.PostFFNLayerNorm)
	if err != nil {
		return VisionLayerF32{}, err
	}
	q, qRows, qCols, err := loadVisionFloatMatrix(weights, lb.QProj)
	if err != nil {
		return VisionLayerF32{}, err
	}
	k, kRows, kCols, err := loadVisionFloatMatrix(weights, lb.KProj)
	if err != nil {
		return VisionLayerF32{}, err
	}
	v, vRows, vCols, err := loadVisionFloatMatrix(weights, lb.VProj)
	if err != nil {
		return VisionLayerF32{}, err
	}
	o, oRows, oCols, err := loadVisionFloatMatrix(weights, lb.OProj)
	if err != nil {
		return VisionLayerF32{}, err
	}
	qNorm, err := loadVisionFloatVector(weights, lb.QNorm)
	if err != nil {
		return VisionLayerF32{}, err
	}
	kNorm, err := loadVisionFloatVector(weights, lb.KNorm)
	if err != nil {
		return VisionLayerF32{}, err
	}
	gate, gateRows, gateCols, err := loadVisionFloatMatrix(weights, lb.MLPGateProj)
	if err != nil {
		return VisionLayerF32{}, err
	}
	up, upRows, upCols, err := loadVisionFloatMatrix(weights, lb.MLPUpProj)
	if err != nil {
		return VisionLayerF32{}, err
	}
	down, downRows, downCols, err := loadVisionFloatMatrix(weights, lb.MLPDownProj)
	if err != nil {
		return VisionLayerF32{}, err
	}
	hidden := len(inputNorm)
	if hidden <= 0 || len(postAttnNorm) != hidden || len(preFFNNorm) != hidden || len(postFFNNorm) != hidden {
		return VisionLayerF32{}, fmt.Errorf("DiffusionGemma vision layer %d norm shape mismatch hidden=%d", layer, hidden)
	}
	if qRows != hidden || qCols != hidden || kRows != hidden || kCols != hidden || vRows != hidden || vCols != hidden || oRows != hidden || oCols != hidden {
		return VisionLayerF32{}, fmt.Errorf("DiffusionGemma vision layer %d attention matrix shape mismatch", layer)
	}
	if len(qNorm) <= 0 || len(kNorm) != len(qNorm) || hidden%len(qNorm) != 0 {
		return VisionLayerF32{}, fmt.Errorf("DiffusionGemma vision layer %d q/k norm shape mismatch q=%d k=%d hidden=%d", layer, len(qNorm), len(kNorm), hidden)
	}
	if gateRows <= 0 || gateCols != hidden || upRows != gateRows || upCols != hidden || downRows != hidden || downCols != gateRows {
		return VisionLayerF32{}, fmt.Errorf("DiffusionGemma vision layer %d MLP matrix shape mismatch", layer)
	}
	return VisionLayerF32{
		InputLayerNorm:         inputNorm,
		PostAttentionLayerNorm: postAttnNorm,
		PreFFNLayerNorm:        preFFNNorm,
		PostFFNLayerNorm:       postFFNNorm,
		QProj:                  q,
		KProj:                  k,
		VProj:                  v,
		OProj:                  o,
		QNorm:                  qNorm,
		KNorm:                  kNorm,
		MLPGateProj:            gate,
		MLPUpProj:              up,
		MLPDownProj:            down,
		MLPIntermediate:        gateRows,
	}, nil
}

func loadVisionFloatVector(weights *VisionWeights, binding *TensorBinding) ([]float32, error) {
	if binding == nil {
		return nil, fmt.Errorf("DiffusionGemma missing vision vector binding")
	}
	t, err := weights.CachedFloatTensor(binding.Name)
	if err != nil {
		return nil, err
	}
	if len(t.Shape) != 1 || len(t.Data) != t.Shape[0] {
		return nil, fmt.Errorf("DiffusionGemma vision tensor %q shape %v is not rank-1", binding.Name, t.Shape)
	}
	return t.Data, nil
}

func loadVisionFloatMatrix(weights *VisionWeights, binding *TensorBinding) ([]float32, int, int, error) {
	if binding == nil {
		return nil, 0, 0, fmt.Errorf("DiffusionGemma missing vision matrix binding")
	}
	t, err := weights.CachedFloatTensor(binding.Name)
	if err != nil {
		return nil, 0, 0, err
	}
	if len(t.Shape) != 2 || t.Shape[0] <= 0 || t.Shape[1] <= 0 || len(t.Data) != t.Shape[0]*t.Shape[1] {
		return nil, 0, 0, fmt.Errorf("DiffusionGemma vision tensor %q shape %v is not rank-2", binding.Name, t.Shape)
	}
	return t.Data, t.Shape[0], t.Shape[1], nil
}
