package model

import (
	"fmt"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/loader/gguf"
)

func (c GGUFLlamaConfig) IsQwenNextHybridGGUF() bool {
	return c.SSMInnerSize > 0 || c.SSMStateSize > 0 || c.FullAttentionInterval > 0 || c.AttentionKeyLength > 0 || c.AttentionValueLength > 0
}

func (c GGUFLlamaConfig) ValidateQwenNextHybridMetadata() error {
	if !c.IsQwenNextHybridGGUF() {
		return nil
	}
	if c.HiddenSize <= 0 || c.SSMInnerSize <= 0 || c.SSMConvKernel <= 0 || c.SSMGroupCount <= 0 || c.SSMStateSize <= 0 || c.SSMTimeStepRank <= 0 {
		return fmt.Errorf("incomplete QwenNext GGUF SSM metadata hidden=%d inner=%d kernel=%d groups=%d state=%d rank=%d", c.HiddenSize, c.SSMInnerSize, c.SSMConvKernel, c.SSMGroupCount, c.SSMStateSize, c.SSMTimeStepRank)
	}
	if c.AttentionKeyLength <= 0 || c.AttentionValueLength <= 0 || c.FullAttentionInterval <= 0 {
		return fmt.Errorf("incomplete QwenNext GGUF attention metadata key=%d value=%d interval=%d", c.AttentionKeyLength, c.AttentionValueLength, c.FullAttentionInterval)
	}
	return nil
}

func loadGGUFQwenNextHybridTensors(g *gguf.GGUF, layerIdx int, layer *GGUFLlamaLayer) error {
	if g == nil || layer == nil {
		return fmt.Errorf("nil GGUF QwenNext load input")
	}
	p := fmt.Sprintf("blk.%d.", layerIdx)
	loadMatrixOptional := func(name string) (*gguf.QuantMatrix, error) {
		t, ok := g.TensorByName(name)
		if !ok {
			return nil, nil
		}
		return g.MatrixFromTensor(t)
	}
	loadFloatOptional := func(name string) ([]float32, error) {
		t, ok := g.TensorByName(name)
		if !ok {
			return nil, nil
		}
		return g.DequantF32(t)
	}
	var err error
	if layer.FusedQKVM, err = loadMatrixOptional(p + "attn_qkv.weight"); err != nil {
		return err
	}
	if layer.AttnGateM, err = loadMatrixOptional(p + "attn_gate.weight"); err != nil {
		return err
	}
	if layer.SSMOutM, err = loadMatrixOptional(p + "ssm_out.weight"); err != nil {
		return err
	}
	if layer.SSMAlphaM, err = loadMatrixOptional(p + "ssm_alpha.weight"); err != nil {
		return err
	}
	if layer.SSMBetaM, err = loadMatrixOptional(p + "ssm_beta.weight"); err != nil {
		return err
	}
	if layer.SSMConv1D, err = loadFloatOptional(p + "ssm_conv1d.weight"); err != nil {
		return err
	}
	if layer.SSMA, err = loadFloatOptional(p + "ssm_a"); err != nil {
		return err
	}
	if layer.SSMDTBias, err = loadFloatOptional(p + "ssm_dt.bias"); err != nil {
		return err
	}
	if layer.SSMNorm, err = loadFloatOptional(p + "ssm_norm.weight"); err != nil {
		return err
	}
	return nil
}

func (c GGUFLlamaConfig) QwenNextLinearShapes() (loaderconfig.Qwen35LinearAttentionShapes, error) {
	return loaderconfig.Qwen35LinearAttentionShapesFor(c.HiddenSize, c.SSMInnerSize, c.SSMStateSize, c.SSMConvKernel, c.SSMTimeStepRank, c.SSMGroupCount)
}

func (l GGUFLlamaLayer) HasQwenNextHybridTensors() bool {
	return l.FusedQKVM != nil || l.SSMOutM != nil || l.SSMConv1D != nil || l.SSMA != nil
}

type GGUFQwenNextState struct {
	Conv []float32
	SSM  []float32
	Pos  int
}

func (c GGUFLlamaConfig) NewQwenNextState() (GGUFQwenNextState, error) {
	shapes, err := c.QwenNextLinearShapes()
	if err != nil {
		return GGUFQwenNextState{}, err
	}
	ssmLen := c.SSMTimeStepRank * c.SSMInnerSize * c.SSMStateSize
	if ssmLen < 0 {
		return GGUFQwenNextState{}, fmt.Errorf("QwenNext SSM state overflow")
	}
	return GGUFQwenNextState{Conv: make([]float32, shapes.ConvDim*c.SSMConvKernel), SSM: make([]float32, ssmLen)}, nil
}

func (m *GGUFLlama) projectQwenNextFusedQKV(layer *GGUFLlamaLayer, x []float32) (q, k, v []float32, err error) {
	if m == nil || layer == nil || layer.FusedQKVM == nil {
		return nil, nil, nil, fmt.Errorf("missing QwenNext fused qkv tensor")
	}
	shapes, err := m.Config.QwenNextLinearShapes()
	if err != nil {
		return nil, nil, nil, err
	}
	if len(x) != m.Config.HiddenSize {
		return nil, nil, nil, fmt.Errorf("QwenNext fused qkv input len=%d want %d", len(x), m.Config.HiddenSize)
	}
	if layer.FusedQKVM.InDim != m.Config.HiddenSize || layer.FusedQKVM.OutDim != shapes.ConvDim {
		return nil, nil, nil, fmt.Errorf("QwenNext fused qkv dims in/out=%d/%d want %d/%d", layer.FusedQKVM.InDim, layer.FusedQKVM.OutDim, m.Config.HiddenSize, shapes.ConvDim)
	}
	projected := make([]float32, shapes.ConvDim)
	m.gemvMaybe(projected, x, nil, layer.FusedQKVM, m.Config.HiddenSize, shapes.ConvDim)
	return splitGGUFQwenNextFusedQKV(projected, shapes)
}

func (m *GGUFLlama) projectQwenNextGate(layer *GGUFLlamaLayer, x []float32) ([]float32, error) {
	if m == nil || layer == nil || layer.AttnGateM == nil {
		return nil, fmt.Errorf("missing QwenNext attention gate tensor")
	}
	shapes, err := m.Config.QwenNextLinearShapes()
	if err != nil {
		return nil, err
	}
	z := make([]float32, shapes.ValueDim)
	m.gemvMaybe(z, x, nil, layer.AttnGateM, m.Config.HiddenSize, shapes.ValueDim)
	return z, nil
}

func splitGGUFQwenNextFusedQKV(projected []float32, shapes loaderconfig.Qwen35LinearAttentionShapes) (q, k, v []float32, err error) {
	if len(projected) != shapes.ConvDim || shapes.KeyDim <= 0 || shapes.ValueDim <= 0 || shapes.ConvDim != shapes.KeyDim*2+shapes.ValueDim {
		return nil, nil, nil, fmt.Errorf("QwenNext fused qkv len=%d want conv=%d key=%d value=%d", len(projected), shapes.ConvDim, shapes.KeyDim, shapes.ValueDim)
	}
	q = append([]float32(nil), projected[:shapes.KeyDim]...)
	k = append([]float32(nil), projected[shapes.KeyDim:2*shapes.KeyDim]...)
	v = append([]float32(nil), projected[2*shapes.KeyDim:]...)
	return q, k, v, nil
}

func updateGGUFQwenNextConvStateInPlace(state, q, k, v []float32, kernel int) error {
	if kernel <= 0 {
		return fmt.Errorf("invalid QwenNext conv kernel %d", kernel)
	}
	convDim := len(q) + len(k) + len(v)
	if convDim <= 0 || len(state) != convDim*kernel {
		return fmt.Errorf("QwenNext conv state len=%d want %d", len(state), convDim*kernel)
	}
	copy(state, state[convDim:])
	pos := (kernel - 1) * convDim
	copy(state[pos:pos+len(q)], q)
	copy(state[pos+len(q):pos+len(q)+len(k)], k)
	copy(state[pos+len(q)+len(k):pos+convDim], v)
	return nil
}

func applyGGUFQwenNextDepthwiseConv(state, weight []float32, convDim, kernel int) ([]float32, error) {
	if convDim <= 0 || kernel <= 0 || len(state) != convDim*kernel || len(weight) != convDim*kernel {
		return nil, fmt.Errorf("QwenNext conv dims state=%d weight=%d want %d", len(state), len(weight), convDim*kernel)
	}
	out := make([]float32, convDim)
	for c := 0; c < convDim; c++ {
		var sum float32
		for k := 0; k < kernel; k++ {
			sum += state[k*convDim+c] * weight[k*convDim+c]
		}
		out[c] = sum
	}
	return out, nil
}
