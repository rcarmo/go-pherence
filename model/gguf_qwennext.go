package model

import (
	"fmt"

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

func (l GGUFLlamaLayer) HasQwenNextHybridTensors() bool {
	return l.FusedQKVM != nil || l.SSMOutM != nil || l.SSMConv1D != nil || l.SSMA != nil
}
