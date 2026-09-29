package nemotronasr

import (
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// LoadIndexedEncoderBlock reuses the bounded encoder-0 mathematics for one
// released block. Weight loading alone does not qualify a layer's outputs.
func LoadIndexedEncoderBlock(file *safetensors.File, layer int) (*Encoder0Block, error) {
	if file == nil || layer < 1 || layer >= 24 {
		return nil, fmt.Errorf("invalid Nemotron ASR encoder layer or checkpoint")
	}
	prefix := fmt.Sprintf("encoder.layers.%d.", layer)
	load := func(name string, dims ...int) ([]float32, error) {
		values, shape, err := file.GetFloat32(prefix + name)
		if err != nil {
			return nil, err
		}
		if len(shape) != len(dims) {
			return nil, fmt.Errorf("invalid encoder layer %d %s rank %v", layer, name, shape)
		}
		count := 1
		for i, dim := range dims {
			if shape[i] != dim {
				return nil, fmt.Errorf("invalid encoder layer %d %s shape %v", layer, name, shape)
			}
			count *= dim
		}
		if len(values) != count {
			return nil, fmt.Errorf("invalid encoder layer %d %s length", layer, name)
		}
		for _, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("non-finite encoder layer %d %s", layer, name)
			}
		}
		return values, nil
	}
	block := &Encoder0Block{
		FF1:       &Encoder0FeedForward1{},
		Attention: &Encoder0Attention{qkv: &Encoder0QKV{}},
		Conv:      &Encoder0Convolution{},
		FF2:       &Encoder0FeedForward2{},
	}
	var err error
	ff1 := block.FF1
	if ff1.gamma, err = load("norm_feed_forward1.weight", encoderWidth); err != nil {
		return nil, err
	}
	if ff1.beta, err = load("norm_feed_forward1.bias", encoderWidth); err != nil {
		return nil, err
	}
	if ff1.first, err = load("feed_forward1.linear1.weight", encoderFFWidth, encoderWidth); err != nil {
		return nil, err
	}
	if ff1.last, err = load("feed_forward1.linear2.weight", encoderWidth, encoderFFWidth); err != nil {
		return nil, err
	}
	attn := block.Attention
	qkv := attn.qkv
	if qkv.gamma, err = load("norm_self_att.weight", encoderWidth); err != nil {
		return nil, err
	}
	if qkv.beta, err = load("norm_self_att.bias", encoderWidth); err != nil {
		return nil, err
	}
	if qkv.q, err = load("self_attn.q_proj.weight", encoderWidth, encoderWidth); err != nil {
		return nil, err
	}
	if qkv.k, err = load("self_attn.k_proj.weight", encoderWidth, encoderWidth); err != nil {
		return nil, err
	}
	if qkv.v, err = load("self_attn.v_proj.weight", encoderWidth, encoderWidth); err != nil {
		return nil, err
	}
	if attn.relativeWeight, err = load("self_attn.relative_k_proj.weight", encoderWidth, encoderWidth); err != nil {
		return nil, err
	}
	if attn.biasU, err = load("self_attn.bias_u", asrAttentionHeads, asrAttentionHeadWidth); err != nil {
		return nil, err
	}
	if attn.biasV, err = load("self_attn.bias_v", asrAttentionHeads, asrAttentionHeadWidth); err != nil {
		return nil, err
	}
	if attn.outputWeight, err = load("self_attn.o_proj.weight", encoderWidth, encoderWidth); err != nil {
		return nil, err
	}
	conv := block.Conv
	if conv.preGamma, err = load("norm_conv.weight", encoderWidth); err != nil {
		return nil, err
	}
	if conv.preBeta, err = load("norm_conv.bias", encoderWidth); err != nil {
		return nil, err
	}
	if conv.point1, err = load("conv.pointwise_conv1.weight", 2*encoderWidth, encoderWidth, 1); err != nil {
		return nil, err
	}
	if conv.depth, err = load("conv.depthwise_conv.weight", encoderWidth, 1, encoderConvKernel); err != nil {
		return nil, err
	}
	if conv.depthGamma, err = load("conv.norm.weight", encoderWidth); err != nil {
		return nil, err
	}
	if conv.depthBeta, err = load("conv.norm.bias", encoderWidth); err != nil {
		return nil, err
	}
	if conv.point2, err = load("conv.pointwise_conv2.weight", encoderWidth, encoderWidth, 1); err != nil {
		return nil, err
	}
	ff2 := block.FF2
	if ff2.gamma, err = load("norm_feed_forward2.weight", encoderWidth); err != nil {
		return nil, err
	}
	if ff2.beta, err = load("norm_feed_forward2.bias", encoderWidth); err != nil {
		return nil, err
	}
	if ff2.first, err = load("feed_forward2.linear1.weight", encoderFFWidth, encoderWidth); err != nil {
		return nil, err
	}
	if ff2.last, err = load("feed_forward2.linear2.weight", encoderWidth, encoderFFWidth); err != nil {
		return nil, err
	}
	if ff2.outGamma, err = load("norm_out.weight", encoderWidth); err != nil {
		return nil, err
	}
	if ff2.outBeta, err = load("norm_out.bias", encoderWidth); err != nil {
		return nil, err
	}
	return block, nil
}
