package qwen3tts

import (
	"fmt"
	"math"
	"path/filepath"

	"github.com/rcarmo/go-pherence/loader/weights"
)

// Decoder12HzConfig is the fixed published Qwen3-TTS tokenizer decoder
// topology. SamplesPerFrame is the exact neural upsampling product.
type Decoder12HzConfig struct {
	CodebookDim, LatentDim, HiddenSize, Layers, Heads, HeadDim int
	IntermediateSize, Quantizers, CodebookSize, DecoderDim     int
	PreUpsampleRates, DecoderUpsampleRates                     []int
	RMSNormEps, RoPETheta                                      float32
}

func DefaultDecoder12HzConfig() Decoder12HzConfig {
	return Decoder12HzConfig{CodebookDim: 256, LatentDim: 1024, HiddenSize: 512, Layers: 8, Heads: 16, HeadDim: 64, IntermediateSize: 1024, Quantizers: 16, CodebookSize: 2048, DecoderDim: 1536, PreUpsampleRates: []int{2, 2}, DecoderUpsampleRates: []int{8, 5, 4, 3}, RMSNormEps: 1e-5, RoPETheta: 10000}
}

func tinyDecoder12HzConfig() Decoder12HzConfig {
	return Decoder12HzConfig{CodebookDim: 2, LatentDim: 4, HiddenSize: 4, Layers: 1, Heads: 1, HeadDim: 4, IntermediateSize: 4, Quantizers: 16, CodebookSize: 2048, DecoderDim: 16, PreUpsampleRates: []int{2, 2}, DecoderUpsampleRates: []int{8, 5, 4, 3}, RMSNormEps: 1e-5, RoPETheta: 10000}
}

func (c Decoder12HzConfig) SamplesPerFrame() (int, error) {
	if c.CodebookDim <= 0 || c.LatentDim <= 0 || c.HiddenSize <= 0 || c.Heads <= 0 || c.HeadDim <= 0 || sizeProduct(c.Heads, c.HeadDim) <= 0 || c.Layers <= 0 || c.IntermediateSize <= 0 || c.Quantizers != 16 || c.CodebookSize != 2048 || c.DecoderDim <= 0 || c.RMSNormEps <= 0 || c.RoPETheta <= 0 || len(c.PreUpsampleRates) != 2 || len(c.DecoderUpsampleRates) != 4 {
		return 0, fmt.Errorf("invalid Qwen3-TTS Decoder12Hz topology: %+v", c)
	}
	factor := 1
	for _, rate := range append(append([]int(nil), c.PreUpsampleRates...), c.DecoderUpsampleRates...) {
		if rate <= 0 {
			return 0, fmt.Errorf("invalid Qwen3-TTS Decoder12Hz upsample rate=%d", rate)
		}
		factor = sizeProduct(factor, rate)
		if factor < 0 {
			return 0, fmt.Errorf("Qwen3-TTS Decoder12Hz upsample product overflows")
		}
	}
	return factor, nil
}

type decoderConv1D struct {
	weight, bias               []float32
	inChannels, outChannels, k int
	dilation                   int
}

type decoderTransConv1D struct {
	weight, bias               []float32
	inChannels, outChannels, k int
	stride                     int
}

type decoderLayer struct {
	inputNorm, postNorm, attentionScale, mlpScale []float32
	q, k, v, out                                  talkerLinear
	gate, up, down                                talkerLinear
}

type decoderConvNeXt struct {
	depthwise                   decoderConv1D
	normWeight, normBias, gamma []float32
	fc1, fc2                    talkerLinear
}

type decoderUpsampleStage struct {
	trans decoderTransConv1D
	block decoderConvNeXt
}

type snakeBeta struct{ alpha, beta []float32 }

type decoderResidual struct {
	act1, act2   snakeBeta
	conv1, conv2 decoderConv1D
}

type decoderBlockCPU struct {
	act snakeBeta
	up  decoderTransConv1D
	res [3]decoderResidual
}

// Decoder12HzCPU owns the F32 speech-tokenizer decoder tensors. Mutable
// activations are request-local; weights never alias the source.
type Decoder12HzCPU struct {
	cfg                               Decoder12HzConfig
	codebooks                         [][]float32
	firstProjection, restProjection   []float32
	preConv                           decoderConv1D
	inputProjection, outputProjection talkerLinear
	layers                            []decoderLayer
	finalNorm                         []float32
	preUpsample                       []decoderUpsampleStage
	decoderInit                       decoderConv1D
	decoderBlocks                     []decoderBlockCPU
	finalSnake                        snakeBeta
	finalConv                         decoderConv1D
}

func LoadDecoder12HzCPUFromDir(dir string) (*Decoder12HzCPU, error) {
	path := filepath.Join(dir, "speech_tokenizer")
	src, err := weights.OpenSafetensors(path)
	if err != nil {
		return nil, err
	}
	defer src.Close()
	return LoadDecoder12HzCPU(src, DefaultDecoder12HzConfig())
}

func LoadDecoder12HzCPU(src Float32TensorSource, cfg Decoder12HzConfig) (*Decoder12HzCPU, error) {
	if src == nil {
		return nil, fmt.Errorf("nil Qwen3-TTS Decoder12Hz tensor source")
	}
	spf, err := cfg.SamplesPerFrame()
	if err != nil || spf != decoder12HzSamplesPerFrame {
		return nil, fmt.Errorf("Qwen3-TTS Decoder12Hz samples/frame=%d: %w", spf, err)
	}
	m := &Decoder12HzCPU{cfg: cfg, codebooks: make([][]float32, cfg.Quantizers), layers: make([]decoderLayer, cfg.Layers), preUpsample: make([]decoderUpsampleStage, len(cfg.PreUpsampleRates)), decoderBlocks: make([]decoderBlockCPU, len(cfg.DecoderUpsampleRates))}
	first, err := loadNormalizedCodebook(src, "decoder.quantizer.rvq_first.vq.layers.0._codebook", cfg.CodebookSize, cfg.CodebookDim)
	if err != nil {
		return nil, err
	}
	m.codebooks[0] = first
	for i := 1; i < cfg.Quantizers; i++ {
		m.codebooks[i], err = loadNormalizedCodebook(src, fmt.Sprintf("decoder.quantizer.rvq_rest.vq.layers.%d._codebook", i-1), cfg.CodebookSize, cfg.CodebookDim)
		if err != nil {
			return nil, err
		}
	}
	if m.firstProjection, err = loadDecoderTensor(src, "decoder.quantizer.rvq_first.output_proj.weight", []int{cfg.CodebookDim * 2, cfg.CodebookDim, 1}); err != nil {
		return nil, err
	}
	if m.restProjection, err = loadDecoderTensor(src, "decoder.quantizer.rvq_rest.output_proj.weight", []int{cfg.CodebookDim * 2, cfg.CodebookDim, 1}); err != nil {
		return nil, err
	}
	if m.preConv, err = loadDecoderConv(src, "decoder.pre_conv.conv", cfg.CodebookDim*2, cfg.LatentDim, 3, 1); err != nil {
		return nil, err
	}
	if m.inputProjection, err = loadTalkerLinearOwned(src, "decoder.pre_transformer.input_proj", cfg.LatentDim, cfg.HiddenSize, true); err != nil {
		return nil, err
	}
	if m.outputProjection, err = loadTalkerLinearOwned(src, "decoder.pre_transformer.output_proj", cfg.HiddenSize, cfg.LatentDim, true); err != nil {
		return nil, err
	}
	attentionWidth := cfg.Heads * cfg.HeadDim
	for i := range m.layers {
		p := fmt.Sprintf("decoder.pre_transformer.layers.%d", i)
		l := &m.layers[i]
		if l.inputNorm, err = loadDecoderTensor(src, p+".input_layernorm.weight", []int{cfg.HiddenSize}); err != nil {
			return nil, err
		}
		if l.postNorm, err = loadDecoderTensor(src, p+".post_attention_layernorm.weight", []int{cfg.HiddenSize}); err != nil {
			return nil, err
		}
		if l.attentionScale, err = loadDecoderTensor(src, p+".self_attn_layer_scale.scale", []int{cfg.HiddenSize}); err != nil {
			return nil, err
		}
		if l.mlpScale, err = loadDecoderTensor(src, p+".mlp_layer_scale.scale", []int{cfg.HiddenSize}); err != nil {
			return nil, err
		}
		if l.q, err = loadTalkerLinearOwned(src, p+".self_attn.q_proj", cfg.HiddenSize, attentionWidth, false); err != nil {
			return nil, err
		}
		if l.k, err = loadTalkerLinearOwned(src, p+".self_attn.k_proj", cfg.HiddenSize, attentionWidth, false); err != nil {
			return nil, err
		}
		if l.v, err = loadTalkerLinearOwned(src, p+".self_attn.v_proj", cfg.HiddenSize, attentionWidth, false); err != nil {
			return nil, err
		}
		if l.out, err = loadTalkerLinearOwned(src, p+".self_attn.o_proj", attentionWidth, cfg.HiddenSize, false); err != nil {
			return nil, err
		}
		if l.gate, err = loadTalkerLinearOwned(src, p+".mlp.gate_proj", cfg.HiddenSize, cfg.IntermediateSize, false); err != nil {
			return nil, err
		}
		if l.up, err = loadTalkerLinearOwned(src, p+".mlp.up_proj", cfg.HiddenSize, cfg.IntermediateSize, false); err != nil {
			return nil, err
		}
		if l.down, err = loadTalkerLinearOwned(src, p+".mlp.down_proj", cfg.IntermediateSize, cfg.HiddenSize, false); err != nil {
			return nil, err
		}
	}
	if m.finalNorm, err = loadDecoderTensor(src, "decoder.pre_transformer.norm.weight", []int{cfg.HiddenSize}); err != nil {
		return nil, err
	}
	channels := cfg.LatentDim
	for i, rate := range cfg.PreUpsampleRates {
		p := fmt.Sprintf("decoder.upsample.%d", i)
		stage := &m.preUpsample[i]
		if stage.trans, err = loadDecoderTransConvDynamic(src, p+".0.conv", channels, rate, rate); err != nil {
			return nil, err
		}
		channels = stage.trans.outChannels
		if stage.block, err = loadDecoderConvNeXt(src, p+".1", channels); err != nil {
			return nil, err
		}
	}
	if channels != cfg.LatentDim {
		return nil, fmt.Errorf("Qwen3-TTS Decoder12Hz pre-upsample output channels=%d want latent=%d", channels, cfg.LatentDim)
	}
	if m.decoderInit, err = loadDecoderConv(src, "decoder.decoder.0.conv", cfg.LatentDim, cfg.DecoderDim, 7, 1); err != nil {
		return nil, err
	}
	channels = cfg.DecoderDim
	for i, rate := range cfg.DecoderUpsampleRates {
		outChannels := channels / 2
		if m.decoderBlocks[i], err = loadDecoderBlock(src, fmt.Sprintf("decoder.decoder.%d.block", i+1), channels, outChannels, rate); err != nil {
			return nil, err
		}
		channels = outChannels
	}
	if m.finalSnake, err = loadSnake(src, "decoder.decoder.5", channels); err != nil {
		return nil, err
	}
	if m.finalConv, err = loadDecoderConv(src, "decoder.decoder.6.conv", channels, 1, 7, 1); err != nil {
		return nil, err
	}
	return m, nil
}

func loadTalkerLinearOwned(src Float32TensorSource, prefix string, inDim, outDim int, bias bool) (talkerLinear, error) {
	weight, err := loadDecoderTensor(src, prefix+".weight", []int{outDim, inDim})
	if err != nil {
		return talkerLinear{}, err
	}
	l := talkerLinear{weight: weight, inDim: inDim, outDim: outDim}
	if bias {
		l.bias, err = loadDecoderTensor(src, prefix+".bias", []int{outDim})
	}
	return l, err
}

func loadDecoderTensor(src Float32TensorSource, name string, shape []int) ([]float32, error) {
	data, got, err := src.GetFloat32(name)
	if err != nil {
		return nil, fmt.Errorf("load %s: %w", name, err)
	}
	if !equalTalkerShape(got, shape) {
		return nil, fmt.Errorf("load %s: shape=%v want %v", name, got, shape)
	}
	want := sizeProduct(shape...)
	if want < 0 || len(data) != want {
		return nil, fmt.Errorf("load %s: values=%d want %d", name, len(data), want)
	}
	out := append([]float32(nil), data...)
	for i, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, fmt.Errorf("load %s: non-finite value at %d", name, i)
		}
	}
	return out, nil
}

func loadNormalizedCodebook(src Float32TensorSource, prefix string, vocab, dim int) ([]float32, error) {
	sum, err := loadDecoderTensor(src, prefix+".embedding_sum", []int{vocab, dim})
	if err != nil {
		return nil, err
	}
	usage, err := loadDecoderTensor(src, prefix+".cluster_usage", []int{vocab})
	if err != nil {
		return nil, err
	}
	for row := 0; row < vocab; row++ {
		den := max(usage[row], float32(1e-7))
		for col := 0; col < dim; col++ {
			sum[row*dim+col] /= den
		}
	}
	return sum, nil
}

func loadDecoderConv(src Float32TensorSource, prefix string, in, out, kernel, dilation int) (decoderConv1D, error) {
	w, err := loadDecoderTensor(src, prefix+".weight", []int{out, in, kernel})
	if err != nil {
		return decoderConv1D{}, err
	}
	b, err := loadDecoderTensor(src, prefix+".bias", []int{out})
	if err != nil {
		return decoderConv1D{}, err
	}
	return decoderConv1D{weight: w, bias: b, inChannels: in, outChannels: out, k: kernel, dilation: dilation}, nil
}

func loadDecoderTransConv(src Float32TensorSource, prefix string, in, out, kernel, stride int) (decoderTransConv1D, error) {
	conv, err := loadDecoderTransConvDynamic(src, prefix, in, kernel, stride)
	if err != nil {
		return decoderTransConv1D{}, err
	}
	if conv.outChannels != out {
		return decoderTransConv1D{}, fmt.Errorf("load %s.weight: output channels=%d want %d", prefix, conv.outChannels, out)
	}
	return conv, nil
}

func loadDecoderTransConvDynamic(src Float32TensorSource, prefix string, in, kernel, stride int) (decoderTransConv1D, error) {
	name := prefix + ".weight"
	data, shape, err := src.GetFloat32(name)
	if err != nil {
		return decoderTransConv1D{}, fmt.Errorf("load %s: %w", name, err)
	}
	if len(shape) != 3 || shape[0] != in || shape[1] <= 0 || shape[2] != kernel {
		return decoderTransConv1D{}, fmt.Errorf("load %s: shape=%v want [%d >0 %d]", name, shape, in, kernel)
	}
	out := shape[1]
	want := sizeProduct(in, out, kernel)
	if want < 0 || len(data) != want {
		return decoderTransConv1D{}, fmt.Errorf("load %s: values=%d want %d", name, len(data), want)
	}
	weight := append([]float32(nil), data...)
	for i, value := range weight {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return decoderTransConv1D{}, fmt.Errorf("load %s: non-finite value at %d", name, i)
		}
	}
	bias, err := loadDecoderTensor(src, prefix+".bias", []int{out})
	if err != nil {
		return decoderTransConv1D{}, err
	}
	return decoderTransConv1D{weight: weight, bias: bias, inChannels: in, outChannels: out, k: kernel, stride: stride}, nil
}

func loadSnake(src Float32TensorSource, prefix string, channels int) (snakeBeta, error) {
	a, err := loadDecoderTensor(src, prefix+".alpha", []int{channels})
	if err != nil {
		return snakeBeta{}, err
	}
	b, err := loadDecoderTensor(src, prefix+".beta", []int{channels})
	if err != nil {
		return snakeBeta{}, err
	}
	return snakeBeta{alpha: a, beta: b}, nil
}

func loadDecoderConvNeXt(src Float32TensorSource, prefix string, channels int) (decoderConvNeXt, error) {
	var b decoderConvNeXt
	var err error
	if b.depthwise, err = loadDecoderConv(src, prefix+".dwconv.conv", 1, channels, 7, 1); err != nil {
		return b, err
	}
	if b.normWeight, err = loadDecoderTensor(src, prefix+".norm.weight", []int{channels}); err != nil {
		return b, err
	}
	if b.normBias, err = loadDecoderTensor(src, prefix+".norm.bias", []int{channels}); err != nil {
		return b, err
	}
	if b.fc1, err = loadTalkerLinearOwned(src, prefix+".pwconv1", channels, 4*channels, true); err != nil {
		return b, err
	}
	if b.fc2, err = loadTalkerLinearOwned(src, prefix+".pwconv2", 4*channels, channels, true); err != nil {
		return b, err
	}
	if b.gamma, err = loadDecoderTensor(src, prefix+".gamma", []int{channels}); err != nil {
		return b, err
	}
	return b, nil
}

func loadDecoderResidual(src Float32TensorSource, prefix string, channels, dilation int) (decoderResidual, error) {
	var r decoderResidual
	var err error
	if r.act1, err = loadSnake(src, prefix+".act1", channels); err != nil {
		return r, err
	}
	if r.conv1, err = loadDecoderConv(src, prefix+".conv1.conv", channels, channels, 7, dilation); err != nil {
		return r, err
	}
	if r.act2, err = loadSnake(src, prefix+".act2", channels); err != nil {
		return r, err
	}
	if r.conv2, err = loadDecoderConv(src, prefix+".conv2.conv", channels, channels, 1, 1); err != nil {
		return r, err
	}
	return r, nil
}

func loadDecoderBlock(src Float32TensorSource, prefix string, in, out, rate int) (decoderBlockCPU, error) {
	var b decoderBlockCPU
	var err error
	if b.act, err = loadSnake(src, prefix+".0", in); err != nil {
		return b, err
	}
	if b.up, err = loadDecoderTransConv(src, prefix+".1.conv", in, out, 2*rate, rate); err != nil {
		return b, err
	}
	for i, dilation := range []int{1, 3, 9} {
		if b.res[i], err = loadDecoderResidual(src, fmt.Sprintf("%s.%d", prefix, i+2), out, dilation); err != nil {
			return b, err
		}
	}
	return b, nil
}
