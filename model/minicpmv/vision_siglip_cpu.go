package minicpmv

import (
	"fmt"
	"math"
	"strings"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/internal/checked"
	"github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/loader/weights"
)

type visionLinear struct {
	weight        []float32
	bias          []float32
	inDim, outDim int
}

type siglipVisionLayer struct {
	norm1Weight, norm1Bias []float32
	norm2Weight, norm2Bias []float32
	q, k, v, out           visionLinear
	fc1, fc2               visionLinear
}

// SigLIPVisionCPU owns decoded F32 weights for the MiniCPM-V/O SigLIP tower.
// It implements the upstream full-attention, pre-norm encoder without pooling.
type SigLIPVisionCPU struct {
	hidden, intermediate, layers, heads, headDim int
	imageSize, patchSize, channels, positions    int
	eps                                          float32
	patchWeight, patchBias, positionEmbedding    []float32
	blocks                                       []siglipVisionLayer
	postNormWeight, postNormBias                 []float32
}

func LoadSigLIPVisionCPUFromDir(dir string, cfg config.MiniCPMVConfig) (*SigLIPVisionCPU, error) {
	src, err := weights.OpenSafetensors(dir)
	if err != nil {
		return nil, err
	}
	defer src.Close()
	return LoadSigLIPVisionCPU(src, cfg)
}

func LoadSigLIPVisionCPU(src Float32TensorSource, cfg config.MiniCPMVConfig) (*SigLIPVisionCPU, error) {
	if src == nil {
		return nil, fmt.Errorf("nil MiniCPM-V/O SigLIP tensor source")
	}
	v := cfg.VisionConfig
	if v == nil || !strings.Contains(strings.ToLower(v.ModelType), "siglip") {
		return nil, fmt.Errorf("MiniCPM-V/O SigLIP vision_config is required")
	}
	channels := v.NumChannels
	if channels == 0 {
		channels = 3
	}
	eps := v.LayerNormEps
	if eps == 0 {
		eps = 1e-6
	}
	hiddenAct := v.HiddenAct
	if hiddenAct == "" {
		hiddenAct = "gelu_pytorch_tanh"
	}
	if v.HiddenSize <= 0 || v.IntermediateSize <= 0 || v.NumHiddenLayers <= 0 || v.NumAttentionHeads <= 0 || v.HiddenSize%v.NumAttentionHeads != 0 || v.ImageSize <= 0 || v.PatchSize <= 0 || v.ImageSize%v.PatchSize != 0 || channels <= 0 {
		return nil, fmt.Errorf("invalid MiniCPM-V/O SigLIP dimensions hidden=%d intermediate=%d layers=%d heads=%d image=%d patch=%d channels=%d", v.HiddenSize, v.IntermediateSize, v.NumHiddenLayers, v.NumAttentionHeads, v.ImageSize, v.PatchSize, channels)
	}
	if hiddenAct != "gelu_pytorch_tanh" || eps <= 0 || !finite(eps) || v.AttentionDropout != 0 {
		return nil, fmt.Errorf("unsupported MiniCPM-V/O SigLIP policy activation=%q eps=%g dropout=%g", hiddenAct, eps, v.AttentionDropout)
	}
	grid := v.ImageSize / v.PatchSize
	positions, ok := checked.MulInt(grid, grid)
	if !ok {
		return nil, fmt.Errorf("MiniCPM-V/O SigLIP position count overflows")
	}
	m := &SigLIPVisionCPU{hidden: v.HiddenSize, intermediate: v.IntermediateSize, layers: v.NumHiddenLayers, heads: v.NumAttentionHeads, headDim: v.HiddenSize / v.NumAttentionHeads, imageSize: v.ImageSize, patchSize: v.PatchSize, channels: channels, positions: positions, eps: float32(eps), blocks: make([]siglipVisionLayer, v.NumHiddenLayers)}
	var err error
	if m.patchWeight, err = loadTextTensor(src, "vpm.embeddings.patch_embedding.weight", []int{m.hidden, channels, m.patchSize, m.patchSize}); err != nil {
		return nil, err
	}
	if m.patchBias, err = loadTextTensor(src, "vpm.embeddings.patch_embedding.bias", []int{m.hidden}); err != nil {
		return nil, err
	}
	if m.positionEmbedding, err = loadTextTensor(src, "vpm.embeddings.position_embedding.weight", []int{m.positions, m.hidden}); err != nil {
		return nil, err
	}
	if m.postNormWeight, err = loadTextTensor(src, "vpm.post_layernorm.weight", []int{m.hidden}); err != nil {
		return nil, err
	}
	if m.postNormBias, err = loadTextTensor(src, "vpm.post_layernorm.bias", []int{m.hidden}); err != nil {
		return nil, err
	}
	for i := range m.blocks {
		prefix := fmt.Sprintf("vpm.encoder.layers.%d", i)
		b := &m.blocks[i]
		if b.norm1Weight, err = loadTextTensor(src, prefix+".layer_norm1.weight", []int{m.hidden}); err != nil {
			return nil, err
		}
		if b.norm1Bias, err = loadTextTensor(src, prefix+".layer_norm1.bias", []int{m.hidden}); err != nil {
			return nil, err
		}
		if b.norm2Weight, err = loadTextTensor(src, prefix+".layer_norm2.weight", []int{m.hidden}); err != nil {
			return nil, err
		}
		if b.norm2Bias, err = loadTextTensor(src, prefix+".layer_norm2.bias", []int{m.hidden}); err != nil {
			return nil, err
		}
		if b.q, err = loadVisionLinear(src, prefix+".self_attn.q_proj", m.hidden, m.hidden); err != nil {
			return nil, err
		}
		if b.k, err = loadVisionLinear(src, prefix+".self_attn.k_proj", m.hidden, m.hidden); err != nil {
			return nil, err
		}
		if b.v, err = loadVisionLinear(src, prefix+".self_attn.v_proj", m.hidden, m.hidden); err != nil {
			return nil, err
		}
		if b.out, err = loadVisionLinear(src, prefix+".self_attn.out_proj", m.hidden, m.hidden); err != nil {
			return nil, err
		}
		if b.fc1, err = loadVisionLinear(src, prefix+".mlp.fc1", m.hidden, m.intermediate); err != nil {
			return nil, err
		}
		if b.fc2, err = loadVisionLinear(src, prefix+".mlp.fc2", m.intermediate, m.hidden); err != nil {
			return nil, err
		}
	}
	return m, nil
}

func loadVisionLinear(src Float32TensorSource, prefix string, inDim, outDim int) (visionLinear, error) {
	weight, err := loadTextTensor(src, prefix+".weight", []int{outDim, inDim})
	if err != nil {
		return visionLinear{}, err
	}
	bias, err := loadTextTensor(src, prefix+".bias", []int{outDim})
	if err != nil {
		return visionLinear{}, err
	}
	return visionLinear{weight: weight, bias: bias, inDim: inDim, outDim: outDim}, nil
}

func (l visionLinear) forward(dst, input []float32) error {
	if len(dst) != l.outDim || len(input) != l.inDim || !simd.GemvRows(dst, input, l.weight, l.outDim, l.inDim) {
		return fmt.Errorf("invalid MiniCPM-V/O vision linear buffers out/in=%d/%d want %d/%d", len(dst), len(input), l.outDim, l.inDim)
	}
	if len(l.bias) != len(dst) {
		return fmt.Errorf("invalid MiniCPM-V/O vision linear bias=%d want %d", len(l.bias), len(dst))
	}
	for i := range dst {
		dst[i] += l.bias[i]
	}
	return nil
}

func (m *SigLIPVisionCPU) EncodeImage(pixelValues []float32, shape [4]int) ([]float32, error) {
	if m == nil {
		return nil, fmt.Errorf("nil MiniCPM-V/O SigLIP vision tower")
	}
	height, width := shape[2], shape[3]
	if shape[0] != 1 || shape[1] != m.channels || height <= 0 || width <= 0 || height > m.imageSize || width > m.imageSize || height%m.patchSize != 0 || width%m.patchSize != 0 {
		return nil, fmt.Errorf("invalid MiniCPM-V/O SigLIP image shape=%v want [1 %d H W] with positive patch-aligned H/W <= %d", shape, m.channels, m.imageSize)
	}
	pixels, ok := checked.MulInt(m.channels, height)
	if ok {
		pixels, ok = checked.MulInt(pixels, width)
	}
	if !ok || len(pixelValues) != pixels {
		return nil, fmt.Errorf("invalid MiniCPM-V/O SigLIP pixel values=%d want %d", len(pixelValues), pixels)
	}
	if err := validateFiniteTextOutput("SigLIP pixel", pixelValues); err != nil {
		return nil, err
	}
	hidden, tokens, err := m.patchEmbed(pixelValues, height, width)
	if err != nil {
		return nil, err
	}
	for i := range m.blocks {
		hidden, err = m.blocks[i].forward(hidden, tokens, m.hidden, m.heads, m.headDim, m.eps)
		if err != nil {
			return nil, fmt.Errorf("MiniCPM-V/O SigLIP layer %d: %w", i, err)
		}
	}
	out := make([]float32, len(hidden))
	if !simd.LayerNormLastAxisTo(out, hidden, tokens, m.hidden, m.postNormWeight, m.postNormBias, m.eps) {
		return nil, fmt.Errorf("MiniCPM-V/O SigLIP post LayerNorm failed")
	}
	if err := validateFiniteTextOutput("SigLIP output", out); err != nil {
		return nil, err
	}
	return out, nil
}

func (m *SigLIPVisionCPU) patchEmbed(pixelValues []float32, height, width int) ([]float32, int, error) {
	gridH, gridW := height/m.patchSize, width/m.patchSize
	tokens, ok := checked.MulInt(gridH, gridW)
	if !ok || tokens <= 0 {
		return nil, 0, fmt.Errorf("MiniCPM-V/O SigLIP patch count overflows")
	}
	out := make([]float32, tokens*m.hidden)
	plane := height * width
	kernelSize := m.channels * m.patchSize * m.patchSize
	positionSide := m.imageSize / m.patchSize
	for py := 0; py < gridH; py++ {
		positionY := py * positionSide / gridH
		for px := 0; px < gridW; px++ {
			positionX := px * positionSide / gridW
			positionID := positionY*positionSide + positionX
			position := py*gridW + px
			row := out[position*m.hidden : (position+1)*m.hidden]
			for channel := 0; channel < m.hidden; channel++ {
				weight := m.patchWeight[channel*kernelSize : (channel+1)*kernelSize]
				sum := m.patchBias[channel]
				index := 0
				for c := 0; c < m.channels; c++ {
					for ky := 0; ky < m.patchSize; ky++ {
						pixelRow := c*plane + (py*m.patchSize+ky)*width + px*m.patchSize
						for kx := 0; kx < m.patchSize; kx++ {
							sum += pixelValues[pixelRow+kx] * weight[index]
							index++
						}
					}
				}
				row[channel] = sum + m.positionEmbedding[positionID*m.hidden+channel]
			}
		}
	}
	return out, tokens, nil
}

func (b *siglipVisionLayer) forward(input []float32, tokens, hidden, heads, headDim int, eps float32) ([]float32, error) {
	norm1 := make([]float32, len(input))
	if !simd.LayerNormLastAxisTo(norm1, input, tokens, hidden, b.norm1Weight, b.norm1Bias, eps) {
		return nil, fmt.Errorf("layer norm 1 failed")
	}
	q, k, v := make([]float32, len(input)), make([]float32, len(input)), make([]float32, len(input))
	for token := 0; token < tokens; token++ {
		row := norm1[token*hidden : (token+1)*hidden]
		if err := b.q.forward(q[token*hidden:(token+1)*hidden], row); err != nil {
			return nil, err
		}
		if err := b.k.forward(k[token*hidden:(token+1)*hidden], row); err != nil {
			return nil, err
		}
		if err := b.v.forward(v[token*hidden:(token+1)*hidden], row); err != nil {
			return nil, err
		}
	}
	attention := make([]float32, len(input))
	scores := make([]float32, tokens)
	scale := float32(1 / math.Sqrt(float64(headDim)))
	for query := 0; query < tokens; query++ {
		for head := 0; head < heads; head++ {
			qHead := q[query*hidden+head*headDim : query*hidden+(head+1)*headDim]
			for key := 0; key < tokens; key++ {
				kHead := k[key*hidden+head*headDim : key*hidden+(head+1)*headDim]
				scores[key] = simd.Sdot(qHead, kHead) * scale
			}
			if !simd.SoftmaxInPlace(scores) {
				return nil, fmt.Errorf("attention softmax failed")
			}
			outHead := attention[query*hidden+head*headDim : query*hidden+(head+1)*headDim]
			for key := 0; key < tokens; key++ {
				vHead := v[key*hidden+head*headDim : key*hidden+(head+1)*headDim]
				simd.VecScaleAdd(outHead, outHead, vHead, scores[key])
			}
		}
	}
	projected := make([]float32, len(input))
	for token := 0; token < tokens; token++ {
		if err := b.out.forward(projected[token*hidden:(token+1)*hidden], attention[token*hidden:(token+1)*hidden]); err != nil {
			return nil, err
		}
	}
	residual := make([]float32, len(input))
	if !simd.VecAddTo(residual, input, projected) {
		return nil, fmt.Errorf("attention residual failed")
	}
	norm2 := make([]float32, len(input))
	if !simd.LayerNormLastAxisTo(norm2, residual, tokens, hidden, b.norm2Weight, b.norm2Bias, eps) {
		return nil, fmt.Errorf("layer norm 2 failed")
	}
	mlpOut := make([]float32, len(input))
	intermediate := make([]float32, b.fc1.outDim)
	for token := 0; token < tokens; token++ {
		if err := b.fc1.forward(intermediate, norm2[token*hidden:(token+1)*hidden]); err != nil {
			return nil, err
		}
		if !simd.GELUTanhTo(intermediate, intermediate) {
			return nil, fmt.Errorf("MLP GELU failed")
		}
		if err := b.fc2.forward(mlpOut[token*hidden:(token+1)*hidden], intermediate); err != nil {
			return nil, err
		}
	}
	out := make([]float32, len(input))
	if !simd.VecAddTo(out, residual, mlpOut) {
		return nil, fmt.Errorf("MLP residual failed")
	}
	return out, nil
}
