package minicpmv

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/internal/checked"
	"github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/loader/weights"
)

type timmVisionLayer struct {
	norm1Weight, norm1Bias []float32
	norm2Weight, norm2Bias []float32
	qkv, projection        visionLinear
	fc1, fc2               visionLinear
}

// TimmSigLIPVisionCPU implements the fused-QKV timm VisionTransformer layout
// used by the published MiniCPM-V 2.0 checkpoint.
type TimmSigLIPVisionCPU struct {
	hidden, intermediate, layers, heads, headDim int
	imageSize, patchSize, channels               int
	positionSide                                 int
	eps                                          float32
	patchWeight, patchBias, positionEmbedding    []float32
	blocks                                       []timmVisionLayer
	finalNormWeight, finalNormBias               []float32
}

func LoadTimmSigLIPVisionCPUFromDir(dir string, cfg config.MiniCPMVConfig) (*TimmSigLIPVisionCPU, error) {
	src, err := weights.OpenSafetensors(dir)
	if err != nil {
		return nil, err
	}
	defer src.Close()
	return LoadTimmSigLIPVisionCPU(src, cfg)
}

func LoadTimmSigLIPVisionCPU(src Float32TensorSource, cfg config.MiniCPMVConfig) (*TimmSigLIPVisionCPU, error) {
	if src == nil {
		return nil, fmt.Errorf("nil MiniCPM-V timm vision tensor source")
	}
	s := cfg.MiniCPMVSummary()
	if cfg.VisionEncoder != "vit_so400m_patch14_siglip_384.webli" || s.VisionHiddenSize <= 0 || s.VisionLayers <= 0 || s.VisionHeads <= 0 || s.VisionHiddenSize%s.VisionHeads != 0 || s.ImageSize <= 0 || s.PatchSize <= 0 || s.ImageSize%s.PatchSize != 0 {
		return nil, fmt.Errorf("unsupported MiniCPM-V timm vision encoder=%q hidden=%d layers=%d heads=%d image=%d patch=%d", cfg.VisionEncoder, s.VisionHiddenSize, s.VisionLayers, s.VisionHeads, s.ImageSize, s.PatchSize)
	}
	const channels = 3
	if s.VisionIntermediate <= 0 {
		return nil, fmt.Errorf("invalid MiniCPM-V timm vision intermediate=%d", s.VisionIntermediate)
	}
	m := &TimmSigLIPVisionCPU{hidden: s.VisionHiddenSize, intermediate: s.VisionIntermediate, layers: s.VisionLayers, heads: s.VisionHeads, headDim: s.VisionHiddenSize / s.VisionHeads, imageSize: s.ImageSize, patchSize: s.PatchSize, channels: channels, eps: 1e-6, blocks: make([]timmVisionLayer, s.VisionLayers)}
	var err error
	if m.patchWeight, err = loadTextTensor(src, "vpm.patch_embed.proj.weight", []int{m.hidden, channels, m.patchSize, m.patchSize}); err != nil {
		return nil, err
	}
	if m.patchBias, err = loadTextTensor(src, "vpm.patch_embed.proj.bias", []int{m.hidden}); err != nil {
		return nil, err
	}
	position, positionShape, err := src.GetFloat32("vpm.pos_embed")
	if err != nil {
		return nil, fmt.Errorf("load vpm.pos_embed: %w", err)
	}
	if len(positionShape) != 3 || positionShape[0] != 1 || positionShape[2] != m.hidden || len(position) != positionShape[1]*m.hidden {
		return nil, fmt.Errorf("load vpm.pos_embed: shape=%v want [1 square,%d]", positionShape, m.hidden)
	}
	m.positionSide = int(math.Sqrt(float64(positionShape[1])))
	if m.positionSide*m.positionSide != positionShape[1] {
		return nil, fmt.Errorf("load vpm.pos_embed: positions=%d are not square", positionShape[1])
	}
	if err := validateFiniteTextOutput("timm vision position", position); err != nil {
		return nil, err
	}
	m.positionEmbedding = position
	if m.finalNormWeight, err = loadTextTensor(src, "vpm.norm.weight", []int{m.hidden}); err != nil {
		return nil, err
	}
	if m.finalNormBias, err = loadTextTensor(src, "vpm.norm.bias", []int{m.hidden}); err != nil {
		return nil, err
	}
	for i := range m.blocks {
		prefix := fmt.Sprintf("vpm.blocks.%d", i)
		b := &m.blocks[i]
		if b.norm1Weight, err = loadTextTensor(src, prefix+".norm1.weight", []int{m.hidden}); err != nil {
			return nil, err
		}
		if b.norm1Bias, err = loadTextTensor(src, prefix+".norm1.bias", []int{m.hidden}); err != nil {
			return nil, err
		}
		if b.norm2Weight, err = loadTextTensor(src, prefix+".norm2.weight", []int{m.hidden}); err != nil {
			return nil, err
		}
		if b.norm2Bias, err = loadTextTensor(src, prefix+".norm2.bias", []int{m.hidden}); err != nil {
			return nil, err
		}
		if b.qkv, err = loadVisionLinear(src, prefix+".attn.qkv", m.hidden, 3*m.hidden); err != nil {
			return nil, err
		}
		if b.projection, err = loadVisionLinear(src, prefix+".attn.proj", m.hidden, m.hidden); err != nil {
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

func (m *TimmSigLIPVisionCPU) EncodeImage(pixelValues []float32, shape [4]int) ([]float32, error) {
	if m == nil {
		return nil, fmt.Errorf("nil MiniCPM-V timm vision tower")
	}
	height, width := shape[2], shape[3]
	if shape[0] != 1 || shape[1] != m.channels || height <= 0 || width <= 0 || height > m.imageSize || width > m.imageSize || height%m.patchSize != 0 || width%m.patchSize != 0 {
		return nil, fmt.Errorf("invalid MiniCPM-V timm image shape=%v", shape)
	}
	pixels, ok := checked.MulInt(m.channels, height)
	if ok {
		pixels, ok = checked.MulInt(pixels, width)
	}
	if !ok || len(pixelValues) != pixels {
		return nil, fmt.Errorf("invalid MiniCPM-V timm pixels=%d want %d", len(pixelValues), pixels)
	}
	if err := validateFiniteTextOutput("timm vision pixel", pixelValues); err != nil {
		return nil, err
	}
	hidden, tokens, err := m.patchEmbed(pixelValues, height, width)
	if err != nil {
		return nil, err
	}
	for i := range m.blocks {
		hidden, err = m.blocks[i].forward(hidden, tokens, m.hidden, m.heads, m.headDim, m.eps)
		if err != nil {
			return nil, fmt.Errorf("MiniCPM-V timm vision layer %d: %w", i, err)
		}
	}
	out := make([]float32, len(hidden))
	if !simd.LayerNormLastAxisTo(out, hidden, tokens, m.hidden, m.finalNormWeight, m.finalNormBias, m.eps) {
		return nil, fmt.Errorf("MiniCPM-V timm final LayerNorm failed")
	}
	if err := validateFiniteTextOutput("timm vision output", out); err != nil {
		return nil, err
	}
	return out, nil
}

func (m *TimmSigLIPVisionCPU) patchEmbed(pixelValues []float32, height, width int) ([]float32, int, error) {
	gridH, gridW := height/m.patchSize, width/m.patchSize
	tokens, ok := checked.MulInt(gridH, gridW)
	if !ok || tokens <= 0 {
		return nil, 0, fmt.Errorf("MiniCPM-V timm patch count overflows")
	}
	out := make([]float32, tokens*m.hidden)
	plane := height * width
	kernelSize := m.channels * m.patchSize * m.patchSize
	for py := 0; py < gridH; py++ {
		positionY := py * m.positionSide / gridH
		for px := 0; px < gridW; px++ {
			positionX := px * m.positionSide / gridW
			positionID := positionY*m.positionSide + positionX
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

func (b *timmVisionLayer) forward(input []float32, tokens, hidden, heads, headDim int, eps float32) ([]float32, error) {
	norm1 := make([]float32, len(input))
	if !simd.LayerNormLastAxisTo(norm1, input, tokens, hidden, b.norm1Weight, b.norm1Bias, eps) {
		return nil, fmt.Errorf("layer norm 1 failed")
	}
	q, k, v := make([]float32, len(input)), make([]float32, len(input)), make([]float32, len(input))
	packed := make([]float32, 3*hidden)
	for token := 0; token < tokens; token++ {
		if err := b.qkv.forward(packed, norm1[token*hidden:(token+1)*hidden]); err != nil {
			return nil, err
		}
		copy(q[token*hidden:(token+1)*hidden], packed[:hidden])
		copy(k[token*hidden:(token+1)*hidden], packed[hidden:2*hidden])
		copy(v[token*hidden:(token+1)*hidden], packed[2*hidden:])
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
		if err := b.projection.forward(projected[token*hidden:(token+1)*hidden], attention[token*hidden:(token+1)*hidden]); err != nil {
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
	mlp := make([]float32, len(input))
	intermediate := make([]float32, b.fc1.outDim)
	for token := 0; token < tokens; token++ {
		if err := b.fc1.forward(intermediate, norm2[token*hidden:(token+1)*hidden]); err != nil {
			return nil, err
		}
		if !simd.GELUTanhTo(intermediate, intermediate) {
			return nil, fmt.Errorf("MLP GELU failed")
		}
		if err := b.fc2.forward(mlp[token*hidden:(token+1)*hidden], intermediate); err != nil {
			return nil, err
		}
	}
	out := make([]float32, len(input))
	if !simd.VecAddTo(out, residual, mlp) {
		return nil, fmt.Errorf("MLP residual failed")
	}
	return out, nil
}

var _ VisionTower = (*TimmSigLIPVisionCPU)(nil)
