package nemotronasr

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

const asrAttentionHeads = 8
const asrAttentionHeadWidth = 128

// Encoder0Attention owns the released first encoder attention weights. Only
// complete unmasked offline windows, without key/value cache, are supported.
// The later convolution, FF2, encoder layers and RNN-T are separate.
type Encoder0Attention struct {
	qkv            *Encoder0QKV
	relativeWeight []float32
	biasU, biasV   []float32
	outputWeight   []float32
}

func LoadEncoder0Attention(file *safetensors.File) (*Encoder0Attention, error) {
	if file == nil {
		return nil, fmt.Errorf("nil Nemotron ASR checkpoint")
	}
	qkv, err := LoadEncoder0QKV(file)
	if err != nil {
		return nil, err
	}
	prefix := "encoder.layers.0.self_attn."
	load := func(name string, dims ...int) ([]float32, error) {
		values, shape, err := file.GetFloat32(prefix + name)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		if len(shape) != len(dims) {
			return nil, fmt.Errorf("%s: invalid rank %v", name, shape)
		}
		size := 1
		for i, dim := range dims {
			if shape[i] != dim {
				return nil, fmt.Errorf("%s: invalid shape %v", name, shape)
			}
			size *= dim
		}
		if len(values) != size {
			return nil, fmt.Errorf("%s: invalid element count", name)
		}
		for _, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("%s: non-finite weight", name)
			}
		}
		return values, nil
	}
	m := &Encoder0Attention{qkv: qkv}
	if m.relativeWeight, err = load("relative_k_proj.weight", encoderWidth, encoderWidth); err != nil {
		return nil, err
	}
	if m.biasU, err = load("bias_u", asrAttentionHeads, asrAttentionHeadWidth); err != nil {
		return nil, err
	}
	if m.biasV, err = load("bias_v", asrAttentionHeads, asrAttentionHeadWidth); err != nil {
		return nil, err
	}
	if m.outputWeight, err = load("o_proj.weight", encoderWidth, encoderWidth); err != nil {
		return nil, err
	}
	return m, nil
}

// ForwardOffline returns independent owned attention and residual arrays for
// an unmasked [rows,1024] FF1 window starting at position zero. The released
// encoder applies a chunk mask even to five rows; this direct operator path
// does not implement that mask or the key/value cache.
func (m *Encoder0Attention) ForwardOffline(input []float32, rows int) (attention, residual []float32, err error) {
	return m.forwardOffline(input, rows, -1)
}

// ForwardOfflineLookahead applies the released chunk rule for the supported
// zero- and three-token lookahead settings. No cached key/value state exists.
func (m *Encoder0Attention) ForwardOfflineLookahead(input []float32, rows, lookahead int) (attention, residual []float32, err error) {
	if lookahead != 0 && lookahead != 3 {
		return nil, nil, fmt.Errorf("unsupported Nemotron ASR encoder-0 lookahead")
	}
	return m.forwardOffline(input, rows, lookahead)
}

func (m *Encoder0Attention) forwardOffline(input []float32, rows, lookahead int) (attention, residual []float32, err error) {
	if m == nil || m.qkv == nil || len(m.relativeWeight) != encoderWidth*encoderWidth || len(m.biasU) != encoderWidth || len(m.biasV) != encoderWidth || len(m.outputWeight) != encoderWidth*encoderWidth {
		return nil, nil, fmt.Errorf("invalid Nemotron ASR encoder-0 attention weights")
	}
	if rows < 1 || rows > 5 || len(input) != rows*encoderWidth {
		return nil, nil, fmt.Errorf("invalid Nemotron ASR encoder-0 attention window")
	}
	q, k, v, err := m.qkv.Project(input, rows)
	if err != nil {
		return nil, nil, err
	}
	encoded := encoder0RelativePositions(rows)
	positions := 2*rows - 1
	relative := make([]float32, len(encoded))
	if !simd.DenseNTTo(relative, encoded, m.relativeWeight, positions, encoderWidth, encoderWidth, 1, encoderWidth, encoderWidth, encoderWidth) {
		return nil, nil, fmt.Errorf("Nemotron ASR relative projection rejected shape")
	}
	mixed := make([]float32, rows*encoderWidth)
	var scores [5]float32
	scaling := float32(1 / math.Sqrt(asrAttentionHeadWidth))
	for head := 0; head < asrAttentionHeads; head++ {
		base := head * asrAttentionHeadWidth
		for row := 0; row < rows; row++ {
			for source := 0; source < rows; source++ {
				var content, positional float32
				// _rel_shift maps a relative index to position (L-1)+source-row.
				pos := rows - 1 + source - row
				for dim := 0; dim < asrAttentionHeadWidth; dim++ {
					off := base + dim
					content += (q[row*encoderWidth+off] + m.biasU[off]) * k[source*encoderWidth+off]
					positional += (q[row*encoderWidth+off] + m.biasV[off]) * relative[pos*encoderWidth+off]
				}
				scores[source] = (content + positional) * scaling
				if lookahead >= 0 && source/(lookahead+1) > row/(lookahead+1) {
					scores[source] = float32(math.Inf(-1))
				}
			}
			if !simd.SoftmaxInPlace(scores[:rows]) {
				return nil, nil, fmt.Errorf("Nemotron ASR attention softmax failed")
			}
			for dim := 0; dim < asrAttentionHeadWidth; dim++ {
				var sum float32
				for source := 0; source < rows; source++ {
					sum += scores[source] * v[source*encoderWidth+base+dim]
				}
				mixed[row*encoderWidth+base+dim] = sum
			}
		}
	}
	attention = make([]float32, len(input))
	if !simd.DenseNTTo(attention, mixed, m.outputWeight, rows, encoderWidth, encoderWidth, 1, encoderWidth, encoderWidth, encoderWidth) {
		return nil, nil, fmt.Errorf("Nemotron ASR attention output rejected shape")
	}
	residual = make([]float32, len(input))
	for i, value := range attention {
		residual[i] = input[i] + value
	}
	return attention, residual, nil
}

// encoder0RelativePositions follows the released Transformer-XL convention:
// positions L-1 through -(L-1), with interleaved F32 sine and cosine.
func encoder0RelativePositions(rows int) []float32 {
	positions := 2*rows - 1
	encoded := make([]float32, positions*encoderWidth)
	var invFreq [encoderWidth / 2]float32
	for freq := range invFreq {
		invFreq[freq] = float32(1 / math.Pow(10000, float64(2*freq)/encoderWidth))
	}
	for pos := 0; pos < positions; pos++ {
		p := float32(rows - 1 - pos)
		for freq, inv := range invFreq {
			angle := float64(p * inv)
			encoded[pos*encoderWidth+2*freq] = float32(math.Sin(angle))
			encoded[pos*encoderWidth+2*freq+1] = float32(math.Cos(angle))
		}
	}
	return encoded
}
