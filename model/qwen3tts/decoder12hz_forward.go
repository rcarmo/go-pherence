package qwen3tts

import (
	"fmt"
	"math"

	simd "github.com/rcarmo/go-pherence/backends/simd/runtime"
)

// DecodeWaveform joins staged semantic/acoustic outputs and executes the F32
// tokenizer decoder. Output is owned mono PCM at 24kHz, clamped to [-1,1].
func (m *Decoder12HzCPU) DecodeWaveform(plan RuntimeRequestPlan, semantic, acoustic []uint32) ([]float32, error) {
	if m == nil {
		return nil, fmt.Errorf("nil Qwen3-TTS Decoder12Hz runtime")
	}
	contract, err := NewDecoder12HzExecutionContract(plan)
	if err != nil {
		return nil, err
	}
	codes, err := contract.JoinInput(semantic, acoustic)
	if err != nil {
		return nil, err
	}
	out, err := m.decodeCodes(codes, len(semantic))
	if err != nil {
		return nil, err
	}
	if err := contract.ValidateOutputForFrames(out, len(semantic)); err != nil {
		return nil, err
	}
	return out, nil
}

func (m *Decoder12HzCPU) decodeCodes(codes []uint32, frames int) ([]float32, error) {
	if frames <= 0 || len(codes) != frames*m.cfg.Quantizers {
		return nil, fmt.Errorf("invalid Qwen3-TTS Decoder12Hz codes=%d frames=%d", len(codes), frames)
	}
	c := m.cfg.CodebookDim
	quantized := make([]float32, frames*2*c)
	first, rest := make([]float32, c), make([]float32, c)
	firstProjected, restProjected := make([]float32, 2*c), make([]float32, 2*c)
	for frame := 0; frame < frames; frame++ {
		clear(first)
		clear(rest)
		semantic := int(codes[frame*m.cfg.Quantizers] % uint32(m.cfg.CodebookSize))
		copy(first, m.codebooks[0][semantic*c:(semantic+1)*c])
		for group := 1; group < m.cfg.Quantizers; group++ {
			code := int(codes[frame*m.cfg.Quantizers+group])
			if code < 0 || code >= m.cfg.CodebookSize {
				return nil, fmt.Errorf("Qwen3-TTS Decoder12Hz code frame=%d group=%d value=%d", frame, group, code)
			}
			row := m.codebooks[group][code*c : (code+1)*c]
			for i := range rest {
				rest[i] += row[i]
			}
		}
		if !simd.GemvRows(firstProjected, first, m.firstProjection, 2*c, c) || !simd.GemvRows(restProjected, rest, m.restProjection, 2*c, c) {
			return nil, fmt.Errorf("Qwen3-TTS Decoder12Hz codebook projection failed")
		}
		for channel := 0; channel < 2*c; channel++ {
			quantized[channel*frames+frame] = firstProjected[channel] + restProjected[channel]
		}
	}
	hidden, length, err := m.preConv.forward(quantized, frames)
	if err != nil {
		return nil, err
	}
	timeMajor := channelToTime(hidden, m.cfg.LatentDim, length)
	projected := make([]float32, length*m.cfg.HiddenSize)
	for row := 0; row < length; row++ {
		if err := m.inputProjection.forward(projected[row*m.cfg.HiddenSize:(row+1)*m.cfg.HiddenSize], timeMajor[row*m.cfg.LatentDim:(row+1)*m.cfg.LatentDim]); err != nil {
			return nil, err
		}
	}
	rope := simd.BuildRoPEFreqs(length, m.cfg.HeadDim/2, m.cfg.HeadDim, float64(m.cfg.RoPETheta))
	for i := range m.layers {
		projected, err = m.layers[i].forward(projected, length, m.cfg, rope)
		if err != nil {
			return nil, fmt.Errorf("Qwen3-TTS Decoder12Hz transformer layer %d: %w", i, err)
		}
	}
	normed := make([]float32, len(projected))
	for row := 0; row < length; row++ {
		copy(normed[row*m.cfg.HiddenSize:(row+1)*m.cfg.HiddenSize], projected[row*m.cfg.HiddenSize:(row+1)*m.cfg.HiddenSize])
		if !simd.RMSNormTo(normed[row*m.cfg.HiddenSize:(row+1)*m.cfg.HiddenSize], m.finalNorm, m.cfg.RMSNormEps) {
			return nil, fmt.Errorf("Qwen3-TTS Decoder12Hz final RMSNorm failed")
		}
	}
	outTime := make([]float32, length*m.cfg.LatentDim)
	for row := 0; row < length; row++ {
		if err := m.outputProjection.forward(outTime[row*m.cfg.LatentDim:(row+1)*m.cfg.LatentDim], normed[row*m.cfg.HiddenSize:(row+1)*m.cfg.HiddenSize]); err != nil {
			return nil, err
		}
	}
	hidden = timeToChannel(outTime, length, m.cfg.LatentDim)
	for i := range m.preUpsample {
		hidden, length, err = m.preUpsample[i].forward(hidden, length)
		if err != nil {
			return nil, fmt.Errorf("Qwen3-TTS Decoder12Hz pre-upsample %d: %w", i, err)
		}
	}
	hidden, length, err = m.decoderInit.forward(hidden, length)
	if err != nil {
		return nil, err
	}
	for i := range m.decoderBlocks {
		hidden, length, err = m.decoderBlocks[i].forward(hidden, length)
		if err != nil {
			return nil, fmt.Errorf("Qwen3-TTS Decoder12Hz decoder block %d: %w", i, err)
		}
	}
	if err := m.finalSnake.forwardInPlace(hidden, length); err != nil {
		return nil, err
	}
	hidden, length, err = m.finalConv.forward(hidden, length)
	if err != nil {
		return nil, err
	}
	if length != frames*decoder12HzSamplesPerFrame || len(hidden) != length {
		return nil, fmt.Errorf("Qwen3-TTS Decoder12Hz output samples=%d want=%d", len(hidden), frames*decoder12HzSamplesPerFrame)
	}
	for i, value := range hidden {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return nil, fmt.Errorf("Qwen3-TTS Decoder12Hz non-finite sample at %d", i)
		}
		hidden[i] = max(float32(-1), min(float32(1), value))
	}
	return hidden, nil
}

func (l decoderLayer) forward(input []float32, rows int, cfg Decoder12HzConfig, rope []float32) ([]float32, error) {
	h := cfg.HiddenSize
	norm := make([]float32, len(input))
	q, k, v := make([]float32, len(input)), make([]float32, len(input)), make([]float32, len(input))
	for row := 0; row < rows; row++ {
		copy(norm[row*h:(row+1)*h], input[row*h:(row+1)*h])
		if !simd.RMSNormTo(norm[row*h:(row+1)*h], l.inputNorm, cfg.RMSNormEps) {
			return nil, fmt.Errorf("input RMSNorm failed")
		}
		if err := l.q.forward(q[row*h:(row+1)*h], norm[row*h:(row+1)*h]); err != nil {
			return nil, err
		}
		if err := l.k.forward(k[row*h:(row+1)*h], norm[row*h:(row+1)*h]); err != nil {
			return nil, err
		}
		if err := l.v.forward(v[row*h:(row+1)*h], norm[row*h:(row+1)*h]); err != nil {
			return nil, err
		}
		if !simd.ApplyRoPETo(q[row*h:(row+1)*h], rope, row, cfg.Heads, cfg.HeadDim) || !simd.ApplyRoPETo(k[row*h:(row+1)*h], rope, row, cfg.Heads, cfg.HeadDim) {
			return nil, fmt.Errorf("RoPE failed")
		}
	}
	attention := make([]float32, len(input))
	scores := make([]float32, rows)
	scale := float32(1 / math.Sqrt(float64(cfg.HeadDim)))
	for row := 0; row < rows; row++ {
		for head := 0; head < cfg.Heads; head++ {
			qh := q[row*h+head*cfg.HeadDim : row*h+(head+1)*cfg.HeadDim]
			for key := 0; key <= row; key++ {
				scores[key] = simd.Sdot(qh, k[key*h+head*cfg.HeadDim:key*h+(head+1)*cfg.HeadDim]) * scale
			}
			if !simd.SoftmaxInPlace(scores[:row+1]) {
				return nil, fmt.Errorf("attention softmax failed")
			}
			dst := attention[row*h+head*cfg.HeadDim : row*h+(head+1)*cfg.HeadDim]
			for key := 0; key <= row; key++ {
				simd.VecScaleAdd(dst, dst, v[key*h+head*cfg.HeadDim:key*h+(head+1)*cfg.HeadDim], scores[key])
			}
		}
	}
	residual := make([]float32, len(input))
	attnOut := make([]float32, h)
	for row := 0; row < rows; row++ {
		if err := l.out.forward(attnOut, attention[row*h:(row+1)*h]); err != nil {
			return nil, err
		}
		for i := 0; i < h; i++ {
			residual[row*h+i] = input[row*h+i] + attnOut[i]*l.attentionScale[i]
		}
	}
	mlpNorm := make([]float32, len(input))
	gate := make([]float32, cfg.IntermediateSize)
	up := make([]float32, cfg.IntermediateSize)
	down := make([]float32, h)
	for row := 0; row < rows; row++ {
		copy(mlpNorm[row*h:(row+1)*h], residual[row*h:(row+1)*h])
		if !simd.RMSNormTo(mlpNorm[row*h:(row+1)*h], l.postNorm, cfg.RMSNormEps) {
			return nil, fmt.Errorf("post RMSNorm failed")
		}
		if err := l.gate.forward(gate, mlpNorm[row*h:(row+1)*h]); err != nil {
			return nil, err
		}
		if err := l.up.forward(up, mlpNorm[row*h:(row+1)*h]); err != nil {
			return nil, err
		}
		if !simd.SiLUMulTo(gate, gate, up) {
			return nil, fmt.Errorf("SwiGLU failed")
		}
		if err := l.down.forward(down, gate); err != nil {
			return nil, err
		}
		for i := 0; i < h; i++ {
			residual[row*h+i] += down[i] * l.mlpScale[i]
		}
	}
	return residual, nil
}

func (c decoderConv1D) forward(input []float32, length int) ([]float32, int, error) {
	groups := c.groups()
	if length <= 0 || len(input) != c.inChannels*groups*length {
		return nil, 0, fmt.Errorf("invalid causal conv input=%d length=%d", len(input), length)
	}
	actualIn := c.inChannels * groups
	out := make([]float32, c.outChannels*length)
	for oc := 0; oc < c.outChannels; oc++ {
		group := oc * groups / c.outChannels
		for t := 0; t < length; t++ {
			sum := c.bias[oc]
			for icg := 0; icg < c.inChannels; icg++ {
				ic := group*c.inChannels + icg
				for tap := 0; tap < c.k; tap++ {
					src := t - c.dilation*(c.k-1-tap)
					if src >= 0 {
						sum += input[ic*length+src] * c.weight[(oc*c.inChannels+icg)*c.k+tap]
					}
				}
			}
			out[oc*length+t] = sum
		}
	}
	_ = actualIn
	return out, length, nil
}

func (c decoderConv1D) groups() int {
	if c.inChannels == 1 && c.outChannels > 1 {
		return c.outChannels
	}
	return 1
}

func (c decoderTransConv1D) forward(input []float32, length int) ([]float32, int, error) {
	if length <= 0 || len(input) != c.inChannels*length {
		return nil, 0, fmt.Errorf("invalid transposed conv input=%d", len(input))
	}
	rawLen := (length-1)*c.stride + c.k
	raw := make([]float32, c.outChannels*rawLen)
	for oc := 0; oc < c.outChannels; oc++ {
		for t := 0; t < rawLen; t++ {
			raw[oc*rawLen+t] = c.bias[oc]
		}
	}
	for ic := 0; ic < c.inChannels; ic++ {
		for t := 0; t < length; t++ {
			x := input[ic*length+t]
			for oc := 0; oc < c.outChannels; oc++ {
				w := c.weight[(ic*c.outChannels+oc)*c.k : (ic*c.outChannels+oc+1)*c.k]
				for tap := 0; tap < c.k; tap++ {
					raw[oc*rawLen+t*c.stride+tap] += x * w[tap]
				}
			}
		}
	}
	outLen := length * c.stride
	out := make([]float32, c.outChannels*outLen)
	for oc := 0; oc < c.outChannels; oc++ {
		copy(out[oc*outLen:(oc+1)*outLen], raw[oc*rawLen:oc*rawLen+outLen])
	}
	return out, outLen, nil
}

func (b decoderConvNeXt) forward(input []float32, length int) ([]float32, error) {
	hidden, _, err := b.depthwise.forward(input, length)
	if err != nil {
		return nil, err
	}
	channels := len(b.normWeight)
	tm := channelToTime(hidden, channels, length)
	norm := make([]float32, len(tm))
	if !simd.LayerNormLastAxisTo(norm, tm, length, channels, b.normWeight, b.normBias, 1e-6) {
		return nil, fmt.Errorf("ConvNeXt LayerNorm failed")
	}
	wide := make([]float32, b.fc1.outDim)
	rowOut := make([]float32, channels)
	out := append([]float32(nil), input...)
	for row := 0; row < length; row++ {
		if err := b.fc1.forward(wide, norm[row*channels:(row+1)*channels]); err != nil {
			return nil, err
		}
		simd.GELUExact(wide, wide)
		if err := b.fc2.forward(rowOut, wide); err != nil {
			return nil, err
		}
		for ch := 0; ch < channels; ch++ {
			out[ch*length+row] += rowOut[ch] * b.gamma[ch]
		}
	}
	return out, nil
}

func (s decoderUpsampleStage) forward(input []float32, length int) ([]float32, int, error) {
	out, outLen, err := s.trans.forward(input, length)
	if err != nil {
		return nil, 0, err
	}
	out, err = s.block.forward(out, outLen)
	return out, outLen, err
}

func (s snakeBeta) forwardInPlace(x []float32, length int) error {
	channels := len(s.alpha)
	if channels == 0 || len(s.beta) != channels || len(x) != channels*length {
		return fmt.Errorf("invalid SnakeBeta buffers")
	}
	for ch := 0; ch < channels; ch++ {
		a, b := float32(math.Exp(float64(s.alpha[ch]))), float32(math.Exp(float64(s.beta[ch])))+1e-9
		for t := 0; t < length; t++ {
			i := ch*length + t
			y := float32(math.Sin(float64(a * x[i])))
			x[i] += y * y / b
		}
	}
	return nil
}

func (r decoderResidual) forward(input []float32, length int) ([]float32, error) {
	hidden := append([]float32(nil), input...)
	if err := r.act1.forwardInPlace(hidden, length); err != nil {
		return nil, err
	}
	hidden, _, err := r.conv1.forward(hidden, length)
	if err != nil {
		return nil, err
	}
	if err := r.act2.forwardInPlace(hidden, length); err != nil {
		return nil, err
	}
	hidden, _, err = r.conv2.forward(hidden, length)
	if err != nil {
		return nil, err
	}
	for i := range hidden {
		hidden[i] += input[i]
	}
	return hidden, nil
}

func (b decoderBlockCPU) forward(input []float32, length int) ([]float32, int, error) {
	hidden := append([]float32(nil), input...)
	if err := b.act.forwardInPlace(hidden, length); err != nil {
		return nil, 0, err
	}
	hidden, length, err := b.up.forward(hidden, length)
	if err != nil {
		return nil, 0, err
	}
	for i := range b.res {
		hidden, err = b.res[i].forward(hidden, length)
		if err != nil {
			return nil, 0, err
		}
	}
	return hidden, length, nil
}

func channelToTime(x []float32, channels, length int) []float32 {
	out := make([]float32, len(x))
	for ch := 0; ch < channels; ch++ {
		for t := 0; t < length; t++ {
			out[t*channels+ch] = x[ch*length+t]
		}
	}
	return out
}
func timeToChannel(x []float32, length, channels int) []float32 {
	out := make([]float32, len(x))
	for t := 0; t < length; t++ {
		for ch := 0; ch < channels; ch++ {
			out[ch*length+t] = x[t*channels+ch]
		}
	}
	return out
}

var _ Decoder12HzRuntime = (*Decoder12HzCPU)(nil)
