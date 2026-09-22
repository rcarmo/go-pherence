package pockettts

import (
	"fmt"
	"math"

	"github.com/rcarmo/go-pherence/internal/checked"
)

type TransformerLayerGradient struct {
	Norm1Weight, Norm1Bias      []float32
	Norm2Weight, Norm2Bias      []float32
	InProjection, OutProjection LinearF32Gradient
	FC1, FC2                    LinearF32Gradient
	LayerScale1, LayerScale2    []float32
}

type TransformerGradients struct {
	Layers                 []TransformerLayerGradient
	FinalWeight, FinalBias []float32
}

type transformerNormTape struct {
	input, normalized []float32
	inv               []float32
}

type transformerLayerTape struct {
	input, q, k, v, attention, projected, afterAttention []float32
	probabilities                                        []float32
	norm1, norm2                                         transformerNormTape
	fc1Pre, fc1, down, output                            []float32
}

type transformerTape struct {
	layers []transformerLayerTape
	final  transformerNormTape
}

// ForwardBackward evaluates the stateless causal transformer and returns exact
// F32 parameter and input gradients. BF16 inference storage is rejected.
func (m *TransformerCPU) ForwardBackward(sequence, dOutput []float32, rows int) (output []float32, gradients *TransformerGradients, dSequence []float32, err error) {
	if err = validateTrainableTransformer(m, sequence, dOutput, rows); err != nil {
		return nil, nil, nil, err
	}
	tape, output := m.forwardTraining(rows, sequence)
	gradients = newTransformerGradients(m)
	dSequence = m.backwardTraining(rows, tape, dOutput, gradients)
	return output, gradients, dSequence, nil
}

func validateTrainableTransformer(m *TransformerCPU, sequence, dOutput []float32, rows int) error {
	if m == nil || rows <= 0 || m.Width <= 0 || m.Heads <= 0 || m.HeadDim <= 0 || m.Width != m.Heads*m.HeadDim || m.HeadDim%2 != 0 || len(m.Layers) == 0 || len(dOutput) != len(sequence) || m.MaxPeriod <= 0 {
		return fmt.Errorf("invalid Pocket TTS trainable transformer input")
	}
	elements, ok := checked.MulInt(rows, m.Width)
	if !ok || len(sequence) != elements {
		return fmt.Errorf("invalid Pocket TTS trainable transformer input")
	}
	if _, ok = checked.MulInt(rows, rows); !ok {
		return fmt.Errorf("invalid Pocket TTS trainable transformer attention shape")
	}
	if m.Context < 0 || (m.FinalWeight != nil && (len(m.FinalWeight) != m.Width || len(m.FinalBias) != m.Width)) {
		return fmt.Errorf("invalid Pocket TTS trainable transformer topology")
	}
	for _, layer := range m.Layers {
		linears := []LinearF32{layer.InProjection, layer.OutProjection, layer.FC1, layer.FC2}
		if len(layer.Norm1Weight) != m.Width || len(layer.Norm1Bias) != m.Width || len(layer.Norm2Weight) != m.Width || len(layer.Norm2Bias) != m.Width || layer.InProjection.In != m.Width || layer.InProjection.Out != 3*m.Width || layer.OutProjection.In != m.Width || layer.OutProjection.Out != m.Width || layer.FC1.In != m.Width || layer.FC2.Out != m.Width || layer.FC2.In != layer.FC1.Out || (layer.LayerScale1 != nil && len(layer.LayerScale1) != m.Width) || (layer.LayerScale2 != nil && len(layer.LayerScale2) != m.Width) {
			return fmt.Errorf("invalid Pocket TTS trainable transformer layer")
		}
		for _, linear := range linears {
			if len(linear.WeightBF16) != 0 || len(linear.Weight) != linear.In*linear.Out || linear.Bias != nil {
				return fmt.Errorf("Pocket TTS transformer training requires bias-free owned F32 linears")
			}
		}
	}
	for _, values := range [][]float32{sequence, dOutput} {
		for _, value := range values {
			if !isFinite(value) {
				return fmt.Errorf("Pocket TTS trainable transformer input is non-finite")
			}
		}
	}
	return nil
}

func newTransformerGradients(m *TransformerCPU) *TransformerGradients {
	g := &TransformerGradients{Layers: make([]TransformerLayerGradient, len(m.Layers))}
	for i, layer := range m.Layers {
		g.Layers[i] = TransformerLayerGradient{
			Norm1Weight: make([]float32, m.Width), Norm1Bias: make([]float32, m.Width),
			Norm2Weight: make([]float32, m.Width), Norm2Bias: make([]float32, m.Width),
			InProjection: newLinearGradient(layer.InProjection), OutProjection: newLinearGradient(layer.OutProjection),
			FC1: newLinearGradient(layer.FC1), FC2: newLinearGradient(layer.FC2),
		}
		if layer.LayerScale1 != nil {
			g.Layers[i].LayerScale1 = make([]float32, m.Width)
		}
		if layer.LayerScale2 != nil {
			g.Layers[i].LayerScale2 = make([]float32, m.Width)
		}
	}
	if m.FinalWeight != nil {
		g.FinalWeight, g.FinalBias = make([]float32, m.Width), make([]float32, m.Width)
	}
	return g
}

func (m *TransformerCPU) forwardTraining(rows int, sequence []float32) (*transformerTape, []float32) {
	tape := &transformerTape{layers: make([]transformerLayerTape, len(m.Layers))}
	hidden := append([]float32(nil), sequence...)
	for i := range m.Layers {
		tape.layers[i], hidden = m.Layers[i].forwardTraining(rows, m.Width, m.Heads, m.HeadDim, m.Context, m.MaxPeriod, hidden)
	}
	if m.FinalWeight != nil {
		tape.final, hidden = transformerNormForward(rows, m.Width, hidden, m.FinalWeight, m.FinalBias, 1e-5)
	}
	return tape, hidden
}

func (l TransformerLayerCPU) forwardTraining(rows, width, heads, headDim, context int, maxPeriod float64, input []float32) (transformerLayerTape, []float32) {
	tape := transformerLayerTape{input: append([]float32(nil), input...)}
	tape.norm1, _ = transformerNormForward(rows, width, input, l.Norm1Weight, l.Norm1Bias, 1e-5)
	tape.q, tape.k, tape.v = make([]float32, rows*width), make([]float32, rows*width), make([]float32, rows*width)
	for row := 0; row < rows; row++ {
		qkv := linearForwardTraining(l.InProjection, tape.norm1.normalized[row*width:(row+1)*width])
		copy(tape.q[row*width:(row+1)*width], qkv[:width])
		copy(tape.k[row*width:(row+1)*width], qkv[width:2*width])
		copy(tape.v[row*width:(row+1)*width], qkv[2*width:])
		ropeScalar(tape.q[row*width:(row+1)*width], row, heads, headDim, maxPeriod, false)
		ropeScalar(tape.k[row*width:(row+1)*width], row, heads, headDim, maxPeriod, false)
	}
	tape.attention = make([]float32, rows*width)
	tape.probabilities = make([]float32, heads*rows*rows)
	scale := float32(1 / math.Sqrt(float64(headDim)))
	for query := 0; query < rows; query++ {
		start := 0
		if context > 0 && query-context+1 > 0 {
			start = query - context + 1
		}
		for head := 0; head < heads; head++ {
			base := (head*rows + query) * rows
			maxScore := float32(-math.MaxFloat32)
			for key := start; key <= query; key++ {
				score := dotScalar(tape.q[query*width+head*headDim:query*width+(head+1)*headDim], tape.k[key*width+head*headDim:key*width+(head+1)*headDim]) * scale
				tape.probabilities[base+key] = score
				if score > maxScore {
					maxScore = score
				}
			}
			sum := float32(0)
			for key := start; key <= query; key++ {
				value := float32(math.Exp(float64(tape.probabilities[base+key] - maxScore)))
				tape.probabilities[base+key] = value
				sum += value
			}
			for key := start; key <= query; key++ {
				probability := tape.probabilities[base+key] / sum
				tape.probabilities[base+key] = probability
				for d := 0; d < headDim; d++ {
					tape.attention[query*width+head*headDim+d] += probability * tape.v[key*width+head*headDim+d]
				}
			}
		}
	}
	tape.projected = make([]float32, rows*width)
	tape.afterAttention = make([]float32, rows*width)
	for row := 0; row < rows; row++ {
		projected := linearForwardTraining(l.OutProjection, tape.attention[row*width:(row+1)*width])
		copy(tape.projected[row*width:(row+1)*width], projected)
		for i := 0; i < width; i++ {
			update := projected[i]
			if l.LayerScale1 != nil {
				update *= l.LayerScale1[i]
			}
			tape.afterAttention[row*width+i] = input[row*width+i] + update
		}
	}
	tape.norm2, _ = transformerNormForward(rows, width, tape.afterAttention, l.Norm2Weight, l.Norm2Bias, 1e-5)
	tape.fc1Pre = make([]float32, rows*l.FC1.Out)
	tape.fc1 = make([]float32, len(tape.fc1Pre))
	tape.down = make([]float32, rows*width)
	tape.output = append([]float32(nil), tape.afterAttention...)
	for row := 0; row < rows; row++ {
		pre := linearForwardTraining(l.FC1, tape.norm2.normalized[row*width:(row+1)*width])
		copy(tape.fc1Pre[row*l.FC1.Out:(row+1)*l.FC1.Out], pre)
		for i, value := range pre {
			tape.fc1[row*l.FC1.Out+i] = geluTanh(value)
		}
		down := linearForwardTraining(l.FC2, tape.fc1[row*l.FC1.Out:(row+1)*l.FC1.Out])
		copy(tape.down[row*width:(row+1)*width], down)
		for i := 0; i < width; i++ {
			update := down[i]
			if l.LayerScale2 != nil {
				update *= l.LayerScale2[i]
			}
			tape.output[row*width+i] += update
		}
	}
	return tape, append([]float32(nil), tape.output...)
}

func transformerNormForward(rows, width int, input, weight, bias []float32, epsilon float32) (transformerNormTape, []float32) {
	tape := transformerNormTape{input: append([]float32(nil), input...), normalized: make([]float32, len(input)), inv: make([]float32, rows)}
	for row := 0; row < rows; row++ {
		base, _, _, inv := layerNormBase(input[row*width:(row+1)*width], epsilon)
		tape.inv[row] = inv
		for i := 0; i < width; i++ {
			tape.normalized[row*width+i] = base[i]*weight[i] + bias[i]
		}
	}
	return tape, append([]float32(nil), tape.normalized...)
}

func (m *TransformerCPU) backwardTraining(rows int, tape *transformerTape, dOutput []float32, gradients *TransformerGradients) []float32 {
	dHidden := append([]float32(nil), dOutput...)
	if m.FinalWeight != nil {
		dHidden = transformerNormBackward(rows, m.Width, tape.final, dHidden, m.FinalWeight, gradients.FinalWeight, gradients.FinalBias)
	}
	for i := len(m.Layers) - 1; i >= 0; i-- {
		dHidden = m.Layers[i].backwardTraining(rows, m.Width, m.Heads, m.HeadDim, m.Context, m.MaxPeriod, tape.layers[i], dHidden, &gradients.Layers[i])
	}
	return dHidden
}

func (l TransformerLayerCPU) backwardTraining(rows, width, heads, headDim, context int, maxPeriod float64, tape transformerLayerTape, dOutput []float32, gradient *TransformerLayerGradient) []float32 {
	dAfter := append([]float32(nil), dOutput...)
	dDown := make([]float32, rows*width)
	for row := 0; row < rows; row++ {
		for i := 0; i < width; i++ {
			g := dOutput[row*width+i]
			if l.LayerScale2 != nil {
				gradient.LayerScale2[i] += g * tape.down[row*width+i]
				g *= l.LayerScale2[i]
			}
			dDown[row*width+i] = g
		}
	}
	dNorm2 := make([]float32, rows*width)
	for row := 0; row < rows; row++ {
		dFC1 := linearBackwardTraining(l.FC2, tape.fc1[row*l.FC1.Out:(row+1)*l.FC1.Out], dDown[row*width:(row+1)*width], &gradient.FC2)
		for i := range dFC1 {
			dFC1[i] *= geluTanhDerivative(tape.fc1Pre[row*l.FC1.Out+i])
		}
		dNorm := linearBackwardTraining(l.FC1, tape.norm2.normalized[row*width:(row+1)*width], dFC1, &gradient.FC1)
		copy(dNorm2[row*width:(row+1)*width], dNorm)
	}
	addInPlace(dAfter, transformerNormBackward(rows, width, tape.norm2, dNorm2, l.Norm2Weight, gradient.Norm2Weight, gradient.Norm2Bias))
	dInput := append([]float32(nil), dAfter...)
	dProjected := make([]float32, rows*width)
	for row := 0; row < rows; row++ {
		for i := 0; i < width; i++ {
			g := dAfter[row*width+i]
			if l.LayerScale1 != nil {
				gradient.LayerScale1[i] += g * tape.projected[row*width+i]
				g *= l.LayerScale1[i]
			}
			dProjected[row*width+i] = g
		}
	}
	dAttention := make([]float32, rows*width)
	for row := 0; row < rows; row++ {
		d := linearBackwardTraining(l.OutProjection, tape.attention[row*width:(row+1)*width], dProjected[row*width:(row+1)*width], &gradient.OutProjection)
		copy(dAttention[row*width:(row+1)*width], d)
	}
	dQ, dK, dV := make([]float32, rows*width), make([]float32, rows*width), make([]float32, rows*width)
	scale := float32(1 / math.Sqrt(float64(headDim)))
	for query := 0; query < rows; query++ {
		start := 0
		if context > 0 && query-context+1 > 0 {
			start = query - context + 1
		}
		for head := 0; head < heads; head++ {
			base := (head*rows + query) * rows
			dProb := make([]float32, rows)
			weighted := float32(0)
			for key := start; key <= query; key++ {
				for d := 0; d < headDim; d++ {
					dProb[key] += dAttention[query*width+head*headDim+d] * tape.v[key*width+head*headDim+d]
					dV[key*width+head*headDim+d] += tape.probabilities[base+key] * dAttention[query*width+head*headDim+d]
				}
				weighted += dProb[key] * tape.probabilities[base+key]
			}
			for key := start; key <= query; key++ {
				ds := tape.probabilities[base+key] * (dProb[key] - weighted) * scale
				for d := 0; d < headDim; d++ {
					dQ[query*width+head*headDim+d] += ds * tape.k[key*width+head*headDim+d]
					dK[key*width+head*headDim+d] += ds * tape.q[query*width+head*headDim+d]
				}
			}
		}
	}
	for row := 0; row < rows; row++ {
		ropeScalar(dQ[row*width:(row+1)*width], row, heads, headDim, maxPeriod, true)
		ropeScalar(dK[row*width:(row+1)*width], row, heads, headDim, maxPeriod, true)
	}
	dNorm1 := make([]float32, rows*width)
	for row := 0; row < rows; row++ {
		packed := make([]float32, 3*width)
		copy(packed, dQ[row*width:(row+1)*width])
		copy(packed[width:], dK[row*width:(row+1)*width])
		copy(packed[2*width:], dV[row*width:(row+1)*width])
		d := linearBackwardTraining(l.InProjection, tape.norm1.normalized[row*width:(row+1)*width], packed, &gradient.InProjection)
		copy(dNorm1[row*width:(row+1)*width], d)
	}
	addInPlace(dInput, transformerNormBackward(rows, width, tape.norm1, dNorm1, l.Norm1Weight, gradient.Norm1Weight, gradient.Norm1Bias))
	return dInput
}

func transformerNormBackward(rows, width int, tape transformerNormTape, dOutput, weight, dWeight, dBias []float32) []float32 {
	dInput := make([]float32, rows*width)
	for row := 0; row < rows; row++ {
		// Recompute the normalized base from immutable input so zero affine
		// weights do not make the tape ambiguous.
		base, _, _, _ := layerNormBase(tape.input[row*width:(row+1)*width], 1e-5)
		dBase := make([]float32, width)
		for i := 0; i < width; i++ {
			g := dOutput[row*width+i]
			dWeight[i] += g * base[i]
			dBias[i] += g
			dBase[i] = g * weight[i]
		}
		copy(dInput[row*width:(row+1)*width], layerNormBackward(base, tape.inv[row], dBase))
	}
	return dInput
}

func ropeScalar(values []float32, position, heads, headDim int, maxPeriod float64, inverse bool) {
	pairs := headDim / 2
	for head := 0; head < heads; head++ {
		for pair := 0; pair < pairs; pair++ {
			theta := float64(position) * math.Pow(maxPeriod, -float64(pair)/float64(pairs))
			sine, cosine := float32(math.Sin(theta)), float32(math.Cos(theta))
			if inverse {
				sine = -sine
			}
			i := head*headDim + 2*pair
			real, imag := values[i], values[i+1]
			values[i] = real*cosine - imag*sine
			values[i+1] = imag*cosine + real*sine
		}
	}
}
func dotScalar(a, b []float32) float32 {
	sum := float32(0)
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
func geluTanh(x float32) float32 {
	const c = float64(0.7978845608028654)
	xf := float64(x)
	return float32(.5 * xf * (1 + math.Tanh(c*(xf+.044715*xf*xf*xf))))
}
func geluTanhDerivative(x float32) float32 {
	const c = float64(0.7978845608028654)
	xf := float64(x)
	u := c * (xf + .044715*xf*xf*xf)
	th := math.Tanh(u)
	du := c * (1 + 3*.044715*xf*xf)
	return float32(.5*(1+th) + .5*xf*(1-th*th)*du)
}
