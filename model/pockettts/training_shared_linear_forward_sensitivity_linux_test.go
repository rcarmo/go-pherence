//go:build linux

package pockettts

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

// Test-only serial, separately rounded F32 products/sums from shipped linear
// forward GLSL. No change to dispatch flags or production arithmetic.
func sharedSerialLinearForward(l LinearF32, x []float32, rows int) []float32 {
	out := make([]float32, rows*l.Out)
	for row := 0; row < rows; row++ {
		for col := 0; col < l.Out; col++ {
			var sum float32
			for q := 0; q < l.In; q++ {
				sum += float32(x[row*l.In+q] * l.Weight[col*l.In+q])
			}
			if l.Bias != nil {
				sum += l.Bias[col]
			}
			out[row*l.Out+col] = sum
		}
	}
	return out
}

type sharedLinearForwardSensitivityBackend struct {
	shaderGelu    bool // explicit second intervention; backward remains CPU
	shaderNorm    bool // F32 sqrt then F32 reciprocal, forward only
	modified      int
	maxDifference float64
	layers        []sharedAttentionLayerObservation
}

// Match the forward stats rounding boundary, leaving CPU backward deliberately
// unchanged for sensitivity isolation. Host sqrt is not native qualification.
func sharedRoundedNormForward(rows, width int, input, weight, bias []float32, enabled bool) (transformerNormTape, []float32) {
	if !enabled {
		return transformerNormForward(rows, width, input, weight, bias, 1e-5)
	}
	tape := transformerNormTape{input: append([]float32(nil), input...), normalized: make([]float32, len(input)), inv: make([]float32, rows)}
	for row := 0; row < rows; row++ {
		var mean, variance float32
		for _, x := range input[row*width : (row+1)*width] {
			mean += x
		}
		mean /= float32(width)
		for _, x := range input[row*width : (row+1)*width] {
			d := x - mean
			variance += float32(d * d)
		}
		variance /= float32(width)
		sqrt := float32(math.Sqrt(float64(variance + float32(1e-5))))
		inv := float32(1) / sqrt
		tape.inv[row] = inv
		for col := 0; col < width; col++ {
			at := row*width + col
			tape.normalized[at] = float32((input[at]-mean)*inv)*weight[col] + bias[col]
		}
	}
	return tape, tape.normalized
}

func sharedSerialTransformerLayer(l TransformerLayerCPU, rows, width, heads, dim, contextRows int, period float64, input []float32, shaderGelu, shaderNorm bool) transformerLayerTape {
	var tape transformerLayerTape
	tape.norm1, _ = sharedRoundedNormForward(rows, width, input, l.Norm1Weight, l.Norm1Bias, shaderNorm)
	qkv := sharedSerialLinearForward(l.InProjection, tape.norm1.normalized, rows)
	tape.q, tape.k, tape.v = make([]float32, rows*width), make([]float32, rows*width), make([]float32, rows*width)
	for row := 0; row < rows; row++ {
		copy(tape.q[row*width:(row+1)*width], qkv[row*3*width:row*3*width+width])
		copy(tape.k[row*width:(row+1)*width], qkv[row*3*width+width:row*3*width+2*width])
		copy(tape.v[row*width:(row+1)*width], qkv[row*3*width+2*width:(row+1)*3*width])
		ropeScalar(tape.q[row*width:(row+1)*width], row, heads, dim, period, false)
		ropeScalar(tape.k[row*width:(row+1)*width], row, heads, dim, period, false)
	}
	_, tape.probabilities, tape.attention = sharedAttentionForwardDiagnostic(tape.q, tape.k, tape.v, rows, heads, dim, contextRows)
	tape.projected = sharedSerialLinearForward(l.OutProjection, tape.attention, rows)
	tape.output = make([]float32, rows*width)
	for i := range tape.output {
		v := tape.projected[i]
		if l.LayerScale1 != nil {
			v *= l.LayerScale1[i%width]
		}
		tape.output[i] = input[i] + v
	}
	tape.norm2, _ = sharedRoundedNormForward(rows, width, tape.output, l.Norm2Weight, l.Norm2Bias, shaderNorm)
	tape.fc1Pre = sharedSerialLinearForward(l.FC1, tape.norm2.normalized, rows)
	tape.fc1 = make([]float32, len(tape.fc1Pre))
	for i, x := range tape.fc1Pre {
		if shaderGelu {
			tape.fc1[i] = sharedShaderGeluDiagnostic(x)
		} else {
			tape.fc1[i] = geluTanh(x)
		}
	}
	tape.down = sharedSerialLinearForward(l.FC2, tape.fc1, rows)
	for i := range tape.output {
		v := tape.down[i]
		if l.LayerScale2 != nil {
			v *= l.LayerScale2[i%width]
		}
		tape.output[i] += v
	}
	return tape
}

func (b *sharedLinearForwardSensitivityBackend) EvaluateTransformer(m *TransformerCPU, sequence, seed []float32, rows int) (TransformerTrainingResult, error) {
	if b == nil {
		return TransformerTrainingResult{}, fmt.Errorf("nil serial forward sensitivity")
	}
	if err := validateTrainableTransformer(m, sequence, seed, rows); err != nil {
		return TransformerTrainingResult{}, err
	}
	baseline, baselineOutput := m.forwardTraining(rows, sequence)
	tape := &transformerTape{layers: make([]transformerLayerTape, len(m.Layers))}
	hidden := sequence
	observations := make([]sharedAttentionLayerObservation, len(m.Layers))
	for i, l := range m.Layers {
		tape.layers[i] = sharedSerialTransformerLayer(l, rows, m.Width, m.Heads, m.HeadDim, m.Context, m.MaxPeriod, hidden, b.shaderGelu, b.shaderNorm)
		hidden = tape.layers[i].output
		o := &observations[i]
		o.Layer = i
		old, now := baseline.layers[i], tape.layers[i]
		for j := range now.q {
			o.QKVProjection.add(old.q[j], now.q[j])
			o.QKVProjection.add(old.k[j], now.k[j])
			o.QKVProjection.add(old.v[j], now.v[j])
			o.Output.add(old.output[j], now.output[j])
			o.OutProjection.add(old.projected[j], now.projected[j])
			o.FC2Projection.add(old.down[j], now.down[j])
		}
		for j := range now.fc1Pre {
			o.FC1Projection.add(old.fc1Pre[j], now.fc1Pre[j])
		}
		for j := range now.probabilities {
			o.Probability.add(old.probabilities[j], now.probabilities[j])
		}
	}
	if m.FinalWeight != nil {
		tape.final, hidden = sharedRoundedNormForward(rows, m.Width, hidden, m.FinalWeight, m.FinalBias, b.shaderNorm)
	}
	for i := range hidden {
		d := math.Abs(float64(hidden[i]) - float64(baselineOutput[i]))
		if d > 0 {
			b.modified++
		}
		b.maxDifference = max(b.maxDifference, d)
	}
	b.layers = observations
	g := newTransformerGradients(m)
	dx, err := m.backwardTrainingWithAttention(rows, tape, seed, g, nil)
	if err != nil {
		return TransformerTrainingResult{}, err
	}
	all := map[string][]float32{}
	addFlowLMGradients(all, &FlowLMTrainingGradients{Transformer: g})
	for name := range all {
		if !strings.HasPrefix(name, "flow_lm.transformer.layers.") && !strings.HasPrefix(name, "flow_lm.out_norm.") {
			delete(all, name)
		}
	}
	return TransformerTrainingResult{Output: hidden, DSequence: dx, Parameters: all}, nil
}

func TestPocketSharedSerialLinearForwardSensitivity(t *testing.T) {
	l := LinearF32{In: 2, Out: 2, Weight: []float32{.25, -.5, .75, .125}, Bias: []float32{.1, -.2}}
	got := sharedSerialLinearForward(l, []float32{.5, -.25, .125, .75}, 2)
	assertSliceClose(t, "independent dyadic affine", got, []float32{.35, .14375, -.24375, -.0125}, 2e-7)
	m := tinyTrainableTransformer()
	rows := 4
	x, seed := make([]float32, rows*m.Width), make([]float32, rows*m.Width)
	for i := range x {
		x[i] = float32(i%7-3) * .1
		seed[i] = float32(i%3-1) * .03
	}
	for _, contextRows := range []int{0, 1, 2, 8} {
		m.Context = contextRows
		b := &sharedLinearForwardSensitivityBackend{}
		r, err := b.EvaluateTransformer(m, x, seed, rows)
		if err != nil {
			t.Fatal(err)
		}
		if err := validateTransformerBackendResult(m, r, x, seed, rows); err != nil {
			t.Fatal(err)
		}
		want, g, dx, err := m.ForwardBackward(x, seed, rows)
		if err != nil {
			t.Fatal(err)
		}
		assertSliceClose(t, "serial transformer forward", r.Output, want, 3e-5)
		assertSliceClose(t, "serial transformer dx", r.DSequence, dx, 3e-5)
		grads := map[string][]float32{}
		addFlowLMGradients(grads, &FlowLMTrainingGradients{Transformer: g})
		for name, got := range r.Parameters {
			assertSliceClose(t, name, got, grads[name], 3e-5)
		}
		if len(b.layers) != len(m.Layers) {
			t.Fatal("layer observations")
		}
	}
	b := &sharedLinearForwardSensitivityBackend{shaderGelu: true}
	r, err := b.EvaluateTransformer(m, x, seed, rows)
	if err != nil {
		t.Fatal(err)
	}
	want, _, _, err := m.ForwardBackward(x, seed, rows)
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, "serial/GELU transformer forward", r.Output, want, 3e-5)
	if err := validateTransformerBackendResult(m, r, x, seed, rows); err != nil {
		t.Fatal(err)
	}
	if _, err := (*sharedLinearForwardSensitivityBackend)(nil).EvaluateTransformer(m, x, seed, rows); err == nil {
		t.Fatal("nil backend")
	}
	if _, err := (&sharedLinearForwardSensitivityBackend{}).EvaluateTransformer(nil, nil, nil, 0); err == nil {
		t.Fatal("invalid geometry")
	}
}
