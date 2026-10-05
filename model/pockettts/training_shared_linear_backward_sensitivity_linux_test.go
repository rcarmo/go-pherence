//go:build linux

package pockettts

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

type sharedLinearAdjointObservation struct {
	Layer                                  int
	FC2, FC1, OutProjection, QKVProjection sharedAttentionError
	GeluAdjoint                            sharedAttentionError
}

type sharedLinearBackwardSensitivityBackend struct {
	serialForward    bool
	shaderDerivative bool
	observations     []sharedLinearAdjointObservation
}

// Preserve the normal CPU dWeight/dBias reduction. Only the returned dInput
// uses the shipped linear backward mode-zero serial F32 reduction formula.
func sharedSerialLinearInputAdjoint(l LinearF32, input, seed []float32, rows int, g *LinearF32Gradient, transpose []float32, e *sharedAttentionError) []float32 {
	cpu := linearBackwardRowsTrainingScratch(l, input, seed, rows, g, transpose)
	out := make([]float32, rows*l.In)
	for row := 0; row < rows; row++ {
		for col := 0; col < l.In; col++ {
			var sum float32
			for q := 0; q < l.Out; q++ {
				sum += float32(seed[row*l.Out+q] * l.Weight[q*l.In+col])
			}
			out[row*l.In+col] = sum
			if e != nil {
				e.add(cpu[row*l.In+col], sum)
			}
		}
	}
	return out
}

// Independent F32 expression from shipped GELU backward GLSL. Host exp
// remains a limitation; this is not a device implementation or fallback.
func sharedShaderGeluDerivative(x float32) float32 {
	u := float32(.7978845608028654) * (x + ((float32(.044715)*x)*x)*x)
	e := float32(math.Exp(float64(-2 * float32(math.Abs(float64(u))))))
	half := float32(1) / (1 + e)
	if u < 0 {
		half = e / (1 + e)
	}
	sech := (4 * e) / ((1 + e) * (1 + e))
	du := float32(.7978845608028654) * (1 + (float32(.134145)*x)*x)
	return half + ((.5*x)*sech)*du
}

func sharedSerialLayerInputAdjoint(l TransformerLayerCPU, rows, width, heads, dim, contextRows int, period float64, tape transformerLayerTape, seed []float32, g *TransformerLayerGradient, o *sharedLinearAdjointObservation, shaderDerivative bool) []float32 {
	var transpose []float32
	if rows >= 16 {
		transpose = make([]float32, rows*max(width, l.FC1.Out, l.InProjection.Out))
	}
	dAfter := append([]float32(nil), seed...)
	dDown := make([]float32, len(seed))
	for i, v := range seed {
		if l.LayerScale2 != nil {
			g.LayerScale2[i%width] += v * tape.down[i]
			v *= l.LayerScale2[i%width]
		}
		dDown[i] = v
	}
	dFC1 := sharedSerialLinearInputAdjoint(l.FC2, tape.fc1, dDown, rows, &g.FC2, transpose, &o.FC2)
	for i := range dFC1 {
		cpu := dFC1[i] * geluTanhDerivative(tape.fc1Pre[i])
		if shaderDerivative {
			observed := dFC1[i] * sharedShaderGeluDerivative(tape.fc1Pre[i])
			o.GeluAdjoint.add(cpu, observed)
			dFC1[i] = observed
		} else {
			dFC1[i] = cpu
		}
	}
	dNorm2 := sharedSerialLinearInputAdjoint(l.FC1, tape.norm2.normalized, dFC1, rows, &g.FC1, transpose, &o.FC1)
	addInPlace(dAfter, transformerNormBackward(rows, width, tape.norm2, dNorm2, l.Norm2Weight, g.Norm2Weight, g.Norm2Bias))
	dProjected := make([]float32, len(seed))
	for i, v := range dAfter {
		if l.LayerScale1 != nil {
			g.LayerScale1[i%width] += v * tape.projected[i]
			v *= l.LayerScale1[i%width]
		}
		dProjected[i] = v
	}
	dy := sharedSerialLinearInputAdjoint(l.OutProjection, tape.attention, dProjected, rows, &g.OutProjection, transpose, &o.OutProjection)
	_, _, _, dq, dk, dv := sharedAttentionBackwardDiagnostic(tape.q, tape.k, tape.v, tape.probabilities, dy, rows, heads, dim, contextRows)
	for row := 0; row < rows; row++ {
		ropeScalar(dq[row*width:(row+1)*width], row, heads, dim, period, true)
		ropeScalar(dk[row*width:(row+1)*width], row, heads, dim, period, true)
	}
	packed := make([]float32, rows*3*width)
	for row := 0; row < rows; row++ {
		copy(packed[row*3*width:row*3*width+width], dq[row*width:(row+1)*width])
		copy(packed[row*3*width+width:row*3*width+2*width], dk[row*width:(row+1)*width])
		copy(packed[row*3*width+2*width:(row+1)*3*width], dv[row*width:(row+1)*width])
	}
	dNorm1 := sharedSerialLinearInputAdjoint(l.InProjection, tape.norm1.normalized, packed, rows, &g.InProjection, transpose, &o.QKVProjection)
	addInPlace(dAfter, transformerNormBackward(rows, width, tape.norm1, dNorm1, l.Norm1Weight, g.Norm1Weight, g.Norm1Bias))
	return dAfter
}

func (b *sharedLinearBackwardSensitivityBackend) EvaluateTransformer(m *TransformerCPU, x, seed []float32, rows int) (TransformerTrainingResult, error) {
	if b == nil {
		return TransformerTrainingResult{}, fmt.Errorf("nil serial adjoint backend")
	}
	if err := validateTrainableTransformer(m, x, seed, rows); err != nil {
		return TransformerTrainingResult{}, err
	}
	var tape *transformerTape
	var output []float32
	if b.serialForward {
		tape = &transformerTape{layers: make([]transformerLayerTape, len(m.Layers))}
		output = x
		for i, l := range m.Layers {
			tape.layers[i] = sharedSerialTransformerLayer(l, rows, m.Width, m.Heads, m.HeadDim, m.Context, m.MaxPeriod, output, true, false)
			output = tape.layers[i].output
		}
		if m.FinalWeight != nil {
			tape.final, output = transformerNormForward(rows, m.Width, output, m.FinalWeight, m.FinalBias, 1e-5)
		}
	} else {
		tape, output = m.forwardTraining(rows, x)
	}
	g := newTransformerGradients(m)
	dx := seed
	if m.FinalWeight != nil {
		dx = transformerNormBackward(rows, m.Width, tape.final, dx, m.FinalWeight, g.FinalWeight, g.FinalBias)
	}
	obs := make([]sharedLinearAdjointObservation, len(m.Layers))
	for i := len(m.Layers) - 1; i >= 0; i-- {
		obs[i].Layer = i
		dx = sharedSerialLayerInputAdjoint(m.Layers[i], rows, m.Width, m.Heads, m.HeadDim, m.Context, m.MaxPeriod, tape.layers[i], dx, &g.Layers[i], &obs[i], b.shaderDerivative)
	}
	if !zeroTrainingSeed(seed) {
		b.observations = obs
	}
	all := map[string][]float32{}
	addFlowLMGradients(all, &FlowLMTrainingGradients{Transformer: g})
	for name := range all {
		if !strings.HasPrefix(name, "flow_lm.transformer.layers.") && !strings.HasPrefix(name, "flow_lm.out_norm.") {
			delete(all, name)
		}
	}
	return TransformerTrainingResult{Output: output, DSequence: dx, Parameters: all}, nil
}

func TestPocketSharedShaderGeluDerivative(t *testing.T) {
	for _, x := range []float32{-12, -8, -5, -2, -.1, 0, .1, 2, 5, 8, 12} {
		got := sharedShaderGeluDerivative(x)
		if !isFinite(got) || math.Abs(float64(got)-float64(geluTanhDerivative(x))) > 1e-6 {
			t.Fatal("independent GELU derivative", x, got, geluTanhDerivative(x))
		}
	}
	m := tinyTrainableTransformer()
	rows := 4
	x, seed := make([]float32, rows*m.Width), make([]float32, rows*m.Width)
	for i := range x {
		x[i] = float32(i%7-3) * .1
		seed[i] = float32(i%3-1) * .03
	}
	b := &sharedLinearBackwardSensitivityBackend{serialForward: true, shaderDerivative: true}
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
	assertSliceClose(t, "GELU formula forward", r.Output, want, 3e-5)
	assertSliceClose(t, "GELU formula adjoints", r.DSequence, dx, 3e-5)
	grads := map[string][]float32{}
	addFlowLMGradients(grads, &FlowLMTrainingGradients{Transformer: g})
	for name, got := range r.Parameters {
		assertSliceClose(t, name, got, grads[name], 3e-5)
	}
	if len(b.observations) != len(m.Layers) || b.observations[0].GeluAdjoint.Values != rows*m.Layers[0].FC1.Out {
		t.Fatal("GELU adjoint observation count")
	}
}

func TestPocketSharedSerialLinearInputAdjoint(t *testing.T) {
	l := LinearF32{In: 2, Out: 3, Weight: []float32{.25, -.5, .75, .125, -.25, .5}}
	x, seed := []float32{.5, -.25, .125, .75}, []float32{.25, -.5, .75, -.25, .5, -.75}
	g := newLinearGradient(l)
	var obs sharedAttentionError
	got := sharedSerialLinearInputAdjoint(l, x, seed, 2, &g, nil, &obs)
	for row := 0; row < 2; row++ {
		for in := 0; in < 2; in++ {
			var want float64
			for out := 0; out < 3; out++ {
				want += float64(seed[row*3+out]) * float64(l.Weight[out*2+in])
			}
			if math.Abs(float64(got[row*2+in])-want) > 2e-7 {
				t.Fatal("independent input adjoint", got, want)
			}
		}
	}
	if obs.Values != 4 {
		t.Fatal("adjoint observations")
	}
	m := tinyTrainableTransformer()
	rows := 4
	x, seed = make([]float32, rows*m.Width), make([]float32, rows*m.Width)
	for i := range x {
		x[i] = float32(i%7-3) * .1
		seed[i] = float32(i%3-1) * .03
	}
	for _, combined := range []bool{false, true} {
		for _, contextRows := range []int{0, 1, 2, 8} {
			m.Context = contextRows
			b := &sharedLinearBackwardSensitivityBackend{serialForward: combined}
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
			assertSliceClose(t, "serial adjoint forward", r.Output, want, 3e-5)
			assertSliceClose(t, "serial adjoint dx", r.DSequence, dx, 3e-5)
			grads := map[string][]float32{}
			addFlowLMGradients(grads, &FlowLMTrainingGradients{Transformer: g})
			for name, got := range r.Parameters {
				assertSliceClose(t, name, got, grads[name], 3e-5)
			}
			if len(b.observations) != len(m.Layers) {
				t.Fatal("layer observations")
			}
		}
	}
	if _, err := (*sharedLinearBackwardSensitivityBackend)(nil).EvaluateTransformer(m, x, seed, rows); err == nil {
		t.Fatal("nil backend")
	}
	if _, err := (&sharedLinearBackwardSensitivityBackend{}).EvaluateTransformer(nil, nil, nil, 0); err == nil {
		t.Fatal("invalid geometry")
	}
}
