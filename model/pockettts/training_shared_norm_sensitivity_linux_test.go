//go:build linux

package pockettts

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

// Evaluate only the ordinary (zero tangent) LayerNorm adjoint from the shipped
// dual-norm GLSL. Host sqrt/division is an arithmetic diagnostic, not device
// emulation. Forward stats stay those of the unchanged CPU tape.
func sharedNormAdjointDiagnostic(input, seed, weight []float32, inv float32) ([]float32, []float64, error) {
	if len(input) < 2 || len(seed) != len(input) || len(weight) != len(input) || !finiteF32(input) || !finiteF32(seed) || !finiteF32(weight) || !isFinite(inv) || inv <= 0 {
		return nil, nil, fmt.Errorf("norm diagnostic input")
	}
	n := float32(len(input))
	mean := float32(0)
	for _, x := range input {
		mean += x
	}
	mean /= n
	ginv := float32(0)
	for i, x := range input {
		ginv += float32((seed[i] * weight[i]) * (x - mean))
	}
	gvar := ((ginv * (-.5)) * inv) * inv * inv
	out := make([]float32, len(input))
	total := float32(0)
	for i, x := range input {
		g := seed[i] * weight[i]
		value := g * inv
		value += ((gvar * 2) * (x - mean)) / n
		out[i] = value
		total -= value
	}
	for i := range out {
		out[i] += total / n
	}
	// Independent analytic F64 population LayerNorm derivative at original
	// F32 x/w/seed, including newly computed F64 mean/variance/inverse.
	var m, v, sum, dot float64
	for _, x := range input {
		m += float64(x)
	}
	m /= float64(len(input))
	for _, x := range input {
		c := float64(x) - m
		v += c * c
	}
	v /= float64(len(input))
	ir := 1 / math.Sqrt(v+1e-5)
	for i, x := range input {
		g := float64(seed[i]) * float64(weight[i])
		sum += g
		dot += g * ((float64(x) - m) * ir)
	}
	ref := make([]float64, len(input))
	for i, x := range input {
		g := float64(seed[i]) * float64(weight[i])
		ref[i] = ir * (g - sum/float64(len(input)) - ((float64(x)-m)*ir)*dot/float64(len(input)))
	}
	return out, ref, nil
}

type sharedNormSensitivityBackend struct {
	values                                            int
	maxCPUShader, maxCPUReference, maxShaderReference float64
	CPUShaderSquare                                   float64
}

func (b *sharedNormSensitivityBackend) EvaluateTransformer(m *TransformerCPU, sequence, seed []float32, rows int) (TransformerTrainingResult, error) {
	if b == nil {
		return TransformerTrainingResult{}, fmt.Errorf("nil norm diagnostic")
	}
	if err := validateTrainableTransformer(m, sequence, seed, rows); err != nil {
		return TransformerTrainingResult{}, err
	}
	if m.FinalWeight == nil {
		return TransformerTrainingResult{}, fmt.Errorf("final norm required")
	}
	tape, output := m.forwardTraining(rows, sequence)
	g := newTransformerGradients(m)
	cpu := transformerNormBackward(rows, m.Width, tape.final, seed, m.FinalWeight, g.FinalWeight, g.FinalBias)
	dh := make([]float32, len(cpu))
	for row := 0; row < rows; row++ {
		at, end := row*m.Width, (row+1)*m.Width
		x, ref, err := sharedNormAdjointDiagnostic(tape.final.input[at:end], seed[at:end], m.FinalWeight, tape.final.inv[row])
		if err != nil {
			return TransformerTrainingResult{}, err
		}
		copy(dh[at:end], x)
		if !zeroTrainingSeed(seed) {
			for i, v := range x {
				c := float64(cpu[at+i])
				d := math.Abs(c - float64(v))
				b.values++
				b.CPUShaderSquare += d * d
				b.maxCPUShader = math.Max(b.maxCPUShader, d)
				b.maxCPUReference = math.Max(b.maxCPUReference, math.Abs(c-ref[i]))
				b.maxShaderReference = math.Max(b.maxShaderReference, math.Abs(float64(v)-ref[i]))
			}
		}
	}
	for layer := len(m.Layers) - 1; layer >= 0; layer-- {
		var err error
		dh, err = m.Layers[layer].backwardTrainingWithAttention(rows, m.Width, m.Heads, m.HeadDim, m.Context, m.MaxPeriod, tape.layers[layer], dh, &g.Layers[layer], nil)
		if err != nil {
			return TransformerTrainingResult{}, err
		}
	}
	all := map[string][]float32{}
	addFlowLMGradients(all, &FlowLMTrainingGradients{Transformer: g})
	for name := range all {
		if !strings.HasPrefix(name, "flow_lm.transformer.layers.") && !strings.HasPrefix(name, "flow_lm.out_norm.") {
			delete(all, name)
		}
	}
	return TransformerTrainingResult{Output: output, DSequence: dh, Parameters: all}, nil
}

func TestPocketSharedNormSensitivityBoundary(t *testing.T) {
	// Exact dyadic simple derivative: x=(-1,1), constant weights/seed yield
	// zero adjoint; general seeds compare to independent analytic F64 formula.
	for _, seed := range [][]float32{{1, 1}, {.25, -.5}} {
		x := []float32{-1, 1}
		w := []float32{1, 1}
		inv := float32(1 / math.Sqrt(1+1e-5))
		got, ref, err := sharedNormAdjointDiagnostic(x, seed, w, inv)
		if err != nil {
			t.Fatal(err)
		}
		for i, v := range got {
			if math.Abs(float64(v)-ref[i]) > 2e-7 {
				t.Fatal("norm formula", v, ref[i])
			}
		}
	}
	if _, _, err := sharedNormAdjointDiagnostic(nil, nil, nil, 1); err == nil {
		t.Fatal("empty norm")
	}
	if _, _, err := sharedNormAdjointDiagnostic([]float32{1, 2}, []float32{1, 2}, []float32{1, 2}, 0); err == nil {
		t.Fatal("invalid inverse")
	}
	lm := tinyFlowLMTraining()
	rows := 4
	x, seed := make([]float32, rows*lm.Hidden), make([]float32, rows*lm.Hidden)
	for i := range x {
		x[i] = float32(i%7-3) * .1
		seed[i] = float32(i%3-1) * .03
	}
	b := &sharedNormSensitivityBackend{}
	r, err := b.EvaluateTransformer(lm.Transformer, x, seed, rows)
	if err != nil {
		t.Fatal(err)
	}
	if err := validateTransformerBackendResult(lm.Transformer, r, x, seed, rows); err != nil {
		t.Fatal(err)
	}
	if b.values != len(x) {
		t.Fatal("norm boundary count")
	}
}
