package pockettts

import (
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"testing"
)

func audioProjectionFixture() *FlowLMTrainingCPU {
	return &FlowLMTrainingCPU{Hidden: 3, LatentDim: 2, BOS: []float32{.3, -.7}, Input: LinearF32{In: 2, Out: 3, Weight: []float32{.4, -.2, .7, .1, -.3, .6}}}
}
func TestTrainingAudioProjectionTapeReference(t *testing.T) {
	for _, rows := range []int{1, 2, 7, 257} {
		m := audioProjectionFixture()
		batch := FlowLMTrainingBatch{Frames: rows, NormalizedLatents: make([]float32, 2*rows)}
		seed := make([]float32, 3*rows)
		for i := range batch.NormalizedLatents {
			batch.NormalizedLatents[i] = float32(i%11-5) * .08
		}
		for i := range seed {
			seed[i] = float32(i%7-3) * .09
		}
		p, err := m.BuildAudioProjectionTrainingTape(batch, seed, 1<<24)
		if err != nil {
			t.Fatal(rows, err)
		}
		if err = train.ValidateResidentTapeBindings(p.Tape, p.ResidentBindings, 1<<24); err != nil {
			t.Fatal(err)
		}
		out := interpretFlowTape(t, p.Tape)
		values, dx, dw, dbos := make([]float64, 3*rows), make([]float64, 2*rows), make([]float64, 6), make([]float64, 2)
		for row := 0; row < rows; row++ {
			x := m.BOS
			if row > 0 {
				x = batch.NormalizedLatents[(row-1)*2 : row*2]
			}
			for o := 0; o < 3; o++ {
				dy := float64(seed[row*3+o])
				for k := 0; k < 2; k++ {
					values[row*3+o] += float64(m.Input.Weight[o*2+k]) * float64(x[k])
					dw[o*2+k] += dy * float64(x[k])
					d := dy * float64(m.Input.Weight[o*2+k])
					if row == 0 {
						dbos[k] += d
					} else {
						dx[(row-1)*2+k] += d
					}
				}
			}
		}
		check := func(name string, x []float32, y []float64) {
			t.Helper()
			if len(x) != len(y) {
				t.Fatal(name, "shape")
			}
			for i, v := range x {
				if math.Abs(float64(v)-y[i]) > 2e-7+2e-6*math.Abs(y[i]) {
					t.Fatal(rows, name, i, v, y[i])
				}
			}
		}
		check("projected audio", out[0], values)
		check("normalized VJP", out[1], dx)
		check("weight VJP", out[2], dw)
		check("BOS VJP", out[3], dbos)
		for _, v := range out[1][(rows-1)*2:] {
			if v != 0 {
				t.Fatal("unused final row")
			}
		}
		// Independently compose original assembly and projection VJPs.
		assembly, err := m.BuildAudioAssemblyTrainingTape(batch, make([]float32, rows*2), 1<<24)
		if err != nil {
			t.Fatal(err)
		}
		a := interpretFlowTape(t, assembly.Tape)
		projection, err := BuildProjectionTrainingTape(m.Input, "flow_lm.input_linear.weight", "", a[0], seed, rows, 1<<24)
		if err != nil {
			t.Fatal(err)
		}
		b := interpretFlowTape(t, projection.Tape)
		backward, err := m.BuildAudioAssemblyTrainingTape(batch, b[1], 1<<24)
		if err != nil {
			t.Fatal(err)
		}
		c := interpretFlowTape(t, backward.Tape)
		assertSliceClose(t, "legacy output", out[0], b[0], 2e-7)
		assertSliceClose(t, "legacy normalized VJP", out[1], c[1], 2e-7)
		assertSliceClose(t, "legacy weight VJP", out[2], b[2], 2e-7)
		assertSliceClose(t, "legacy BOS VJP", out[3], c[2], 2e-7)
	}
}
func TestTrainingAudioProjectionTapeAdmission(t *testing.T) {
	m := audioProjectionFixture()
	batch := FlowLMTrainingBatch{Frames: 2, NormalizedLatents: []float32{.1, .2, .3, .4}}
	seed := make([]float32, 6)
	for _, frames := range []int{0, 1025} {
		b := batch
		b.Frames = frames
		if _, err := m.BuildAudioProjectionTrainingTape(b, seed, 1<<20); err == nil {
			t.Fatal("frames")
		}
	}
	if _, err := m.BuildAudioProjectionTrainingTape(batch, seed, 1); err == nil {
		t.Fatal("budget")
	}
	if _, err := m.BuildAudioProjectionTrainingTape(batch, nil, 1<<20); err == nil {
		t.Fatal("seed shape")
	}
	for _, which := range []string{"BOS", "input", "weight", "seed"} {
		m := audioProjectionFixture()
		b := FlowLMTrainingBatch{Frames: 2, NormalizedLatents: append([]float32(nil), batch.NormalizedLatents...)}
		s := make([]float32, 6)
		switch which {
		case "BOS":
			m.BOS[0] = float32(math.NaN())
		case "input":
			b.NormalizedLatents[0] = float32(math.NaN())
		case "weight":
			m.Input.Weight[0] = float32(math.NaN())
		case "seed":
			s[0] = float32(math.NaN())
		}
		if _, err := m.BuildAudioProjectionTrainingTape(b, s, 1<<20); err == nil {
			t.Fatal("nonfinite", which)
		}
	}
	m.Input.Bias = []float32{0, 0, 0}
	if _, err := m.BuildAudioProjectionTrainingTape(batch, seed, 1<<20); err == nil {
		t.Fatal("bias topology")
	}
	m.Input.Bias = nil
	p, err := m.BuildAudioProjectionTrainingTape(batch, seed, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	before := interpretFlowTape(t, p.Tape)
	batch.NormalizedLatents[0] = 99
	seed[0] = 99
	m.Input.Weight[0] = 99
	m.BOS[0] = 99
	after := interpretFlowTape(t, p.Tape)
	for i, x := range before {
		assertSliceClose(t, "public owned", after[i], x, 0)
	}
	if validateAudioProjectionBoundary(AudioProjectionTrainingResult{Output: make([]float32, 6), DNormalized: batch.NormalizedLatents}, 2, 3, 2, batch.NormalizedLatents) == nil {
		t.Fatal("borrowed result")
	}
}
