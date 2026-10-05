package pockettts

import (
	"encoding/json"
	"math"
	"os"
	"testing"
)

func TestTrainingBranchTapeReference(t *testing.T) {
	cases := []struct {
		kind   string
		inputs [][]float32
		s      float32
		want   [][]float32
	}{
		{"interpolate", [][]float32{{4, -2}, {-4, 6}}, .25, [][]float32{{-2, 4}, {8, -8}}},
		{"advance", [][]float32{{1, 2}, {3, -4}, {5, 6}}, .5, [][]float32{{2.5, 0}, {5.5, -1}}},
		{"negate", [][]float32{{3, -4}}, 0, [][]float32{{-3, 4}}},
		{"mixed", [][]float32{{2, 3}, {4, -2}, {5, 6}, {7, -1}}, .5, [][]float32{{4, 2}, {1, 1.5}, {19}}},
		{"finish", [][]float32{{2, 3}, {4, -2}, {5, 6}, {7, -1}, {8, 9}, {19}}, .25, [][]float32{{1.5, .25}, {4.5, .75}, {-16, 28}}},
		{"diagonal-target", [][]float32{{2, 3}, {4, -2}}, .5, [][]float32{{-3, 3.5}}},
		{"scale", [][]float32{{2, -4}}, .5, [][]float32{{1, -2}}},
	}
	for _, tc := range cases {
		t.Run(tc.kind, func(t *testing.T) {
			tape, err := BuildBranchTrainingTape(tc.kind, tc.inputs, tc.s, 1<<20)
			if err != nil {
				t.Fatal(err)
			}
			got := interpretFlowTape(t, tape)
			if err := validateBranchBoundary(tc.kind, got, tc.inputs); err != nil {
				t.Fatal(err)
			}
			for i, want := range tc.want {
				assertSliceClose(t, tc.kind, got[i], want, 0)
			}
		})
	}
	var maxVector, maxReduction float64
	for _, n := range []int{1, 32, 257, 16384} {
		inputs := make([][]float32, 4)
		for j := range inputs {
			inputs[j] = make([]float32, n)
			for i := range inputs[j] {
				inputs[j][i] = float32((i*(j+3))%29-14) * .0037
			}
		}
		tape, err := BuildBranchTrainingTape("mixed", inputs, .375, 64<<20)
		if err != nil {
			t.Fatal(err)
		}
		got := interpretFlowTape(t, tape)
		var dot float64
		for i := range inputs[0] {
			a, b, c, d := float64(inputs[0][i]), float64(inputs[1][i]), float64(inputs[2][i]), float64(inputs[3][i])
			for k, want := range []float64{a + .375*b, .375 * a} {
				e := math.Abs(float64(got[k][i]) - want)
				maxVector = math.Max(maxVector, e)
				if e > 1e-8 {
					t.Fatal("vector F64", n, i, e)
				}
			}
			dot += a*d + b*c
		}
		e := math.Abs(float64(got[2][0]) - dot)
		maxReduction = math.Max(maxReduction, e)
		if e > 2e-6+2e-5*math.Abs(dot) {
			t.Fatal("dot F64", n, got[2], dot, e)
		}
	}
	t.Logf("Branch F32/F64 max vector=%g reduction=%g over channel1/32/257/16384", maxVector, maxReduction)
}
func TestTrainingBranchTapeAdmission(t *testing.T) {
	for _, bad := range []struct {
		kind   string
		inputs [][]float32
		s      float32
		b      int64
	}{{"unknown", nil, 0, 1024}, {"scale", nil, 1, 1024}, {"scale", [][]float32{nil}, 1, 1024}, {"advance", [][]float32{{1}, {2}}, .5, 1024}, {"interpolate", [][]float32{{1}, {2}}, -1, 1024}, {"scale", [][]float32{{float32(math.NaN())}}, 1, 1024}, {"scale", [][]float32{{1}}, float32(math.Inf(1)), 1024}, {"scale", [][]float32{{1}}, 1, 1}, {"finish", [][]float32{{1}, {1}, {1}, {1}, {1}, {1}}, .5, 1024}} {
		if _, err := BuildBranchTrainingTape(bad.kind, bad.inputs, bad.s, bad.b); err == nil {
			t.Fatal("admitted", bad.kind)
		}
	}
	input := []float32{1, 2}
	if validateBranchBoundary("scale", [][]float32{input}, [][]float32{input}) == nil {
		t.Fatal("borrowed")
	}
	if validateBranchBoundary("advance", [][]float32{input, input}, [][]float32{{0, 0}, {0, 0}, {0, 0}}) == nil {
		t.Fatal("overlap")
	}
	if validateBranchBoundary("scale", [][]float32{nil}, [][]float32{input}) == nil || validateBranchBoundary("bad", nil, [][]float32{input}) == nil || validateBranchBoundary("scale", nil, nil) == nil {
		t.Fatal("extent/kind")
	}
}

// hostBranchTestBackend is wiring evidence only; fixture/F64 expected values
// above and pinned PyTorch below are independent of its interpreter.
type hostBranchTestBackend struct{ t *testing.T }

func (b hostBranchTestBackend) EvaluateBranch(_ *FlowHeadCPU, kind string, inputs [][]float32, s float32) ([][]float32, error) {
	tape, err := BuildBranchTrainingTape(kind, inputs, s, 64<<20)
	if err != nil {
		return nil, err
	}
	return interpretFlowTape(b.t, tape), nil
}
func TestTrainingBranchMinimalLSDPyTorch(t *testing.T) {
	data, err := os.ReadFile("testdata/flow_head_backward_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle flowHeadBackwardOracle
	if err := json.Unmarshal(data, &oracle); err != nil || oracle.UpstreamRevision != UpstreamCommit || oracle.Schema != 1 {
		t.Fatal("pin", err)
	}
	model := flowHeadFromOracle(t, oracle.Parameters)
	model.BranchBackend = hostBranchTestBackend{t}
	backend := &FlowTapeTrainingBackend{Executor: hostFlowTapeExecutor{t}, MaxBytes: 64 << 20}
	want := oracle.LSDMinimal
	r, g, err := model.LSDDistillTrainingBackend(backend, want.Condition, want.S, want.T, want.Noise, want.Target, want.LogVariance)
	if err != nil {
		t.Fatal(err)
	}
	assertClose64(t, "branch LSD loss", r.Loss, float64(want.Loss), 2e-6)
	assertSliceClose(t, "branch LSD residual", r.Residual, want.Residual, 5e-6)
	assertSliceClose(t, "branch LSD target", g.Target, want.DTarget, 2e-5)
	assertSliceClose(t, "branch LSD noise", g.Noise, want.DNoise, 2e-5)
	assertSliceClose(t, "branch LSD condition", g.Condition, want.DCondition, 2e-5)
	assertClose64(t, "branch LSD DS", float64(g.DS), float64(want.DS), 2e-5)
	assertClose64(t, "branch LSD DT", float64(g.DT), float64(want.DT), 2e-5)
	for name, x := range flowHeadGradientMap(g.Flow) {
		assertSliceClose(t, "branch LSD parameter "+name, x, want.Gradients[name], 3e-5)
	}
}
