package pockettts

import (
	"encoding/json"
	train "github.com/rcarmo/go-pherence/backends/training"
	"math"
	"os"
	"testing"
)

func TestTrainingTimeEmbeddingReference(t *testing.T) {
	var maxValue, maxTangent, maxAdjoint float64
	// Independent F64 calculus, with F32 angle rounding as in the CPU contract.
	for _, time := range []float32{-32, -1, -.125, 0, .3, 1, 32} {
		for _, dt := range []float32{0, 1} {
			f := []float32{0, .25, 1, 7.5, 256}
			gv := []float32{.2, -.1, .4, .3, -.5, .1, .8, -.2, .05, .9}
			gt := []float32{-.4, .3, .2, .6, .1, .9, -.5, .7, -.1, .2}
			tape := train.Tape{Tensors: []train.TensorSpec{{Shape: []int{5}, Values: f}, {Shape: []int{2}, Values: []float32{time, dt}}, {Shape: []int{10}, Values: gv}, {Shape: []int{10}, Values: gt}, {Shape: []int{10}}, {Shape: []int{10}}, {Shape: []int{1}}}, Commands: []train.Command{{Op: "time-embedding", Inputs: []int{0, 1, 2, 3}, Output: 4}, {Op: "time-embedding", Inputs: []int{0, 1, 2, 3}, Output: 5, Mode: 1}, {Op: "time-embedding", Inputs: []int{0, 1, 2, 3}, Output: 6, Mode: 2}}, Outputs: []int{4, 5, 6}}
			got := interpretFlowTape(t, tape)
			var wantAdj float64
			for i, freq := range f {
				ff, angle := float64(freq), float64(time*freq)
				s, c := math.Sin(angle), math.Cos(angle)
				for _, v := range []struct {
					got  float32
					want float64
					tol  float64
				}{{got[0][i], c, 6e-8}, {got[0][5+i], s, 6e-8}, {got[1][i], -s * ff * float64(dt), 2e-5}, {got[1][5+i], c * ff * float64(dt), 2e-5}} {
					error := math.Abs(float64(v.got) - v.want)
					if v.tol == 6e-8 {
						maxValue = math.Max(maxValue, error)
					} else {
						maxTangent = math.Max(maxTangent, error)
					}
					if error > v.tol {
						t.Fatal("time reference", time, dt, i, v)
					}
				}
				wantAdj += float64(gv[i])*(-s*ff) + float64(gv[5+i])*(c*ff) + float64(gt[i])*(-c*ff*ff)*float64(dt) + float64(gt[5+i])*(-s*ff*ff)*float64(dt)
			}
			// F32 terms and ordered reduction: bound scales with derivative magnitude;
			// finite-difference evidence below uses smooth F64 arithmetic independently.
			maxAdjoint = math.Max(maxAdjoint, math.Abs(float64(got[2][0])-wantAdj))
			if math.Abs(float64(got[2][0])-wantAdj) > .002+2e-7*math.Abs(wantAdj) {
				t.Fatal("time adjoint", time, dt, got[2], wantAdj)
			}
		}
	}
	t.Logf("F32 versus independent F64 maxima: value=%g tangent=%g time-adjoint=%g; 14 cases, angles through +/-8192", maxValue, maxTangent, maxAdjoint)
	// Scalar reverse-over-forward objective finite difference, not interpreter.
	f := []float64{.3, 1.1, 7}
	gv := []float64{.2, -.4, .1, .8, .3, -.5}
	gt := []float64{-.7, .2, .3, .1, .4, .6}
	time, dt := .37, .8
	objective := func(time float64) float64 {
		var sum float64
		for i, f := range f {
			s, c := math.Sin(time*f), math.Cos(time*f)
			sum += gv[i]*c + gv[3+i]*s + gt[i]*(-s*f*dt) + gt[3+i]*(c*f*dt)
		}
		return sum
	}
	var analytic float64
	for i, f := range f {
		s, c := math.Sin(time*f), math.Cos(time*f)
		analytic += gv[i]*(-s*f) + gv[3+i]*(c*f) + gt[i]*(-c*f*f*dt) + gt[3+i]*(-s*f*f*dt)
	}
	numeric := (objective(time+1e-6) - objective(time-1e-6)) / (2e-6)
	if math.Abs(analytic-numeric) > 2e-8 {
		t.Fatal("finite difference", analytic, numeric)
	}
}
func TestTrainingFlowDeviceTimeMixedParity(t *testing.T) {
	data, err := os.ReadFile("testdata/flow_head_backward_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle flowHeadBackwardOracle
	if err := json.Unmarshal(data, &oracle); err != nil || oracle.Schema != 1 || oracle.UpstreamRevision != UpstreamCommit {
		t.Fatal("oracle pin", err)
	}
	for _, tc := range oracle.Mixed {
		model := flowHeadFromOracle(t, oracle.Parameters)
		tape, err := model.BuildFlowDeviceTimeTrainingTape(oracle.Condition, oracle.Times, oracle.Input, tc.TimeIndex, tc.DOutput, tc.DTangent, 64<<20)
		if err != nil {
			t.Fatal(err)
		}
		result, err := tape.Decode(interpretFlowTape(t, tape.Tape))
		if err != nil {
			t.Fatal(err)
		}
		assertSliceClose(t, "mixed"+" device time output", result.Output, oracle.Output, 3e-6)
		assertSliceClose(t, "mixed"+" device time tangent", result.Tangent, oracle.JVPTimes[tc.TimeIndex], 5e-6)
		assertSliceClose(t, "mixed"+" device time input", result.DInput, tc.DInput, 8e-6)
		assertSliceClose(t, "mixed"+" device time condition", result.DCondition, tc.DCondition, 8e-6)
		assertSliceClose(t, "mixed"+" device time adjoint", result.DTimes, tc.DTimes, 8e-6)
		for name, got := range result.Parameters {
			want, ok := tc.Gradients[name[len("flow_lm.flow_net."):]]
			if !ok {
				t.Fatal(name)
			}
			assertSliceClose(t, "mixed"+" device time parameter "+name, got, want, 1e-5)
		}
		if tape.boundarySize() != 4+len(model.Time) {
			t.Fatal("scalar boundary")
		}
	}
}
