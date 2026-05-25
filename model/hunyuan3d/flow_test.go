package hunyuan3d

import (
	"reflect"
	"testing"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
)

func TestBlendClassifierFreeGuidance(t *testing.T) {
	uncond := []float32{1, 2, 3}
	cond := []float32{2, 0, 5}
	got, err := BlendClassifierFreeGuidance(uncond, cond, 2)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{3, -2, 7}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("cfg blend=%v want %v", got, want)
	}
	if _, err := BlendClassifierFreeGuidance(uncond, cond[:2], 1); err == nil {
		t.Fatal("mismatched CFG tensors accepted")
	}
}

func TestDeterministicLatents(t *testing.T) {
	a, err := DeterministicLatents([]int{2, 3, 4}, 1234)
	if err != nil {
		t.Fatal(err)
	}
	b, err := DeterministicLatents([]int{2, 3, 4}, 1234)
	if err != nil {
		t.Fatal(err)
	}
	c, err := DeterministicLatents([]int{2, 3, 4}, 5678)
	if err != nil {
		t.Fatal(err)
	}
	if len(a) != 24 || !reflect.DeepEqual(a, b) {
		t.Fatalf("deterministic latents not reproducible len=%d equal=%v", len(a), reflect.DeepEqual(a, b))
	}
	if reflect.DeepEqual(a, c) {
		t.Fatal("different seeds produced identical latents")
	}
	if _, err := DeterministicLatents([]int{2, 0, 4}, 1); err == nil {
		t.Fatal("bad latent shape accepted")
	}
}

func TestRunFlowMatchReference(t *testing.T) {
	schedule, err := loaderconfig.Hunyuan3DFlowMatchScheduleFor(loaderconfig.Hunyuan3DSchedulerParams{NumTrainTimesteps: 1000}, 3)
	if err != nil {
		t.Fatal(err)
	}
	initial := []float32{10, 20}
	outputs := [][]float32{{3, -3}, {6, 0}, {0, 9}}
	got, err := RunFlowMatchReference(initial, schedule, outputs)
	if err != nil {
		t.Fatal(err)
	}
	// Three scheduler transitions with terminal sigma: +0.5*out0, +0.5*out1, +0*out2.
	want := []float32{14.5, 18.5}
	for i := range want {
		if got[i] < want[i]-1e-6 || got[i] > want[i]+1e-6 {
			t.Fatalf("flow[%d]=%v want %v full=%v", i, got[i], want[i], got)
		}
	}
	if !reflect.DeepEqual(initial, []float32{10, 20}) {
		t.Fatalf("initial latents mutated: %v", initial)
	}
	if _, err := RunFlowMatchReference(initial, schedule, outputs[:1]); err == nil {
		t.Fatal("wrong output count accepted")
	}
	bad := append([][]float32(nil), outputs...)
	bad[1] = []float32{1}
	if _, err := RunFlowMatchReference(initial, schedule, bad); err == nil {
		t.Fatal("bad model output shape accepted")
	}
}
