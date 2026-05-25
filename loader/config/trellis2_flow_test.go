package config

import "testing"

func TestTrellis2FlowEulerSchedule(t *testing.T) {
	got, err := Trellis2FlowEulerSchedule(5, 0)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{1, 0.75, 0.5, 0.25, 0}
	for i := range want {
		if got.Timesteps[i] != want[i] {
			t.Fatalf("timesteps=%v want %v", got.Timesteps, want)
		}
		if got.ModelTimesteps[i] != want[i]*1000 {
			t.Fatalf("model_timesteps=%v", got.ModelTimesteps)
		}
	}
	if got.TransitionCount != 5 {
		t.Fatalf("transition count=%d", got.TransitionCount)
	}

	shifted, err := Trellis2FlowEulerSchedule(3, 0.2)
	if err != nil {
		t.Fatal(err)
	}
	want = []float64{1, 0.6, 0.2}
	for i := range want {
		if shifted.Timesteps[i] < want[i]-1e-12 || shifted.Timesteps[i] > want[i]+1e-12 {
			t.Fatalf("shifted timesteps=%v want %v", shifted.Timesteps, want)
		}
	}
	if _, err := Trellis2FlowEulerSchedule(0, 0); err == nil {
		t.Fatal("bad steps accepted")
	}
	if _, err := Trellis2FlowEulerSchedule(4, 1.5); err == nil {
		t.Fatal("bad sigma_min accepted")
	}
}

func TestTrellis2FlowEulerStep(t *testing.T) {
	dst := make([]float32, 3)
	if err := Trellis2FlowEulerStep(dst, []float32{10, 20, -5}, []float32{4, -2, 8}, 1, 0.75); err != nil {
		t.Fatal(err)
	}
	want := []float32{9, 20.5, -7}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("step=%v want %v", dst, want)
		}
	}
	if err := Trellis2FlowEulerStep(dst[:2], []float32{1, 2, 3}, []float32{1, 2, 3}, 1, 0); err == nil {
		t.Fatal("shape mismatch accepted")
	}
}

func TestTrellis2VToXStartEps(t *testing.T) {
	x0, eps, err := Trellis2VToXStartEps([]float32{10, 20}, []float32{4, -2}, 0.25)
	if err != nil {
		t.Fatal(err)
	}
	if x0[0] != 9 || x0[1] != 20.5 || eps[0] != 13 || eps[1] != 18.5 {
		t.Fatalf("x0=%v eps=%v", x0, eps)
	}
	if _, _, err := Trellis2VToXStartEps([]float32{1}, []float32{1, 2}, 0.5); err == nil {
		t.Fatal("shape mismatch accepted")
	}
}

func TestTrellis2CFGBlend(t *testing.T) {
	dst := make([]float32, 3)
	if err := Trellis2CFGBlend(dst, []float32{1, 2, 3}, []float32{2, 0, 5}, 2); err != nil {
		t.Fatal(err)
	}
	want := []float32{3, -2, 7}
	for i := range want {
		if dst[i] != want[i] {
			t.Fatalf("cfg=%v want %v", dst, want)
		}
	}
}
