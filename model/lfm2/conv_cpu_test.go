package lfm2

import (
	"strings"
	"testing"
)

func tinyLFM2ConvSource(cfg Config) fakeLFM2TensorSource {
	identity := []float32{1, 0, 0, 1}
	return fakeLFM2TensorSource{
		"model.layers.0.conv.in_proj.weight": {data: []float32{
			1, 0, 0, 1, // B
			1, 0, 0, 1, // C
			1, 0, 0, 1, // x
		}, shape: []int{6, 2}},
		"model.layers.0.conv.out_proj.weight": {data: identity, shape: []int{2, 2}},
		"model.layers.0.conv.conv.weight":     {data: []float32{.5, 1, .25, 2}, shape: []int{2, 1, cfg.ConvLCache}},
	}
}

func TestShortConvCPUForwardTokenMatchesDepthwiseReference(t *testing.T) {
	cfg := tinyLFM2Config(true)
	model, err := LoadShortConvCPU(tinyLFM2ConvSource(cfg), cfg, 0)
	if err != nil {
		t.Fatal(err)
	}
	state := []float32{2, 3, 4, 5}
	out, next, err := model.ForwardToken([]float32{2, 3}, state)
	if err != nil {
		t.Fatal(err)
	}
	// B*x = [4,9], shifted state = [[3,4],[5,9]]. The depthwise sums are
	// [5.5,19.25], then C=[2,3] gates them.
	wantOut := []float32{11, 57.75}
	wantState := []float32{3, 4, 5, 9}
	for i := range wantOut {
		if out[i] != wantOut[i] {
			t.Fatalf("out=%v want=%v", out, wantOut)
		}
	}
	for i := range wantState {
		if next[i] != wantState[i] {
			t.Fatalf("state=%v want=%v", next, wantState)
		}
	}
	if state[0] != 2 || state[1] != 3 || state[2] != 4 || state[3] != 5 {
		t.Fatalf("caller state mutated: %v", state)
	}
}

func TestShortConvCPURejectsWrongLayerAndState(t *testing.T) {
	cfg := tinyLFM2Config(true)
	src := tinyLFM2ConvSource(cfg)
	if _, err := LoadShortConvCPU(src, cfg, 1); err == nil || !strings.Contains(err.Error(), "not a convolution") {
		t.Fatalf("layer error=%v", err)
	}
	model, err := LoadShortConvCPU(src, cfg, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := model.ForwardToken([]float32{1, 2}, []float32{1}); err == nil {
		t.Fatal("accepted short conv state")
	}
	if got := model.NewState(); len(got) != cfg.HiddenSize*cfg.ConvLCache {
		t.Fatalf("state len=%d", len(got))
	}
}

func TestShortConvCPUWithBias(t *testing.T) {
	cfg := tinyLFM2Config(true)
	cfg.ConvBias = true
	src := tinyLFM2ConvSource(cfg)
	src["model.layers.0.conv.in_proj.bias"] = struct {
		data  []float32
		shape []int
	}{make([]float32, 6), []int{6}}
	src["model.layers.0.conv.out_proj.bias"] = struct {
		data  []float32
		shape []int
	}{[]float32{1, -1}, []int{2}}
	src["model.layers.0.conv.conv.bias"] = struct {
		data  []float32
		shape []int
	}{[]float32{.5, .5}, []int{2}}
	model, err := LoadShortConvCPU(src, cfg, 0)
	if err != nil {
		t.Fatal(err)
	}
	out, _, err := model.ForwardToken([]float32{2, 3}, make([]float32, 4))
	if err != nil {
		t.Fatal(err)
	}
	if out[0] != 10 || out[1] != 54.5 {
		t.Fatalf("biased out=%v", out)
	}
}
