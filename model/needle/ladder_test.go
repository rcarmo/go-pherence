package needle

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"

	checkpoint "github.com/rcarmo/go-pherence/loader/needle"
)

type rawLadderConfig struct {
	NumLayers       int   `json:"num_layers"`
	LadderOrder     []int `json:"ladder_order"`
	GlobalLayers    []int `json:"global_layers"`
	EngramLayers    []int `json:"engram_layers"`
	LadderDepths    []int `json:"ladder_depths"`
	LadderSample    bool  `json:"ladder_sample"`
	EmbeddingProbes int   `json:"embedding_probes"`
}

func decodeRawLadderConfig(t testing.TB, raw json.RawMessage) rawLadderConfig {
	t.Helper()
	var cfg rawLadderConfig
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatal(err)
	}
	return cfg
}

func repeatFirstAxis(t checkpoint.Tensor, n int) checkpoint.Tensor {
	shape := append([]int(nil), t.Shape...)
	shape[0] = n
	inner := 1
	for _, d := range t.Shape[1:] {
		inner *= d
	}
	data := make([]float32, n*inner)
	for i := 0; i < n; i++ {
		src := (i % t.Shape[0]) * inner
		copy(data[i*inner:(i+1)*inner], t.Data[src:src+inner])
	}
	return checkpoint.Tensor{Shape: shape, Data: data}
}

func tensorWithAxisMarkers(shape []int, axis int, base float32) checkpoint.Tensor {
	t := checkpoint.Tensor{Shape: append([]int(nil), shape...)}
	total := 1
	for _, d := range shape {
		total *= d
	}
	t.Data = make([]float32, total)
	outer, inner := 1, 1
	for _, d := range shape[:axis] {
		outer *= d
	}
	for _, d := range shape[axis+1:] {
		inner *= d
	}
	for o := 0; o < outer; o++ {
		for row := 0; row < shape[axis]; row++ {
			v := base + float32(row)
			start := (o*shape[axis] + row) * inner
			for i := 0; i < inner; i++ {
				t.Data[start+i] = v
			}
		}
	}
	return t
}

func seqTensor(shape []int, start float32) checkpoint.Tensor {
	total := 1
	for _, d := range shape {
		total *= d
	}
	data := make([]float32, total)
	for i := range data {
		data[i] = start + float32(i)
	}
	return checkpoint.Tensor{Shape: append([]int(nil), shape...), Data: data}
}

func axisMarkers(t checkpoint.Tensor, axis int) []float32 {
	markers := make([]float32, t.Shape[axis])
	outer, inner := 1, 1
	for _, d := range t.Shape[:axis] {
		outer *= d
	}
	for _, d := range t.Shape[axis+1:] {
		inner *= d
	}
	for row := range markers {
		markers[row] = t.Data[row*inner]
		if outer > 1 {
			markers[row] = t.Data[row*inner]
		}
	}
	return markers
}

func makeLadderFixture(t testing.TB, layers int, order, global, engram []int) *Model {
	t.Helper()
	m, _ := fixture(t)
	cp := m.Checkpoint()

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(cp.Config, &raw); err != nil {
		t.Fatal(err)
	}
	for key, val := range map[string]any{
		"num_layers":    layers,
		"ladder_order":  order,
		"global_layers": global,
		"engram_layers": engram,
	} {
		b, err := json.Marshal(val)
		if err != nil {
			t.Fatal(err)
		}
		raw[key] = b
	}
	var err error
	cp.Config, err = json.Marshal(raw)
	if err != nil {
		t.Fatal(err)
	}

	for name, tensor := range cp.Tensors {
		if strings.HasPrefix(name, "stack/") && !strings.HasPrefix(name, "stack/final_norm/") {
			cp.Tensors[name] = repeatFirstAxis(tensor, layers)
		}
	}
	cp.Tensors["stack/layers/block/attn_gate"] = checkpoint.Tensor{Shape: []int{layers}, Data: make([]float32, layers)}
	for i := 0; i < layers; i++ {
		cp.Tensors["stack/layers/block/attn_gate"].Data[i] = 100 + float32(i)
	}

	engrams := map[string]checkpoint.Tensor{}
	for name, tensor := range cp.Tensors {
		site, rest, ok := parseEngramTensorName(name)
		if !ok {
			continue
		}
		if site != 0 {
			continue
		}
		engrams[rest] = cloneCheckpointTensor(tensor)
		delete(cp.Tensors, name)
	}
	for site := 0; site < len(engram); site++ {
		for rest, tensor := range engrams {
			cloned := cloneCheckpointTensor(tensor)
			for i := range cloned.Data {
				cloned.Data[i] = float32(site*1000) + float32(i)
			}
			cp.Tensors[fmt.Sprintf("engrams_%d%s", site, rest)] = cloned
		}
	}

	d := m.config.DModel
	cp.Tensors["embedding_head/probes"] = tensorWithAxisMarkers([]int{layers + 1, 2, d}, 0, 10)
	cp.Tensors["embedding_head/gain"] = tensorWithAxisMarkers([]int{layers + 1, 2}, 0, 20)
	cp.Tensors["embedding_head/query"] = seqTensor([]int{2, d}, 30)
	cp.Tensors["embedding_head/row_bias"] = tensorWithAxisMarkers([]int{2, layers + 1, 2}, 1, 40)
	cp.Tensors["embedding_head/proj/kernel"] = seqTensor([]int{2 * d, 4}, 50)
	cp.Tensors["embedding_head/log_temp"] = checkpoint.Tensor{Shape: []int{1}, Data: []float32{60}}
	cp.Tensors["embedding_head/bias"] = checkpoint.Tensor{Shape: []int{1}, Data: []float32{61}}

	cp.Tensors["confidence_head/probes"] = tensorWithAxisMarkers([]int{layers + 1, 2, d}, 0, 70)
	cp.Tensors["confidence_head/gain"] = tensorWithAxisMarkers([]int{layers + 1, 2}, 0, 80)
	cp.Tensors["confidence_head/query"] = seqTensor([]int{2, d}, 90)
	cp.Tensors["confidence_head/row_bias"] = tensorWithAxisMarkers([]int{2, layers + 1, 2}, 1, 100)
	cp.Tensors["confidence_head/proj/kernel"] = seqTensor([]int{2 * d, 1}, 110)
	cp.Tensors["confidence_head/proj/bias"] = checkpoint.Tensor{Shape: []int{1}, Data: []float32{111}}

	cp.Tensors["router_head/probes"] = tensorWithAxisMarkers([]int{layers + 1, 2, d}, 0, 120)
	cp.Tensors["router_head/gain"] = tensorWithAxisMarkers([]int{layers + 1, 2}, 0, 130)
	cp.Tensors["router_head/query"] = seqTensor([]int{2, d}, 140)
	cp.Tensors["router_head/row_bias"] = tensorWithAxisMarkers([]int{2, layers + 1, 2}, 1, 150)
	cp.Tensors["router_head/proj/kernel"] = seqTensor([]int{2 * d, 3}, 160)
	cp.Tensors["router_head/proj/bias"] = checkpoint.Tensor{Shape: []int{3}, Data: []float32{161, 162, 163}}
	cp.Tensors["router_head/calibration"] = checkpoint.Tensor{Shape: []int{3}, Data: []float32{.9, 0, .6}}

	out, err := New(cp)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

func TestLadderOrderKnown20(t *testing.T) {
	got, err := ladderLayerOrder(20)
	if err != nil {
		t.Fatal(err)
	}
	want := []int{0, 19, 9, 14, 4, 6, 11, 16, 2, 7, 12, 17, 1, 3, 5, 8, 10, 13, 15, 18}
	if !slices.Equal(got, want) {
		t.Fatalf("order %v want %v", got, want)
	}
	selected, _, err := ladderLayerIndices(Config{Layers: 20}, 8)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(selected, []int{0, 4, 6, 9, 11, 14, 16, 19}) {
		t.Fatalf("selected %v", selected)
	}
}

func TestSliceDepthRejectsInvalidInputs(t *testing.T) {
	m2, _ := fixtureFile(t, "needle2.json")
	if _, err := m2.SliceDepth(2); err == nil {
		t.Fatal("Needle2 slice accepted")
	}
	m := makeLadderFixture(t, 6, []int{0, 5, 2, 4, 1, 3}, []int{1, 2, 4, 5}, []int{0, 3, 5})
	for _, depth := range []int{1, 7} {
		if _, err := m.SliceDepth(depth); err == nil {
			t.Fatalf("accepted depth %d", depth)
		}
	}
	bad := m.Checkpoint()
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(bad.Config, &raw); err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal([]int{0, 5, 2, 4, 1, 1})
	raw["ladder_order"] = b
	bad.Config, _ = json.Marshal(raw)
	badModel, err := New(bad)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = badModel.SliceDepth(4); err == nil {
		t.Fatal("invalid ladder_order accepted")
	}
	bad = m.Checkpoint()
	bad.Tensors["stack/extra"] = checkpoint.Tensor{Shape: []int{1}, Data: []float32{1}}
	badModel, err = New(bad)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = badModel.SliceDepth(4); err == nil {
		t.Fatal("unknown stacked geometry accepted")
	}
	bad = m.Checkpoint()
	bad.Tensors["ab_scales/embedding/embedding/a"] = checkpoint.Tensor{Shape: []int{1}, Data: []float32{1}}
	bad.Tensors["ab_scales/embedding/embedding/b"] = checkpoint.Tensor{Shape: []int{1}, Data: []float32{1}}
	badModel, err = New(bad)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = badModel.SliceDepth(4); err == nil {
		t.Fatal("AB scales silently accepted")
	}
	if _, err = badModel.SliceDepth(6); err != nil {
		t.Fatalf("full-depth AB scales should pass: %v", err)
	}
}

func TestSliceDepthCustomNestingHeadsAndEngrams(t *testing.T) {
	m := makeLadderFixture(t, 6, []int{0, 5, 2, 4, 1, 3}, []int{1, 2, 4, 5}, []int{0, 3, 5})
	parent := m.Checkpoint()
	sliced, err := m.SliceDepth(4)
	if err != nil {
		t.Fatal(err)
	}
	cp := sliced.Checkpoint()
	cfg := decodeRawLadderConfig(t, cp.Config)
	if cfg.NumLayers != 4 || !slices.Equal(cfg.LadderOrder, []int{0, 3, 1, 2}) || !slices.Equal(cfg.GlobalLayers, []int{1, 2, 3}) || !slices.Equal(cfg.EngramLayers, []int{0, 3}) || len(cfg.LadderDepths) != 0 || cfg.LadderSample || cfg.EmbeddingProbes != 2 {
		t.Fatalf("raw config %+v", cfg)
	}
	if !slices.Equal(cp.Tensors["stack/layers/block/attn_gate"].Data, []float32{100, 102, 104, 105}) {
		t.Fatalf("attn_gate %v", cp.Tensors["stack/layers/block/attn_gate"].Data)
	}
	if !slices.Equal(cp.Tensors["stack/final_norm/scale"].Data, parent.Tensors["stack/final_norm/scale"].Data) {
		t.Fatal("final_norm changed")
	}
	if !slices.Equal(axisMarkers(cp.Tensors["embedding_head/probes"], 0), []float32{10, 11, 13, 15, 16}) {
		t.Fatalf("embedding probes rows %v", axisMarkers(cp.Tensors["embedding_head/probes"], 0))
	}
	if !slices.Equal(axisMarkers(cp.Tensors["embedding_head/gain"], 0), []float32{20, 21, 23, 25, 26}) {
		t.Fatalf("embedding gain rows %v", axisMarkers(cp.Tensors["embedding_head/gain"], 0))
	}
	if !slices.Equal(axisMarkers(cp.Tensors["embedding_head/row_bias"], 1), []float32{40, 41, 43, 45, 46}) {
		t.Fatalf("embedding row_bias rows %v", axisMarkers(cp.Tensors["embedding_head/row_bias"], 1))
	}
	if !slices.Equal(cp.Tensors["embedding_head/query"].Data, parent.Tensors["embedding_head/query"].Data) || !slices.Equal(cp.Tensors["embedding_head/proj/kernel"].Data, parent.Tensors["embedding_head/proj/kernel"].Data) || !slices.Equal(cp.Tensors["router_head/calibration"].Data, parent.Tensors["router_head/calibration"].Data) {
		t.Fatal("head tensors not preserved")
	}
	if _, ok := cp.Tensors["engrams_2/taps"]; ok {
		t.Fatal("unselected engram retained")
	}
	if got := cp.Tensors["engrams_0/taps"].Data[0]; got != 0 {
		t.Fatalf("engrams_0 remap=%g", got)
	}
	if got := cp.Tensors["engrams_1/taps"].Data[0]; got != 2000 {
		t.Fatalf("engrams_1 remap=%g", got)
	}
	if len(parent.Tensors["embedding_head/probes"].Shape) == 0 || parent.Tensors["embedding_head/probes"].Shape[0] != 7 || len(parent.Tensors["stack/layers/block/attn_gate"].Data) != 6 {
		t.Fatal("parent mutated")
	}
	if _, ok := parent.Tensors["engrams_2/taps"]; !ok {
		t.Fatal("parent engram missing")
	}
	nested, err := sliced.SliceDepth(2)
	if err != nil {
		t.Fatal(err)
	}
	nestedCP := nested.Checkpoint()
	nestedCfg := decodeRawLadderConfig(t, nestedCP.Config)
	if nestedCfg.NumLayers != 2 || !slices.Equal(nestedCfg.LadderOrder, []int{0, 1}) || !slices.Equal(nestedCP.Tensors["stack/layers/block/attn_gate"].Data, []float32{100, 105}) {
		t.Fatalf("nested cfg=%+v gate=%v", nestedCfg, nestedCP.Tensors["stack/layers/block/attn_gate"].Data)
	}
}

func TestSliceDepthSameDepthParity(t *testing.T) {
	m, f := fixture(t)
	sliced, err := m.SliceDepth(m.config.Layers)
	if err != nil {
		t.Fatal(err)
	}
	if sliced == m {
		t.Fatal("reused original model")
	}
	base, err := m.Forward(f.Tokens, Options{})
	if err != nil {
		t.Fatal(err)
	}
	got, err := sliced.Forward(f.Tokens, Options{})
	if err != nil {
		t.Fatal(err)
	}
	compare(t, "same-depth logits", got, base, 0, 0)
	cfg := decodeRawLadderConfig(t, sliced.Checkpoint().Config)
	if cfg.NumLayers != 2 || !slices.Equal(cfg.LadderOrder, []int{0, 1}) || len(cfg.LadderDepths) != 0 || cfg.LadderSample || cfg.EmbeddingProbes != 2 {
		t.Fatalf("same-depth raw config %+v", cfg)
	}
}
