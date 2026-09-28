package qwen

import (
	"math"
	"reflect"
	"sync"
	"testing"
)

func TestQwen35TextBranchVisibility(t *testing.T) {
	meta := testQwen35BaseMeta()
	meta.NumHiddenLayers = 1
	meta.MTPNumHiddenLayers = 0
	meta.LayerTypes = []string{"full_attention"}
	meta.BF16Trajectory = false
	base, err := LoadQwen35BaseModelLayers(CandidateQwen35TensorSource{Source: fullQwen35LayerSource(meta, "model.layers.0")}, meta)
	if err != nil {
		t.Fatal(err)
	}
	// Nonzero projections make visibility errors observable (the shared
	// loader fixture intentionally has zero attention projections).
	for _, data := range [][]float32{base.Layers[0].Full.QW.Data(), base.Layers[0].Full.KW.Data(), base.Layers[0].Full.VW.Data(), base.Layers[0].Full.OW.Data()} {
		for i := range data {
			data[i] = float32((i*7)%13-6) * 0.09
		}
	}
	input := [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}, {1, 1, 0, 0}, {0, 0, 1, 1}}
	before := make([][]float32, len(input))
	for i := range input {
		before[i] = append([]float32(nil), input[i]...)
	}
	pos := []int{0, 1, 2, 3, 4, 5}
	out, err := base.ForwardTextBranch(input, pos, 2, 2, nil, 1e-6, meta)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(input, before) {
		t.Fatal("inputs mutated")
	}
	changed := make([][]float32, len(input))
	copy(changed, input)
	changed[5] = []float32{2, 3, 1, 4}
	next, err := base.ForwardTextBranch(changed, pos, 2, 2, nil, 1e-6, meta)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(out[:4], next[:4]) {
		t.Fatal("candidate reached ancestor")
	}
	if reflect.DeepEqual(out[4], next[4]) {
		t.Fatal("full attention cannot see later token in own node")
	}
	changed[3] = []float32{3, 1, 2, 4}
	next, err = base.ForwardTextBranch(changed, pos, 2, 2, nil, 1e-6, meta)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(out[:2], next[:2]) {
		t.Fatal("question reached state")
	}
	if reflect.DeepEqual(out[2], next[2]) {
		t.Fatal("question not bidirectional")
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			again, e := base.ForwardTextBranch(input, pos, 2, 2, nil, 1e-6, meta)
			if e != nil || !reflect.DeepEqual(out, again) {
				t.Error("shared branch state", e)
			}
		}()
	}
	wg.Wait()
	for _, bad := range [][]int{{0, 1, 2}, {0, 1, 2, 3, 4, -1}, {0, 1, 2, 3, 4, 4096}, {0, 1, 2, 3, 4, 4}} {
		if got, e := base.ForwardTextBranch(input, bad, 2, 2, nil, 1e-6, meta); e == nil || got != nil {
			t.Fatal("bad positions accepted")
		}
	}
	for _, lens := range [][2]int{{0, 2}, {2, 0}, {6, 1}, {1, 5}, {math.MaxInt, 2}} {
		if got, e := base.ForwardTextBranch(input, pos, lens[0], lens[1], nil, 1e-6, meta); e == nil || got != nil {
			t.Fatal("bad nodes accepted")
		}
	}
	if got, e := base.ForwardTextBranch(input, pos, 2, 2, []float32{1}, 1e-6, meta); e == nil || got != nil {
		t.Fatal("short RoPE accepted")
	}
	meta.BF16Trajectory = true
	if got, e := base.ForwardTextBranch(input, pos, 2, 2, nil, 1e-6, meta); e == nil || got != nil {
		t.Fatal("unvalidated BF16 accepted")
	}
	meta.BF16Trajectory = false
	changed[0] = []float32{float32(math.NaN()), 0, 0, 0}
	if got, e := base.ForwardTextBranch(changed, pos, 2, 2, nil, 1e-6, meta); e == nil || got != nil {
		t.Fatal("nonfinite input accepted")
	}
	base.Layers[0].Kind = "invalid"
	if got, e := base.ForwardTextBranch(input, pos, 2, 2, nil, 1e-6, meta); e == nil || got != nil {
		t.Fatal("bad layer accepted")
	}
}

func TestQwen35TextBranchLinearFreshState(t *testing.T) {
	meta := testQwen35BaseMeta()
	meta.NumHiddenLayers = 1
	meta.MTPNumHiddenLayers = 0
	meta.LayerTypes = []string{"linear_attention"}
	meta.BF16Trajectory = false
	base, err := LoadQwen35BaseModelLayers(CandidateQwen35TensorSource{Source: linearQwen35LayerSource(meta, "model.layers.0")}, meta)
	if err != nil {
		t.Fatal(err)
	}
	input := [][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}
	pos := []int{0, 1, 2}
	out, err := base.ForwardTextBranch(input, pos, 1, 1, nil, 1e-6, meta)
	if err != nil {
		t.Fatal(err)
	}
	input[2][0] = 9
	changed, err := base.ForwardTextBranch(input, pos, 1, 1, nil, 1e-6, meta)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(out[:2], changed[:2]) {
		t.Fatal("linear candidate reached ancestors")
	}
	input[2][0] = 0
	again, err := base.ForwardTextBranch(input, pos, 1, 1, nil, 1e-6, meta)
	if err != nil || !reflect.DeepEqual(out, again) {
		t.Fatal("recurrent state retained", err)
	}
}
