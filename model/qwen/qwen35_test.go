package qwen

import (
	"math"
	"strings"
	"testing"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	"github.com/rcarmo/go-pherence/tensor"
)

func testQwen35BaseMeta() loaderconfig.QwenNativeMTPMetadata {
	return loaderconfig.QwenNativeMTPMetadata{
		HiddenSize:            4,
		IntermediateSize:      6,
		NumAttentionHeads:     2,
		NumKeyValueHeads:      1,
		HeadDim:               2,
		LinearConvKernelDim:   3,
		LinearKeyHeadDim:      2,
		LinearNumKeyHeads:     1,
		LinearNumValueHeads:   2,
		LinearValueHeadDim:    2,
		HasLinearAttention:    true,
		FullAttentionInterval: 4,
	}
}

func fullQwen35LayerSource(meta loaderconfig.QwenNativeMTPMetadata, prefix string) fakeQwen35TensorSource {
	shapes, _ := loaderconfig.Qwen35FullAttentionShapesFor(meta.HiddenSize, meta.NumAttentionHeads, meta.NumKeyValueHeads, meta.HeadDim)
	return fakeQwen35TensorSource{
		prefix + ".input_layernorm.weight":          tensor.Ones([]int{4}),
		prefix + ".post_attention_layernorm.weight": tensor.Ones([]int{4}),
		prefix + ".self_attn.q_proj.weight":         tensor.Zeros(shapes.QProj),
		prefix + ".self_attn.k_proj.weight":         tensor.Zeros(shapes.KProj),
		prefix + ".self_attn.v_proj.weight":         tensor.Zeros(shapes.VProj),
		prefix + ".self_attn.o_proj.weight":         tensor.Zeros(shapes.OProj),
		prefix + ".self_attn.q_norm.weight":         tensor.Ones(shapes.QNorm),
		prefix + ".self_attn.k_norm.weight":         tensor.Ones(shapes.KNorm),
		prefix + ".mlp.gate_proj.weight":            tensor.Zeros([]int{6, 4}),
		prefix + ".mlp.up_proj.weight":              tensor.Zeros([]int{6, 4}),
		prefix + ".mlp.down_proj.weight":            tensor.Zeros([]int{4, 6}),
	}
}

func TestAppendQwen35FullAttentionKV(t *testing.T) {
	meta := testQwen35BaseMeta()
	nextK, nextV, err := appendQwen35FullAttentionKV([]float32{1, 2}, []float32{3, 4}, []float32{5, 6}, []float32{7, 8}, meta)
	if err != nil {
		t.Fatalf("appendQwen35FullAttentionKV: %v", err)
	}
	if len(nextK) != 4 || len(nextV) != 4 || nextK[2] != 5 || nextV[3] != 8 {
		t.Fatalf("next K/V=%v/%v", nextK, nextV)
	}
	if _, _, err := appendQwen35FullAttentionKV([]float32{1}, []float32{1}, []float32{1, 2}, []float32{1, 2}, meta); err == nil {
		t.Fatal("bad past KV multiple returned nil error")
	}
	if _, _, err := appendQwen35FullAttentionKV(nil, nil, []float32{1}, []float32{1}, meta); err == nil {
		t.Fatal("bad current KV len returned nil error")
	}
}

func TestCloneQwen35BaseForwardState(t *testing.T) {
	state := Qwen35BaseForwardState{
		FullK:  [][]float32{{1, 2}},
		FullV:  [][]float32{{3, 4}},
		Linear: []Qwen35LinearAttentionState{{Conv: []float32{5}, SSM: []float32{6}, Pos: 7}},
		Pos:    8,
	}
	clone := CloneQwen35BaseForwardState(state)
	clone.FullK[0][0] = 10
	clone.FullV[0][0] = 11
	clone.Linear[0].Conv[0] = 12
	clone.Linear[0].SSM[0] = 13
	if state.FullK[0][0] != 1 || state.FullV[0][0] != 3 || state.Linear[0].Conv[0] != 5 || state.Linear[0].SSM[0] != 6 {
		t.Fatalf("clone aliased original: state=%+v clone=%+v", state, clone)
	}
}

func TestQwen35BaseModelForwardSequence(t *testing.T) {
	meta := testQwen35BaseMeta()
	meta.NumHiddenLayers = 1
	meta.MTPNumHiddenLayers = 0
	meta.LayerTypes = []string{"full_attention"}
	src := CandidateQwen35TensorSource{Source: fullQwen35LayerSource(meta, "model.layers.0")}
	base, err := LoadQwen35BaseModelLayers(src, meta)
	if err != nil {
		t.Fatal(err)
	}
	state, err := NewQwen35BaseForwardState(base, meta)
	if err != nil {
		t.Fatal(err)
	}
	outs, next, err := base.ForwardSequence([][]float32{{1, 0, 0, 0}, {0, 1, 0, 0}}, state, nil, 1e-6, meta)
	if err != nil {
		t.Fatalf("ForwardSequence: %v", err)
	}
	wantKV := 2 * meta.NumKeyValueHeads * meta.HeadDim
	if len(outs) != 2 || next.Pos != 2 || len(next.FullK[0]) != wantKV || len(state.FullK[0]) != 0 {
		t.Fatalf("outs=%v state=%+v next=%+v", outs, state, next)
	}
}

func TestQwen35BaseModelForwardOneDoesNotMutateInputState(t *testing.T) {
	meta := testQwen35BaseMeta()
	meta.NumHiddenLayers = 1
	meta.MTPNumHiddenLayers = 0
	meta.LayerTypes = []string{"full_attention"}
	src := CandidateQwen35TensorSource{Source: fullQwen35LayerSource(meta, "model.layers.0")}
	base, err := LoadQwen35BaseModelLayers(src, meta)
	if err != nil {
		t.Fatal(err)
	}
	state, err := NewQwen35BaseForwardState(base, meta)
	if err != nil {
		t.Fatal(err)
	}
	_, next, err := base.ForwardOne([]float32{1, 0, 0, 0}, state, 0, nil, 1e-6, meta)
	if err != nil {
		t.Fatal(err)
	}
	if len(state.FullK[0]) != 0 || len(next.FullK[0]) == 0 {
		t.Fatalf("state mutated or next empty: state=%+v next=%+v", state, next)
	}
}

func TestQwen35BaseModelForwardOneFullAttention(t *testing.T) {
	meta := testQwen35BaseMeta()
	meta.NumHiddenLayers = 1
	meta.MTPNumHiddenLayers = 0
	meta.LayerTypes = []string{"full_attention"}
	src := CandidateQwen35TensorSource{Source: fullQwen35LayerSource(meta, "model.layers.0")}
	base, err := LoadQwen35BaseModelLayers(src, meta)
	if err != nil {
		t.Fatal(err)
	}
	state, err := NewQwen35BaseForwardState(base, meta)
	if err != nil {
		t.Fatal(err)
	}
	out, next, err := base.ForwardOne([]float32{1, 0, 0, 0}, state, 0, nil, 1e-6, meta)
	if err != nil {
		t.Fatalf("ForwardOne: %v", err)
	}
	if len(out) != meta.HiddenSize || next.Pos != 1 || len(next.FullK[0]) != meta.NumKeyValueHeads*meta.HeadDim {
		t.Fatalf("out=%v next=%+v", out, next)
	}
}

func TestL2NormalizeHeadsInPlace(t *testing.T) {
	x := []float32{3, 4, 5, 12}
	if err := l2NormalizeHeadsInPlace(x, 2, 2, 0); err != nil {
		t.Fatalf("l2NormalizeHeadsInPlace: %v", err)
	}
	want := []float32{0.6, 0.8, 5.0 / 13.0, 12.0 / 13.0}
	for i := range want {
		if math.Abs(float64(x[i]-want[i])) > 1e-6 {
			t.Fatalf("head-normalized vector=%v want %v", x, want)
		}
	}
	if err := l2NormalizeHeadsInPlace([]float32{1, 2, 3}, 2, 2, 0); err == nil {
		t.Fatal("bad head dimensions returned nil error")
	}
}

func TestQwen35BaseModelForwardOneLinearAttention(t *testing.T) {
	meta := testQwen35BaseMeta()
	meta.NumHiddenLayers = 1
	meta.MTPNumHiddenLayers = 0
	meta.LayerTypes = []string{"linear_attention"}
	src := CandidateQwen35TensorSource{Source: linearQwen35LayerSource(meta, "model.layers.0")}
	base, err := LoadQwen35BaseModelLayers(src, meta)
	if err != nil {
		t.Fatal(err)
	}
	state, err := NewQwen35BaseForwardState(base, meta)
	if err != nil {
		t.Fatal(err)
	}
	out, next, err := base.ForwardOne([]float32{1, 0, 0, 0}, state, 0, nil, 1e-6, meta)
	if err != nil {
		t.Fatalf("ForwardOne: %v", err)
	}
	if len(out) != meta.HiddenSize || next.Pos != 1 || next.Linear[0].Pos != 1 {
		t.Fatalf("out=%v next=%+v", out, next)
	}
}

func TestLoadQwen35BaseModelLayers(t *testing.T) {
	meta := testQwen35BaseMeta()
	meta.NumHiddenLayers = 2
	meta.MTPNumHiddenLayers = 0
	meta.LayerTypes = []string{"linear_attention", "full_attention"}
	src := fakeQwen35TensorSource{}
	for k, v := range linearQwen35LayerSource(meta, "model.language_model.model.layers.0") {
		src[k] = v
	}
	for k, v := range fullQwen35LayerSource(meta, "model.language_model.model.layers.1") {
		src[k] = v
	}
	model, err := LoadQwen35BaseModelLayers(CandidateQwen35TensorSource{Source: src}, meta)
	if err != nil {
		t.Fatalf("LoadQwen35BaseModelLayers: %v", err)
	}
	if len(model.Layers) != 2 || model.Layers[0].Kind != Qwen35LinearAttentionLayerKind || model.Layers[1].Kind != Qwen35FullAttentionLayerKind {
		t.Fatalf("layers=%+v", model.Layers)
	}
}

func TestLoadQwen35FullAttentionLayer(t *testing.T) {
	meta := testQwen35BaseMeta()
	src := CandidateQwen35TensorSource{Source: fullQwen35LayerSource(meta, "model.language_model.model.layers.0")}
	l, err := LoadQwen35FullAttentionLayer(src, meta, "model.layers.0")
	if err != nil {
		t.Fatalf("LoadQwen35FullAttentionLayer: %v", err)
	}
	if l.QW == nil || l.GateW == nil {
		t.Fatalf("loaded layer=%+v", l)
	}
}

func TestQwen35FullAttentionLayerForward(t *testing.T) {
	meta := testQwen35BaseMeta()
	src := CandidateQwen35TensorSource{Source: fullQwen35LayerSource(meta, "model.layers.0")}
	l, err := LoadQwen35FullAttentionLayer(src, meta, "model.layers.0")
	if err != nil {
		t.Fatal(err)
	}
	out, curK, curV, err := l.ForwardWithKV([]float32{1, 0, 0, 0}, 0, nil, nil, nil, 1e-6, meta)
	if err != nil {
		t.Fatalf("ForwardWithKV: %v", err)
	}
	if len(out) != meta.HiddenSize || len(curK) != meta.NumKeyValueHeads*meta.HeadDim || len(curV) != meta.NumKeyValueHeads*meta.HeadDim {
		t.Fatalf("out/K/V lens=%d/%d/%d", len(out), len(curK), len(curV))
	}
	if out[0] == 0 {
		t.Fatalf("expected residual output, got %v", out)
	}
}

func TestValidateQwen35FullAttentionLayer(t *testing.T) {
	meta := testQwen35BaseMeta()
	shapes, err := loaderconfig.Qwen35FullAttentionShapesFor(meta.HiddenSize, meta.NumAttentionHeads, meta.NumKeyValueHeads, meta.HeadDim)
	if err != nil {
		t.Fatal(err)
	}
	l := &Qwen35FullAttentionLayer{
		InputNorm: tensor.Ones([]int{4}), PostNorm: tensor.Ones([]int{4}),
		QW: tensor.Zeros(shapes.QProj), KW: tensor.Zeros(shapes.KProj), VW: tensor.Zeros(shapes.VProj), OW: tensor.Zeros(shapes.OProj),
		QNorm: tensor.Ones(shapes.QNorm), KNorm: tensor.Ones(shapes.KNorm),
		GateW: tensor.Zeros([]int{6, 4}), UpW: tensor.Zeros([]int{6, 4}), DownW: tensor.Zeros([]int{4, 6}),
	}
	if err := ValidateQwen35FullAttentionLayer(l, meta, "model.layers.0"); err != nil {
		t.Fatalf("ValidateQwen35FullAttentionLayer: %v", err)
	}
	l.QW = tensor.Zeros([]int{1, 4})
	if err := ValidateQwen35FullAttentionLayer(l, meta, "model.layers.0"); err == nil || !strings.Contains(err.Error(), "q_proj") {
		t.Fatalf("bad q_proj error=%v", err)
	}
}

func linearQwen35LayerSource(meta loaderconfig.QwenNativeMTPMetadata, prefix string) fakeQwen35TensorSource {
	shapes, _ := qwen35LinearAttentionShapesFromMeta(meta)
	return fakeQwen35TensorSource{
		prefix + ".input_layernorm.weight":          tensor.Ones([]int{4}),
		prefix + ".post_attention_layernorm.weight": tensor.Ones([]int{4}),
		prefix + ".linear_attn.in_proj_qkv.weight":  tensor.Zeros(shapes.QKV),
		prefix + ".linear_attn.in_proj_z.weight":    tensor.Zeros(shapes.Gate),
		prefix + ".linear_attn.conv1d.weight":       tensor.Zeros([]int{shapes.ConvDim, meta.LinearConvKernelDim, 1}),
		prefix + ".linear_attn.dt_bias":             tensor.Zeros(shapes.DTBias),
		prefix + ".linear_attn.A":                   tensor.Zeros(shapes.A),
		prefix + ".linear_attn.in_proj_b.weight":    tensor.Zeros(shapes.Beta),
		prefix + ".linear_attn.in_proj_a.weight":    tensor.Zeros(shapes.Alpha),
		prefix + ".linear_attn.norm.weight":         tensor.Ones(shapes.Norm),
		prefix + ".linear_attn.out_proj.weight":     tensor.Zeros(shapes.Out),
		prefix + ".mlp.gate_proj.weight":            tensor.Zeros([]int{6, 4}),
		prefix + ".mlp.up_proj.weight":              tensor.Zeros([]int{6, 4}),
		prefix + ".mlp.down_proj.weight":            tensor.Zeros([]int{4, 6}),
	}
}

func TestQwen35ApplyLoRA(t *testing.T) {
	meta := testQwen35BaseMeta()
	meta.NumHiddenLayers = 1
	meta.MTPNumHiddenLayers = 0
	meta.LayerTypes = []string{"full_attention"}
	src := CandidateQwen35TensorSource{Source: fullQwen35LayerSource(meta, "model.layers.0")}
	m, e := LoadQwen35BaseModelLayers(src, meta)
	if e != nil {
		t.Fatal(e)
	}
	rank := 2
	a := tensor.FromFloat32([]float32{1, 2, 3, 4, 5, 6, 7, 8}, []int{rank, 4})
	b := tensor.FromFloat32([]float32{1, 0, 0, 1, 1, 1, 2, 1, 1, 2, 2, 2, 3, 2, 2, 3}, []int{8, rank})
	if e = m.ApplyLoRA(Qwen35LoRASet{"model.layers.0.self_attn.q_proj": {A: a, B: b, Scale: .5}}); e != nil {
		t.Fatal(e)
	}
	w := m.Layers[0].Full.QW.Data()
	if w[0] != .5*(1*1+0*5) || w[4] != .5*(0*1+1*5) {
		t.Fatalf("merged=%v", w[:8])
	}
	if e = m.ApplyLoRA(Qwen35LoRASet{"bad": {A: a, B: b, Scale: 1}}); e == nil {
		t.Fatal("bad target accepted")
	}
	if e = m.ApplyLoRA(Qwen35LoRASet{"model.layers.0.self_attn.q_proj": {A: tensor.Zeros([]int{1, 3}), B: b, Scale: 1}}); e == nil {
		t.Fatal("bad shape accepted")
	}
}

func TestLoadQwen35LinearAttentionLayerHFLinearLayout(t *testing.T) {
	meta := testQwen35BaseMeta()
	prefix := "model.language_model.model.layers.1"
	src := linearQwen35LayerSource(meta, prefix)
	shapes, _ := qwen35LinearAttentionShapesFromMeta(meta)
	for name, shape := range map[string][]int{".linear_attn.in_proj_qkv.weight": shapes.QKV, ".linear_attn.in_proj_z.weight": shapes.Gate, ".linear_attn.in_proj_b.weight": shapes.Beta, ".linear_attn.in_proj_a.weight": shapes.Alpha, ".linear_attn.out_proj.weight": shapes.Out} {
		key := prefix + name
		old := src[key]
		data := append([]float32(nil), old.Data()...)
		for i := range data {
			data[i] = float32(i+1) / 100
		}
		src[key] = tensor.FromFloat32(data, []int{shape[1], shape[0]})
	}
	l, err := LoadQwen35LinearAttentionLayer(CandidateQwen35TensorSource{Source: src}, meta, "model.layers.1")
	if err != nil {
		t.Fatal(err)
	}
	if got := l.QKVW.Shape(); got[0] != shapes.QKV[0] || got[1] != shapes.QKV[1] {
		t.Fatalf("logical shape=%v want=%v", got, shapes.QKV)
	}
	out := make([]float32, shapes.QKV[1])
	if err := qwen35LinearInto(out, []float32{1, 2, 3, 4}, l.QKVW, nil, nil, meta.HiddenSize, shapes.QKV[1], "test"); err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(out[0]-.3)) > 1e-6 || math.Abs(float64(out[1]-.7)) > 1e-6 {
		t.Fatalf("row-major projection=%v", out)
	}
}

func TestLoadQwen35LinearAttentionLayer(t *testing.T) {
	meta := testQwen35BaseMeta()
	src := CandidateQwen35TensorSource{Source: linearQwen35LayerSource(meta, "model.language_model.model.layers.1")}
	l, err := LoadQwen35LinearAttentionLayer(src, meta, "model.layers.1")
	if err != nil {
		t.Fatalf("LoadQwen35LinearAttentionLayer: %v", err)
	}
	if l.QKVW == nil || l.OutW == nil {
		t.Fatalf("loaded layer=%+v", l)
	}
}

func TestLoadQwen35LinearAttentionLayerConvertsALog(t *testing.T) {
	meta := testQwen35BaseMeta()
	src := linearQwen35LayerSource(meta, "model.language_model.model.layers.1")
	delete(src, "model.language_model.model.layers.1.linear_attn.A")
	src["model.language_model.model.layers.1.linear_attn.A_log"] = tensor.FromFloat32([]float32{0, 1}, []int{2})
	l, err := LoadQwen35LinearAttentionLayer(CandidateQwen35TensorSource{Source: src}, meta, "model.layers.1")
	if err != nil {
		t.Fatalf("LoadQwen35LinearAttentionLayer: %v", err)
	}
	got := l.A.Data()
	if math.Abs(float64(got[0]+1)) > 1e-6 || math.Abs(float64(got[1]+float32(math.E))) > 1e-6 {
		t.Fatalf("converted A=%v, want [-1 -e]", got)
	}
}

func TestSplitQwen35LinearQKVRaw(t *testing.T) {
	meta := testQwen35BaseMeta()
	shapes, err := qwen35LinearAttentionShapesFromMeta(meta)
	if err != nil {
		t.Fatal(err)
	}
	projected := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	parts, err := splitQwen35LinearQKVRaw(projected, shapes)
	if err != nil {
		t.Fatalf("splitQwen35LinearQKVRaw: %v", err)
	}
	check := func(name string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s len=%d want %d (%v)", name, len(got), len(want), got)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s[%d]=%v want %v (%v)", name, i, got[i], want[i], got)
			}
		}
	}
	check("Q", parts.Q, []float32{1, 2})
	check("K", parts.K, []float32{3, 4})
	check("V", parts.V, []float32{5, 6, 7, 8})
	if _, err := splitQwen35LinearQKVRaw(projected[:len(projected)-1], shapes); err == nil {
		t.Fatal("bad raw projected length returned nil error")
	}
}

func TestSplitQwen35LinearQKV(t *testing.T) {
	check := func(t *testing.T, name string, got, want []float32) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s len=%d want %d (%v)", name, len(got), len(want), got)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s[%d]=%v want %v (%v)", name, i, got[i], want[i], got)
			}
		}
	}

	t.Run("expands grouped key heads", func(t *testing.T) {
		meta := testQwen35BaseMeta()
		shapes, err := qwen35LinearAttentionShapesFromMeta(meta)
		if err != nil {
			t.Fatal(err)
		}
		projected := []float32{1, 2, 3, 4, 5, 6, 7, 8}
		parts, err := splitQwen35LinearQKV(projected, shapes)
		if err != nil {
			t.Fatalf("splitQwen35LinearQKV: %v", err)
		}
		check(t, "Q", parts.Q, []float32{1, 2, 1, 2})
		check(t, "K", parts.K, []float32{3, 4, 3, 4})
		check(t, "V", parts.V, []float32{5, 6, 7, 8})
	})

	t.Run("keeps one to one head order", func(t *testing.T) {
		shapes := loaderconfig.Qwen35LinearAttentionShapes{KeyDim: 4, ValueDim: 4, HeadVDim: 2}
		projected := []float32{
			1, 2, 3, 4, 5, 6,
			7, 8, 9, 10, 11, 12,
		}
		parts, err := splitQwen35LinearQKV(projected, shapes)
		if err != nil {
			t.Fatalf("splitQwen35LinearQKV: %v", err)
		}
		check(t, "Q", parts.Q, []float32{1, 2, 7, 8})
		check(t, "K", parts.K, []float32{3, 4, 9, 10})
		check(t, "V", parts.V, []float32{5, 6, 11, 12})
	})

	t.Run("rejects bad projected length", func(t *testing.T) {
		shapes := loaderconfig.Qwen35LinearAttentionShapes{KeyDim: 2, ValueDim: 4, HeadVDim: 2}
		if _, err := splitQwen35LinearQKV(make([]float32, 7), shapes); err == nil {
			t.Fatal("bad projected length returned nil error")
		}
	})

	t.Run("rejects invalid dimensions", func(t *testing.T) {
		badShapes := []loaderconfig.Qwen35LinearAttentionShapes{
			{KeyDim: 2, ValueDim: 4, HeadVDim: 0},
			{KeyDim: 3, ValueDim: 4, HeadVDim: 2},
			{KeyDim: 4, ValueDim: 5, HeadVDim: 2},
			{KeyDim: 4, ValueDim: 6, HeadVDim: 2},
		}
		for _, shapes := range badShapes {
			projected := make([]float32, shapes.KeyDim*2+shapes.ValueDim)
			if _, err := splitQwen35LinearQKV(projected, shapes); err == nil {
				t.Fatalf("splitQwen35LinearQKV(%+v) returned nil error", shapes)
			}
		}
	})
}

func TestApplyQwen35LinearDeltaUpdateUsesTiledKeyHeads(t *testing.T) {
	meta := testQwen35BaseMeta()
	shapes, err := qwen35LinearAttentionShapesFromMeta(meta)
	if err != nil {
		t.Fatal(err)
	}
	state, err := NewQwen35LinearAttentionState(meta)
	if err != nil {
		t.Fatal(err)
	}
	q := []float32{1, 2}
	k := []float32{3, 4}
	v := []float32{1, 1, 1, 1}
	next, out, err := applyQwen35LinearDeltaUpdate(state.SSM, q, k, v, []float32{1, 1}, []float32{1, 1}, []float32{0, 0}, shapes, meta)
	if err != nil {
		t.Fatalf("applyQwen35LinearDeltaUpdate: %v", err)
	}
	if len(next) != len(state.SSM) || len(out) != shapes.ValueDim {
		t.Fatalf("next/out=%d/%v", len(next), out)
	}
	want := float32(5.5 * math.Sqrt(2))
	for i, got := range out {
		if math.Abs(float64(got-want)) > 1e-6 {
			t.Fatalf("out[%d]=%v want %v; out=%v", i, got, want, out)
		}
	}
}

func TestApplyQwen35LinearDeltaUpdate(t *testing.T) {
	meta := testQwen35BaseMeta()
	shapes, err := qwen35LinearAttentionShapesFromMeta(meta)
	if err != nil {
		t.Fatal(err)
	}
	state, err := NewQwen35LinearAttentionState(meta)
	if err != nil {
		t.Fatal(err)
	}
	q := make([]float32, shapes.KeyDim)
	k := make([]float32, shapes.KeyDim)
	v := make([]float32, shapes.ValueDim)
	for i := range q {
		q[i] = 1
	}
	for i := range k {
		k[i] = 1
	}
	for i := range v {
		v[i] = 1
	}
	next, out, err := applyQwen35LinearDeltaUpdate(state.SSM, q, k, v, []float32{1, 1}, []float32{1, 1}, []float32{0.5, 0.5}, shapes, meta)
	if err != nil {
		t.Fatalf("applyQwen35LinearDeltaUpdate: %v", err)
	}
	if len(next) != len(state.SSM) || len(out) != shapes.ValueDim || out[0] == 0 {
		t.Fatalf("next/out=%d/%v", len(next), out)
	}
	if _, _, err := applyQwen35LinearDeltaUpdate(state.SSM[:len(state.SSM)-1], q, k, v, []float32{1, 1}, []float32{1, 1}, []float32{0.5, 0.5}, shapes, meta); err == nil {
		t.Fatal("bad SSM state len returned nil error")
	}
}

func TestPrepareQwen35LinearDeltaParams(t *testing.T) {
	dt, decay, err := prepareQwen35LinearDeltaParams([]float32{0, 1}, []float32{2, 3}, []float32{0, -1}, []float32{-1, -2}, 2)
	if err != nil {
		t.Fatalf("prepareQwen35LinearDeltaParams: %v", err)
	}
	if len(dt) != 2 || len(decay) != 2 || dt[0] <= 0 || decay[0] <= 0 || decay[0] > 1 {
		t.Fatalf("dt/decay=%v/%v", dt, decay)
	}
	if _, _, err := prepareQwen35LinearDeltaParams([]float32{1}, []float32{1}, []float32{1}, []float32{1}, 2); err == nil {
		t.Fatal("bad lengths returned nil error")
	}
}

func TestProjectQwen35LinearAlphaBeta(t *testing.T) {
	input := []float32{1, 2, 3}
	alphaW := []float32{1, 10, 2, 20, 3, 30}
	betaW := []float32{4, 40, 5, 50, 6, 60}
	alpha, beta, err := projectQwen35LinearAlphaBeta(input, alphaW, betaW, 3, 2)
	if err != nil {
		t.Fatalf("projectQwen35LinearAlphaBeta: %v", err)
	}
	if len(alpha) != 2 || alpha[0] != 27 || alpha[1] != 116 || beta[0] != 99 || beta[1] != 242 {
		t.Fatalf("alpha/beta=%v/%v", alpha, beta)
	}
	if _, _, err := projectQwen35LinearAlphaBeta(input[:2], alphaW, betaW, 3, 2); err == nil {
		t.Fatal("bad input len returned nil error")
	}
	if _, _, err := projectQwen35LinearAlphaBeta(input, alphaW[:5], betaW, 3, 2); err == nil {
		t.Fatal("bad weight len returned nil error")
	}
}

func TestApplyQwen35LinearDepthwiseConv(t *testing.T) {
	out, err := applyQwen35LinearDepthwiseConv(
		[]float32{1, 2, 3, 4, 5, 6},
		[]float32{1, 10, 1, 10, 1, 10},
		2,
		3,
	)
	if err != nil {
		t.Fatalf("applyQwen35LinearDepthwiseConv: %v", err)
	}
	if len(out) != 2 || out[0] != 9 || out[1] != 120 {
		t.Fatalf("out=%v", out)
	}
	if _, err := applyQwen35LinearDepthwiseConv([]float32{1}, []float32{1}, 0, 1); err == nil {
		t.Fatal("bad dims returned nil error")
	}
	if _, err := applyQwen35LinearDepthwiseConv([]float32{1, 2}, []float32{1}, 1, 2); err == nil {
		t.Fatal("bad weight len returned nil error")
	}
}

func TestUpdateQwen35LinearConvState(t *testing.T) {
	next, err := updateQwen35LinearConvState([]float32{1, 2, 3, 4, 5, 6}, []float32{7, 8}, 3)
	if err != nil {
		t.Fatalf("updateQwen35LinearConvState: %v", err)
	}
	want := []float32{3, 4, 5, 6, 7, 8}
	for i := range want {
		if next[i] != want[i] {
			t.Fatalf("next=%v want %v", next, want)
		}
	}
	if _, err := updateQwen35LinearConvState([]float32{1, 2}, []float32{3, 4}, 0); err == nil {
		t.Fatal("bad kernel returned nil error")
	}
	if _, err := updateQwen35LinearConvState([]float32{1, 2}, []float32{3, 4}, 2); err == nil {
		t.Fatal("bad state len returned nil error")
	}
}

func TestNewQwen35LinearAttentionState(t *testing.T) {
	meta := testQwen35BaseMeta()
	state, err := NewQwen35LinearAttentionState(meta)
	if err != nil {
		t.Fatalf("NewQwen35LinearAttentionState: %v", err)
	}
	shapes, _ := qwen35LinearAttentionShapesFromMeta(meta)
	if len(state.Conv) != shapes.ConvDim*meta.LinearConvKernelDim {
		t.Fatalf("conv len=%d", len(state.Conv))
	}
	wantSSM := meta.LinearNumValueHeads * meta.LinearValueHeadDim * meta.LinearNumKeyHeads * meta.LinearKeyHeadDim
	if len(state.SSM) != wantSSM {
		t.Fatalf("ssm len=%d want %d", len(state.SSM), wantSSM)
	}
}

func TestQwen35LinearAttentionForward(t *testing.T) {
	meta := testQwen35BaseMeta()
	src := CandidateQwen35TensorSource{Source: linearQwen35LayerSource(meta, "model.layers.1")}
	l, err := LoadQwen35LinearAttentionLayer(src, meta, "model.layers.1")
	if err != nil {
		t.Fatal(err)
	}
	state, err := NewQwen35LinearAttentionState(meta)
	if err != nil {
		t.Fatal(err)
	}
	out, next, err := l.ForwardWithState([]float32{1, 0, 0, 0}, state, 1e-6, meta)
	if err != nil {
		t.Fatalf("ForwardWithState: %v", err)
	}
	if len(out) != meta.HiddenSize || len(next.Conv) != len(state.Conv) || len(next.SSM) != len(state.SSM) || next.Pos != 1 {
		t.Fatalf("out=%v next=%+v", out, next)
	}
}

func TestValidateQwen35LinearAttentionLayer(t *testing.T) {
	meta := testQwen35BaseMeta()
	shapes, err := qwen35LinearAttentionShapesFromMeta(meta)
	if err != nil {
		t.Fatal(err)
	}
	l := &Qwen35LinearAttentionLayer{
		InputNorm: tensor.Ones([]int{4}), PostNorm: tensor.Ones([]int{4}),
		QKVW: tensor.Zeros(shapes.QKV), GateW: tensor.Zeros(shapes.Gate), Conv1D: tensor.Zeros([]int{shapes.ConvDim, 1, meta.LinearConvKernelDim}),
		DTBias: tensor.Zeros(shapes.DTBias), A: tensor.Zeros(shapes.A), BetaW: tensor.Zeros(shapes.Beta), AlphaW: tensor.Zeros(shapes.Alpha),
		Norm: tensor.Ones(shapes.Norm), OutW: tensor.Zeros(shapes.Out),
		MLPGateW: tensor.Zeros([]int{6, 4}), MLPUpW: tensor.Zeros([]int{6, 4}), MLPDownW: tensor.Zeros([]int{4, 6}),
	}
	if err := ValidateQwen35LinearAttentionLayer(l, meta, "model.layers.1"); err != nil {
		t.Fatalf("ValidateQwen35LinearAttentionLayer: %v", err)
	}
	l.Conv1D = tensor.Zeros([]int{1, shapes.ConvDim})
	if err := ValidateQwen35LinearAttentionLayer(l, meta, "model.layers.1"); err == nil || !strings.Contains(err.Error(), "conv1d") {
		t.Fatalf("bad conv1d error=%v", err)
	}
}
