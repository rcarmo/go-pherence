package qwen

import (
	"reflect"
	"testing"

	loaderconfig "github.com/rcarmo/go-pherence/loader/config"
	basemodel "github.com/rcarmo/go-pherence/model"
	"github.com/rcarmo/go-pherence/tensor"
)

func TestQwen35BaseModelForwardPrefillLayerStreamedSpansMatchesSequential(t *testing.T) {
	base, meta := buildQwen35PrefillStreamedTestModel(t)
	inputs := qwen35PrefillStreamedInputs(130, meta.HiddenSize)
	rope := NewQwen35RoPEFreqs(meta, len(inputs)+1)
	state, err := NewQwen35BaseForwardState(base, meta)
	if err != nil {
		t.Fatal(err)
	}
	wantOuts, wantPrefix, wantFinal := qwen35ForwardOneReference(t, base, inputs, state, rope, 1e-6, meta)
	seqOuts, seqFinal, err := base.ForwardSequence(inputs, state, rope, 1e-6, meta)
	if err != nil {
		t.Fatalf("ForwardSequence: %v", err)
	}
	if !reflect.DeepEqual(seqOuts, wantOuts) {
		t.Fatal("ForwardSequence outputs differ from ForwardOne reference")
	}
	if !reflect.DeepEqual(seqFinal, wantFinal) {
		t.Fatal("ForwardSequence final state differs from ForwardOne reference")
	}
	dims := basemodel.PrefillChunkModelDims{
		HiddenSize:   meta.HiddenSize,
		QDim:         meta.NumAttentionHeads * meta.HeadDim,
		KVDim:        meta.NumKeyValueHeads * meta.HeadDim,
		Intermediate: meta.IntermediateSize,
		Layers:       len(base.Layers),
	}
	estimate, err := basemodel.EstimatePrefillChunkScratch(dims)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name       string
		wantChunk  int
		buildSpans func() []struct {
			Start int
			End   int
		}
	}{
		{
			name:      "full_chunk",
			wantChunk: len(inputs),
			buildSpans: func() []struct {
				Start int
				End   int
			} {
				return []struct {
					Start int
					End   int
				}{{Start: 0, End: len(inputs)}}
			},
		},
		{
			name:      "planner_32",
			wantChunk: 32,
			buildSpans: func() []struct {
				Start int
				End   int
			} {
				return qwen35PrefillPlanStructSpans(t, len(inputs), dims, estimate, 32)
			},
		},
		{
			name:      "planner_64",
			wantChunk: 64,
			buildSpans: func() []struct {
				Start int
				End   int
			} {
				return qwen35PrefillPlanStructSpans(t, len(inputs), dims, estimate, 64)
			},
		},
		{
			name:      "planner_128",
			wantChunk: 128,
			buildSpans: func() []struct {
				Start int
				End   int
			} {
				return qwen35PrefillPlanStructSpans(t, len(inputs), dims, estimate, 128)
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotOuts, gotPrefix, gotFinal, err := base.ForwardPrefillLayerStreamedSpans(inputs, state, rope, 1e-6, meta, tc.buildSpans())
			if err != nil {
				t.Fatalf("ForwardPrefillLayerStreamedSpans: %v", err)
			}
			if !reflect.DeepEqual(gotOuts, wantOuts) {
				t.Fatalf("outputs mismatch for chunk %d", tc.wantChunk)
			}
			if !reflect.DeepEqual(gotPrefix, wantPrefix) {
				t.Fatalf("prefix states mismatch for chunk %d", tc.wantChunk)
			}
			if !reflect.DeepEqual(gotFinal, wantFinal) {
				t.Fatalf("final state mismatch for chunk %d", tc.wantChunk)
			}
		})
	}
}

func TestQwen35BaseModelForwardPrefillLayerStreamedSpansRejectsMalformedSpans(t *testing.T) {
	base, meta := buildQwen35PrefillStreamedTestModel(t)
	inputs := qwen35PrefillStreamedInputs(4, meta.HiddenSize)
	state, err := NewQwen35BaseForwardState(base, meta)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name  string
		spans []struct {
			Start int
			End   int
		}
	}{
		{name: "empty", spans: nil},
		{name: "starts_after_zero", spans: []struct {
			Start int
			End   int
		}{{Start: 1, End: 4}}},
		{name: "gap", spans: []struct {
			Start int
			End   int
		}{{Start: 0, End: 2}, {Start: 3, End: 4}}},
		{name: "overlap", spans: []struct {
			Start int
			End   int
		}{{Start: 0, End: 3}, {Start: 2, End: 4}}},
		{name: "reversed", spans: []struct {
			Start int
			End   int
		}{{Start: 0, End: 2}, {Start: 2, End: 1}}},
		{name: "past_end", spans: []struct {
			Start int
			End   int
		}{{Start: 0, End: 5}}},
		{name: "tail_uncovered", spans: []struct {
			Start int
			End   int
		}{{Start: 0, End: 3}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, _, _, err := base.ForwardPrefillLayerStreamedSpans(inputs, state, nil, 1e-6, meta, tc.spans); err == nil {
				t.Fatal("ForwardPrefillLayerStreamedSpans returned nil error")
			}
		})
	}
}

func buildQwen35PrefillStreamedTestModel(t *testing.T) (*Qwen35BaseModel, loaderconfig.QwenNativeMTPMetadata) {
	t.Helper()
	meta := testQwen35BaseMeta()
	meta.NumHiddenLayers = 2
	meta.MTPNumHiddenLayers = 0
	meta.LayerTypes = []string{"full_attention", "linear_attention"}
	meta.PartialRotaryFactor = 1
	meta.RopeTheta = 10000
	src := fakeQwen35TensorSource{}
	for k, v := range fullQwen35LayerSource(meta, "model.layers.0") {
		src[k] = v
	}
	for k, v := range linearQwen35LayerSource(meta, "model.layers.1") {
		src[k] = v
	}
	src["model.layers.0.self_attn.q_proj.weight"] = qwen35PatternTensorLike(src["model.layers.0.self_attn.q_proj.weight"], 1, 0.02)
	src["model.layers.0.self_attn.k_proj.weight"] = qwen35PatternTensorLike(src["model.layers.0.self_attn.k_proj.weight"], 2, 0.02)
	src["model.layers.0.self_attn.v_proj.weight"] = qwen35PatternTensorLike(src["model.layers.0.self_attn.v_proj.weight"], 3, 0.02)
	src["model.layers.0.self_attn.o_proj.weight"] = qwen35PatternTensorLike(src["model.layers.0.self_attn.o_proj.weight"], 4, 0.02)
	src["model.layers.1.linear_attn.in_proj_qkvz.weight"] = qwen35PatternTensorLike(src["model.layers.1.linear_attn.in_proj_qkvz.weight"], 5, 0.015)
	src["model.layers.1.linear_attn.in_proj_qkv.weight"] = src["model.layers.1.linear_attn.in_proj_qkvz.weight"]
	src["model.layers.1.linear_attn.in_proj_gate.weight"] = qwen35PatternTensorLike(src["model.layers.1.linear_attn.in_proj_gate.weight"], 6, 0.015)
	src["model.layers.1.linear_attn.in_proj_z.weight"] = src["model.layers.1.linear_attn.in_proj_gate.weight"]
	// The loader normalizes checkpoint conv layout [conv,1,kernel] to [conv,kernel,1].
	conv := src["model.layers.1.linear_attn.conv1d.weight"]
	convShape := conv.Shape()
	convData := make([]float32, conv.Numel())
	for i := range convData {
		convData[i] = float32(((7+i*3)%11)-5) * 0.01
	}
	src["model.layers.1.linear_attn.conv1d.weight"] = tensor.FromFloat32(convData, []int{convShape[0], convShape[2], convShape[1]})
	src["model.layers.1.linear_attn.in_proj_ba.weight"] = qwen35PatternTensorLike(src["model.layers.1.linear_attn.in_proj_ba.weight"], 8, 0.01)
	src["model.layers.1.linear_attn.in_proj_b.weight"] = src["model.layers.1.linear_attn.in_proj_ba.weight"]
	src["model.layers.1.linear_attn.in_proj_a.weight"] = qwen35PatternTensorLike(src["model.layers.1.linear_attn.in_proj_a.weight"], 9, 0.01)
	src["model.layers.1.linear_attn.A"] = qwen35NegativePatternTensorLike(src["model.layers.1.linear_attn.A"], 10, 0.02)
	src["model.layers.1.linear_attn.out_proj.weight"] = qwen35PatternTensorLike(src["model.layers.1.linear_attn.out_proj.weight"], 11, 0.015)
	base, err := LoadQwen35BaseModelLayers(CandidateQwen35TensorSource{Source: src}, meta)
	if err != nil {
		t.Fatalf("LoadQwen35BaseModelLayers: %v", err)
	}
	return base, meta
}

func qwen35PatternTensorLike(base *tensor.Tensor, seed int, scale float32) *tensor.Tensor {
	data := make([]float32, base.Numel())
	for i := range data {
		data[i] = float32(((seed+i*3)%11)-5) * scale
	}
	return tensor.FromFloat32(data, append([]int(nil), base.Shape()...))
}

func qwen35NegativePatternTensorLike(base *tensor.Tensor, seed int, scale float32) *tensor.Tensor {
	data := make([]float32, base.Numel())
	for i := range data {
		data[i] = -float32((seed+i)%7+1) * scale
	}
	return tensor.FromFloat32(data, append([]int(nil), base.Shape()...))
}

func qwen35PrefillStreamedInputs(tokens, hidden int) [][]float32 {
	out := make([][]float32, tokens)
	for tok := 0; tok < tokens; tok++ {
		row := make([]float32, hidden)
		for i := range row {
			row[i] = float32(((tok+1)*(i+2))%9-4)*0.2 + float32(tok%5)*0.03
		}
		out[tok] = row
	}
	return out
}

func qwen35ForwardOneReference(t *testing.T, base *Qwen35BaseModel, inputs [][]float32, state Qwen35BaseForwardState, rope []float32, eps float32, meta loaderconfig.QwenNativeMTPMetadata) ([][]float32, []Qwen35BaseForwardState, Qwen35BaseForwardState) {
	t.Helper()
	outs := make([][]float32, 0, len(inputs))
	prefix := make([]Qwen35BaseForwardState, 0, len(inputs))
	cur := state
	for i, input := range inputs {
		out, next, err := base.ForwardOne(input, cur, cur.Pos, rope, eps, meta)
		if err != nil {
			t.Fatalf("ForwardOne step %d: %v", i, err)
		}
		outs = append(outs, out)
		prefix = append(prefix, next)
		cur = next
	}
	return outs, prefix, cur
}

func qwen35PrefillPlanStructSpans(t *testing.T, tokens int, dims basemodel.PrefillChunkModelDims, estimate basemodel.PrefillChunkScratchEstimate, wantChunk int) []struct {
	Start int
	End   int
} {
	t.Helper()
	budget, err := estimate.TotalBytes(wantChunk)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := basemodel.NewPrefillChunkPlan(tokens, dims, budget, []int{32, 64, 128})
	if err != nil {
		t.Fatal(err)
	}
	if plan.ChunkSize != wantChunk {
		t.Fatalf("chunk size=%d want %d", plan.ChunkSize, wantChunk)
	}
	last := plan.Spans[len(plan.Spans)-1]
	if last.End != tokens {
		t.Fatalf("last span=%+v want end %d", last, tokens)
	}
	return plan.StructSpans()
}
