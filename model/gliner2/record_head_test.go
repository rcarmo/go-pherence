package gliner2

import (
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"
)

func TestRecordHeadPublishedShapeBinding(t *testing.T) {
	f, err := os.Open("testdata/config.json")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	cfg, err := LoadConfig(f)
	if err != nil {
		t.Fatal(err)
	}
	src := shapeSource{
		"record_decoder.cand_proj.weight":        {128, 768},
		"record_decoder.cand_proj.bias":          {128},
		"record_decoder.field_proj.weight":       {128, 768},
		"record_decoder.field_proj.bias":         {128},
		"record_decoder.inst_proj.weight":        {128, 768},
		"record_decoder.inst_proj.bias":          {128},
		"record_decoder.instance_embed":          {32, 768},
		"record_decoder.k_proj.weight":           {128, 768},
		"record_decoder.k_proj.bias":             {128},
		"record_decoder.latent_seed_head.weight": {1, 768},
		"record_decoder.latent_seed_head.bias":   {1},
		"record_decoder.null_embed":              {128},
		"record_decoder.object_head.weight":      {1, 768},
		"record_decoder.object_head.bias":        {1},
		"record_decoder.q_proj.weight":           {128, 768},
		"record_decoder.q_proj.bias":             {128},
		"record_decoder.v_proj.weight":           {768, 768},
		"record_decoder.v_proj.bias":             {768},
	}
	head, err := LoadRecordHead(src, 768, cfg.BoundaryHead)
	if err != nil {
		t.Fatal(err)
	}
	if head.RecordDim != 128 || head.InstanceQueries != 32 || head.HiddenSize != 768 {
		t.Fatalf("binding mismatch: %+v", head)
	}
	delete(src, "record_decoder.null_embed")
	if _, err := LoadRecordHead(src, 768, cfg.BoundaryHead); err == nil {
		t.Fatal("missing published tensor accepted")
	}
}

func TestRecordHeadPublishedShapeBindingFromFixture(t *testing.T) {
	data, err := os.ReadFile("testdata/record_head_tensors.json")
	if err != nil {
		t.Fatal(err)
	}
	var meta map[string]struct {
		Shape []int `json:"shape"`
	}
	if err := json.Unmarshal(data, &meta); err != nil {
		t.Fatal(err)
	}
	src := shapeSource{}
	for name, tensor := range meta {
		if strings.HasPrefix(name, "record_decoder.") {
			src[name] = tensor.Shape
		}
	}
	if len(src) == 0 {
		t.Fatal("tmp fixture contains no record_decoder tensors")
	}
	f, err := os.Open("testdata/config.json")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	cfg, err := LoadConfig(f)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := LoadRecordHead(src, 768, cfg.BoundaryHead); err != nil {
		t.Fatal(err)
	}
}

func TestRecordHeadAssignLogitsNoCandidatesReturnsNullOnly(t *testing.T) {
	head := testRecordHead2x2()
	instStates := [][]float32{{1, 2}, {3, 4}}
	fieldQueries := [][]float32{{10, 20}}
	got, err := head.AssignLogits(instStates, fieldQueries, [][][]float32{{}})
	if err != nil {
		t.Fatal(err)
	}
	// Identity projections: (instance + field) dot [1,0] is 11,13.
	want := [][][]float32{{{11}, {13}}}
	assertTensor3Close(t, got, want, 1e-6)
}

func TestRecordHeadAssignLogitsNullColumnPrecedesUnscaledCandidates(t *testing.T) {
	head := testRecordHead2x2()
	instStates := [][]float32{{1, 2}}
	fieldQueries := [][]float32{{3, 4}}
	candidates := [][][]float32{{{5, 7}}}
	got, err := head.AssignLogits(instStates, fieldQueries, candidates)
	if err != nil {
		t.Fatal(err)
	}
	want := [][][]float32{{{4, 62}}}
	assertTensor3Close(t, got, want, 1e-6)
}

func TestRecordHeadAnchorlessStatesNoCandidatesReturnsCopy(t *testing.T) {
	head := testRecordHeadScalar()
	got, err := head.AnchorlessStates(nil)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]float32{{1}, {2}}
	assertMatrixClose(t, got, want, 1e-6)
	got[0][0] = 99
	if head.InstanceEmbed[0] != 1 {
		t.Fatalf("anchorless states aliased learned embeddings: %v", head.InstanceEmbed)
	}
}

func TestRecordHeadAnchorlessStatesCrossAttentionResidual(t *testing.T) {
	head := testRecordHeadScalar()
	got, err := head.AnchorlessStates([][][]float32{{{2}}, {{1}}})
	if err != nil {
		t.Fatal(err)
	}
	want := [][]float32{{2.7310586}, {3.8807971}}
	assertMatrixClose(t, got, want, 1e-5)
}

func TestRecordHeadObjectAndLatentSeedLogits(t *testing.T) {
	head := testRecordHead2x2()
	head.ObjectHead = Linear{InDim: 2, OutDim: 1, Weight: []float32{2, 3}, Bias: []float32{1}}
	head.LatentSeedHead = Linear{InDim: 2, OutDim: 1, Weight: []float32{-1, 4}, Bias: []float32{-2}}
	states := [][]float32{{1, 2}, {3, 4}}
	object, err := head.ObjectLogits(states)
	if err != nil {
		t.Fatal(err)
	}
	latent, err := head.LatentSeedLogits(states)
	if err != nil {
		t.Fatal(err)
	}
	assertSliceClose(t, object, []float32{9, 19}, 1e-6)
	assertSliceClose(t, latent, []float32{5, 11}, 1e-6)
}

func TestRecordHeadValidateProjectionMismatch(t *testing.T) {
	head := testRecordHead2x2()
	head.CandProjection = Linear{InDim: 2, OutDim: 3, Weight: make([]float32, 6), Bias: make([]float32, 3)}
	err := head.Validate()
	if err == nil || !strings.Contains(err.Error(), "cand_proj dims") {
		t.Fatalf("err=%v want cand_proj dims", err)
	}
}

func testRecordHead2x2() RecordHead {
	return RecordHead{
		HiddenSize:      2,
		RecordDim:       2,
		InstanceQueries: 2,
		InstProjection:  Linear{InDim: 2, OutDim: 2, Weight: []float32{1, 0, 0, 1}, Bias: []float32{0, 0}},
		FieldProjection: Linear{InDim: 2, OutDim: 2, Weight: []float32{1, 0, 0, 1}, Bias: []float32{0, 0}},
		CandProjection:  Linear{InDim: 2, OutDim: 2, Weight: []float32{1, 0, 0, 1}, Bias: []float32{0, 0}},
		NullEmbed:       []float32{1, 0},
		ObjectHead:      Linear{InDim: 2, OutDim: 1, Weight: []float32{1, 1}, Bias: []float32{0}},
		LatentSeedHead:  Linear{InDim: 2, OutDim: 1, Weight: []float32{1, -1}, Bias: []float32{0}},
		InstanceEmbed:   []float32{1, 2, 3, 4},
		QProjection:     Linear{InDim: 2, OutDim: 2, Weight: []float32{1, 0, 0, 1}, Bias: []float32{0, 0}},
		KProjection:     Linear{InDim: 2, OutDim: 2, Weight: []float32{1, 0, 0, 1}, Bias: []float32{0, 0}},
		VProjection:     Linear{InDim: 2, OutDim: 2, Weight: []float32{1, 0, 0, 1}, Bias: []float32{0, 0}},
	}
}

func testRecordHeadScalar() RecordHead {
	return RecordHead{
		HiddenSize:      1,
		RecordDim:       1,
		InstanceQueries: 2,
		InstProjection:  Linear{InDim: 1, OutDim: 1, Weight: []float32{1}, Bias: []float32{0}},
		FieldProjection: Linear{InDim: 1, OutDim: 1, Weight: []float32{1}, Bias: []float32{0}},
		CandProjection:  Linear{InDim: 1, OutDim: 1, Weight: []float32{1}, Bias: []float32{0}},
		NullEmbed:       []float32{1},
		ObjectHead:      Linear{InDim: 1, OutDim: 1, Weight: []float32{1}, Bias: []float32{0}},
		LatentSeedHead:  Linear{InDim: 1, OutDim: 1, Weight: []float32{1}, Bias: []float32{0}},
		InstanceEmbed:   []float32{1, 2},
		QProjection:     Linear{InDim: 1, OutDim: 1, Weight: []float32{1}, Bias: []float32{0}},
		KProjection:     Linear{InDim: 1, OutDim: 1, Weight: []float32{1}, Bias: []float32{0}},
		VProjection:     Linear{InDim: 1, OutDim: 1, Weight: []float32{1}, Bias: []float32{0}},
	}
}

func assertTensor3Close(t *testing.T, got, want [][][]float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("rank1 len=%d want=%d", len(got), len(want))
	}
	for i := range want {
		assertMatrixClose(t, got[i], want[i], tol)
	}
}

func assertMatrixClose(t *testing.T, got, want [][]float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("rows=%d want=%d", len(got), len(want))
	}
	for i := range want {
		assertSliceClose(t, got[i], want[i], tol)
	}
}

func assertSliceClose(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len=%d want=%d values=%v", len(got), len(want), got)
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Fatalf("[%d]=%g want %g", i, got[i], want[i])
		}
	}
}
