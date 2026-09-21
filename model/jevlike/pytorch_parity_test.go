package jevlike

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

type pytorchReferenceFixture struct {
	Schema            int        `json:"schema"`
	AbsoluteTolerance float64    `json:"absolute_tolerance"`
	Checkpoint        Checkpoint `json:"checkpoint"`
	Examples          []ChoiceExample `json:"examples"`
	Batch             struct {
		ContextIDs      [][]uint16   `json:"context_ids"`
		ContextMask     [][]bool     `json:"context_mask"`
		OptionIDs       [][][]uint16 `json:"option_ids"`
		OptionTokenMask [][][]bool   `json:"option_token_mask"`
		OptionMask      [][]bool     `json:"option_mask"`
		Labels          []int        `json:"labels"`
	} `json:"batch"`
	Predict struct {
		Logits        [][]float32 `json:"logits"`
		Probabilities [][]float32 `json:"probabilities"`
	} `json:"predict"`
	HeadInputs struct {
		Context     [][][]float32 `json:"context"`
		ContextMask [][]bool      `json:"context_mask"`
		Options     [][][]float32 `json:"options"`
		OptionMask  [][]bool      `json:"option_mask"`
	} `json:"head_inputs"`
	HeadGradients struct {
		DLogits    [][]float32 `json:"d_logits"`
		Parameters []NamedParameter `json:"parameters"`
		DContext   [][][]float32 `json:"d_context"`
		DOptions   [][][]float32 `json:"d_options"`
	} `json:"head_gradients"`
}

func TestPyTorchReferenceParity(t *testing.T) {
	fixture := loadPyTorchReferenceFixture(t)
	if fixture.Schema != 1 {
		t.Fatalf("fixture schema=%d want 1", fixture.Schema)
	}
	if fixture.AbsoluteTolerance <= 0 {
		t.Fatalf("fixture absolute_tolerance=%g must be positive", fixture.AbsoluteTolerance)
	}

	rawCheckpoint, err := json.Marshal(fixture.Checkpoint)
	if err != nil {
		t.Fatal(err)
	}
	checkpoint, err := ReadCheckpoint(bytes.NewReader(rawCheckpoint))
	if err != nil {
		t.Fatalf("ReadCheckpoint() error = %v", err)
	}
	model, err := checkpoint.Tiny()
	if err != nil {
		t.Fatalf("Checkpoint.Tiny() error = %v", err)
	}

	batch, err := BuildByteBatch(fixture.Examples, checkpoint.Config.ContextTokens, checkpoint.Config.OptionTokens)
	if err != nil {
		t.Fatalf("BuildByteBatch() error = %v", err)
	}
	wantBatch := ByteBatch{
		ContextIDs:      fixture.Batch.ContextIDs,
		ContextMask:     fixture.Batch.ContextMask,
		OptionIDs:       fixture.Batch.OptionIDs,
		OptionTokenMask: fixture.Batch.OptionTokenMask,
		OptionMask:      fixture.Batch.OptionMask,
		Labels:          fixture.Batch.Labels,
	}
	if !reflect.DeepEqual(batch, wantBatch) {
		t.Fatalf("BuildByteBatch() mismatch\n got=%#v\nwant=%#v", batch, wantBatch)
	}

	gotPredict, err := model.PredictBatch(batch)
	if err != nil {
		t.Fatalf("PredictBatch() error = %v", err)
	}
	compareFloat2D(t, "predict.logits", gotPredict.Logits, fixture.Predict.Logits, fixture.AbsoluteTolerance)
	compareFloat2D(t, "predict.probabilities", gotPredict.Probabilities, fixture.Predict.Probabilities, fixture.AbsoluteTolerance)

	headLogits, err := model.Head.Forward(
		fixture.HeadInputs.Context,
		fixture.HeadInputs.ContextMask,
		fixture.HeadInputs.Options,
		fixture.HeadInputs.OptionMask,
	)
	if err != nil {
		t.Fatalf("Head.Forward() error = %v", err)
	}
	compareFloat2D(t, "head.forward", headLogits, fixture.Predict.Logits, fixture.AbsoluteTolerance)

	paramGrads, dContext, dOptions, err := model.Head.Backward(
		fixture.HeadInputs.Context,
		fixture.HeadInputs.ContextMask,
		fixture.HeadInputs.Options,
		fixture.HeadInputs.OptionMask,
		fixture.HeadGradients.DLogits,
	)
	if err != nil {
		t.Fatalf("Head.Backward() error = %v", err)
	}
	compareFloat3D(t, "head.d_context", dContext, fixture.HeadGradients.DContext, fixture.AbsoluteTolerance)
	compareFloat3D(t, "head.d_options", dOptions, fixture.HeadGradients.DOptions, fixture.AbsoluteTolerance)
	compareNamedParameterValues(t, "head.parameters", paramGrads, fixture.HeadGradients.Parameters, fixture.AbsoluteTolerance)
}

func loadPyTorchReferenceFixture(t *testing.T) pytorchReferenceFixture {
	t.Helper()
	path := filepath.Join("testdata", "pytorch_reference.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile(%q) error = %v", path, err)
	}
	var fixture pytorchReferenceFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatalf("json.Unmarshal(%q) error = %v", path, err)
	}
	return fixture
}

func compareNamedParameterValues(t *testing.T, label string, got map[string][]float32, want []NamedParameter, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s count=%d want=%d", label, len(got), len(want))
	}
	for _, parameter := range want {
		values, ok := got[parameter.Name]
		if !ok {
			t.Fatalf("%s missing parameter %q", label, parameter.Name)
		}
		if len(values) != len(parameter.Values) {
			t.Fatalf("%s[%s] len=%d want=%d", label, parameter.Name, len(values), len(parameter.Values))
		}
		for i := range values {
			if diff := math.Abs(float64(values[i] - parameter.Values[i])); diff > tolerance {
				t.Fatalf("%s[%s][%d] got=%g want=%g diff=%g tol=%g", label, parameter.Name, i, values[i], parameter.Values[i], diff, tolerance)
			}
		}
	}
}

func compareFloat2D(t *testing.T, label string, got, want [][]float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s rows=%d want=%d", label, len(got), len(want))
	}
	for i := range got {
		if len(got[i]) != len(want[i]) {
			t.Fatalf("%s[%d] cols=%d want=%d", label, i, len(got[i]), len(want[i]))
		}
		for j := range got[i] {
			if diff := math.Abs(float64(got[i][j] - want[i][j])); diff > tolerance {
				t.Fatalf("%s[%d][%d] got=%g want=%g diff=%g tol=%g", label, i, j, got[i][j], want[i][j], diff, tolerance)
			}
		}
	}
}

func compareFloat3D(t *testing.T, label string, got, want [][][]float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s rows=%d want=%d", label, len(got), len(want))
	}
	for i := range got {
		if len(got[i]) != len(want[i]) {
			t.Fatalf("%s[%d] planes=%d want=%d", label, i, len(got[i]), len(want[i]))
		}
		for j := range got[i] {
			if len(got[i][j]) != len(want[i][j]) {
				t.Fatalf("%s[%d][%d] len=%d want=%d", label, i, j, len(got[i][j]), len(want[i][j]))
			}
			for k := range got[i][j] {
				if diff := math.Abs(float64(got[i][j][k] - want[i][j][k])); diff > tolerance {
					t.Fatalf("%s[%d][%d][%d] got=%g want=%g diff=%g tol=%g", label, i, j, k, got[i][j][k], want[i][j][k], diff, tolerance)
				}
			}
		}
	}
}
