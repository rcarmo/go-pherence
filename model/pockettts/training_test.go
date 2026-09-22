package pockettts

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func f64ptr(value float64) *float64 { return &value }

func TestTrainingManifestAdmissionAndCuts(t *testing.T) {
	line := `{"path":"chapter.mp3","start":3,"duration":4,"transcript":"one two three","speaker":"ignored","words":[{"word":"one","start":0.1,"end":0.8},{"word":"two","start":1.3,"end":1.8},{"word":"three","start":2.4,"end":3.3}]}`
	entries, err := ReadAlignedTrainingManifest(strings.NewReader(line+"\n"), 1)
	if err != nil {
		t.Fatal(err)
	}
	cuts := entries[0].EligibleTrainingCuts(0)
	if len(cuts) != 2 || cuts[0].WordIndex != 1 || math.Abs(cuts[0].Seconds-1.05) > 1e-12 || cuts[0].TargetTranscript != "two three" || cuts[1].WordIndex != 2 || math.Abs(cuts[1].Seconds-2.1) > 1e-12 || cuts[1].TargetTranscript != "three" {
		t.Fatalf("unexpected cuts: %+v", cuts)
	}
	path := filepath.Join(t.TempDir(), "train.jsonl")
	if err := os.WriteFile(path, []byte(line+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadAlignedTrainingManifest(path, 2)
	if err != nil || !reflect.DeepEqual(loaded, entries) {
		t.Fatalf("load mismatch: %+v %v", loaded, err)
	}
}

func TestTrainingManifestRejectsMalformed(t *testing.T) {
	cases := []string{
		``,
		`not-json`,
		`{"path":"a","duration":4,"transcript":"x","words":[]}`,
		`{"path":"a","duration":4,"transcript":"x y","words":[{"word":"x","start":null,"end":0.8},{"word":"y","start":1.2,"end":2.0}]}`,
		`{"path":"a","duration":1.5,"transcript":"x y","words":[{"word":"x","start":0,"end":0.5},{"word":"y","start":0.7,"end":1.4}]}`,
	}
	for i, line := range cases {
		if _, err := ReadAlignedTrainingManifest(strings.NewReader(line+"\n"), 1); err == nil {
			t.Fatalf("case %d accepted", i)
		}
	}
	valid := `{"path":"a","duration":4,"transcript":"x y","words":[{"word":"x","start":0,"end":0.8},{"word":"y","start":1.2,"end":3}]}`
	if _, err := ReadAlignedTrainingManifest(strings.NewReader(valid+"\n"+valid+"\n"), 1); err == nil {
		t.Fatal("accepted manifest over max entries")
	}
	entry := TrainingEntry{Path: "a", Duration: 4, Transcript: "x y", Words: []TrainingWord{{Word: "x", Start: f64ptr(1), End: f64ptr(2)}, {Word: "y", Start: f64ptr(0.5), End: f64ptr(3)}}}
	if err := entry.Validate(); err == nil {
		t.Fatal("accepted overlapping/non-monotonic alignment")
	}
}

func TestEOSLossBoundaryAndGradient(t *testing.T) {
	logits := []float32{-0.3, 0.4, 1.2, -9}
	mask := []bool{true, true, false, false}
	loss, gradient, err := EOSLossAndGradient(logits, mask)
	if err != nil {
		t.Fatal(err)
	}
	want := (softplus64(-.3) + softplus64(.4) + softplus64(-1.2)) / 3
	if math.Abs(loss-want) > 1e-8 {
		t.Fatalf("loss=%g want=%g", loss, want)
	}
	if gradient[3] != 0 {
		t.Fatalf("second invalid frame contributed gradient: %v", gradient)
	}
	centralDifferenceSlice(t, "eos", logits, gradient, func() float64 {
		value, _, e := EOSLossAndGradient(logits, mask)
		if e != nil {
			t.Fatal(e)
		}
		return value
	}, 1e-3, 2e-4)
}

func TestFlowLossGradients(t *testing.T) {
	prediction := []float32{.2, -.1, .7, .4}
	noise := []float32{.3, -.2, .1, .5}
	target := []float32{.8, .4, -.2, .9}
	times := []float32{.1, .8}
	_, gradient, err := FlowMatchingLossAndGradient(prediction, noise, target, times, 2, .001)
	if err != nil {
		t.Fatal(err)
	}
	centralDifferenceSlice(t, "flow matching", prediction, gradient, func() float64 {
		value, _, e := FlowMatchingLossAndGradient(prediction, noise, target, times, 2, .001)
		if e != nil {
			t.Fatal(e)
		}
		return value
	}, 1e-3, 2e-4)

	logvar := []float32{.2, -.1}
	_, dPrediction, dLogvar, err := LSDDiagonalLossAndGradient(prediction, target, logvar, 2, true)
	if err != nil {
		t.Fatal(err)
	}
	centralDifferenceSlice(t, "lsd prediction", prediction, dPrediction, func() float64 {
		value, _, _, e := LSDDiagonalLossAndGradient(prediction, target, logvar, 2, true)
		if e != nil {
			t.Fatal(e)
		}
		return value
	}, 1e-3, 2e-4)
	centralDifferenceSlice(t, "lsd logvar", logvar, dLogvar, func() float64 {
		value, _, _, e := LSDDiagonalLossAndGradient(prediction, target, logvar, 2, true)
		if e != nil {
			t.Fatal(e)
		}
		return value
	}, 1e-3, 2e-4)
}

type trainingOracle struct {
	Schema     int                  `json:"schema"`
	Hidden     []float32            `json:"hidden"`
	Noise      []float32            `json:"noise"`
	Target     []float32            `json:"target"`
	Times      []float32            `json:"times"`
	Mask       []bool               `json:"mask"`
	Parameters map[string][]float32 `json:"parameters"`
	Metrics    struct {
		FlowDiagonal float64 `json:"flow_diagonal"`
		EOS          float64 `json:"eos"`
		Loss         float64 `json:"loss"`
	} `json:"metrics"`
	Gradients map[string][]float32 `json:"gradients"`
	AdamW     struct {
		LearningRate        float32              `json:"learning_rate"`
		Beta1               float32              `json:"beta1"`
		Beta2               float32              `json:"beta2"`
		Epsilon             float32              `json:"epsilon"`
		WeightDecay         float32              `json:"weight_decay"`
		ParametersAfterStep map[string][]float32 `json:"parameters_after_step"`
		EMADecay            float32              `json:"ema_decay"`
		EMAAfterStep        map[string][]float32 `json:"ema_after_step"`
	} `json:"adamw"`
}

func loadTrainingOracle(t *testing.T) trainingOracle {
	t.Helper()
	data, err := os.ReadFile("testdata/training_tiny_pytorch.json")
	if err != nil {
		t.Fatal(err)
	}
	var oracle trainingOracle
	if err = json.Unmarshal(data, &oracle); err != nil {
		t.Fatal(err)
	}
	if oracle.Schema != 1 {
		t.Fatalf("oracle schema=%d", oracle.Schema)
	}
	return oracle
}

func modelFromOracle(t *testing.T, oracle trainingOracle) *TinyTrainingModel {
	t.Helper()
	m := &TinyTrainingModel{HiddenDim: 2, LatentDim: 2, FlowWeight: append([]float32(nil), oracle.Parameters["flow.weight"]...), FlowBias: append([]float32(nil), oracle.Parameters["flow.bias"]...), FlowLogVariance: oracle.Parameters["flow.logvar"][0], EOSWeight: append([]float32(nil), oracle.Parameters["eos.weight"]...), EOSBias: oracle.Parameters["eos.bias"][0]}
	if err := m.Validate(); err != nil {
		t.Fatal(err)
	}
	return m
}

func TestTinyTrainingPyTorchParity(t *testing.T) {
	oracle := loadTrainingOracle(t)
	model := modelFromOracle(t, oracle)
	batch := TinyTrainingBatch{Rows: 4, HiddenDim: 2, LatentDim: 2, Hidden: oracle.Hidden, Noise: oracle.Noise, Target: oracle.Target, Times: oracle.Times, Mask: oracle.Mask}
	metrics, gradients, err := ForwardBackwardTinyTraining(model, batch, DefaultTinyTrainingConfig())
	if err != nil {
		t.Fatal(err)
	}
	assertClose64(t, "flow loss", metrics.FlowDiagonal, oracle.Metrics.FlowDiagonal, 2e-7)
	assertClose64(t, "eos loss", metrics.EOS, oracle.Metrics.EOS, 2e-7)
	assertClose64(t, "total loss", metrics.Loss, oracle.Metrics.Loss, 2e-7)
	for name, got := range gradients.parameterMap() {
		assertSliceClose(t, name, got, oracle.Gradients[name], 3e-7)
	}

	trainer, err := NewTinyTrainer(model, AdamWConfig{LearningRate: oracle.AdamW.LearningRate, Beta1: oracle.AdamW.Beta1, Beta2: oracle.AdamW.Beta2, Epsilon: oracle.AdamW.Epsilon, WeightDecay: oracle.AdamW.WeightDecay}, oracle.AdamW.EMADecay)
	if err != nil {
		t.Fatal(err)
	}
	if err = trainer.Step(gradients); err != nil {
		t.Fatal(err)
	}
	for name, got := range model.trainingParameterMap() {
		assertSliceClose(t, "AdamW "+name, got, oracle.AdamW.ParametersAfterStep[name], 3e-7)
		assertSliceClose(t, "EMA "+name, trainer.EMA[name], oracle.AdamW.EMAAfterStep[name], 3e-7)
	}
}

func TestTinyTrainingCheckpointRoundTripAndResume(t *testing.T) {
	oracle := loadTrainingOracle(t)
	model := modelFromOracle(t, oracle)
	batch := TinyTrainingBatch{Rows: 4, HiddenDim: 2, LatentDim: 2, Hidden: oracle.Hidden, Noise: oracle.Noise, Target: oracle.Target, Times: oracle.Times, Mask: oracle.Mask}
	config := AdamWConfig{LearningRate: oracle.AdamW.LearningRate, Beta1: oracle.AdamW.Beta1, Beta2: oracle.AdamW.Beta2, Epsilon: oracle.AdamW.Epsilon, WeightDecay: oracle.AdamW.WeightDecay}
	trainer, _ := NewTinyTrainer(model, config, oracle.AdamW.EMADecay)
	_, gradients, _ := ForwardBackwardTinyTraining(model, batch, DefaultTinyTrainingConfig())
	if err := trainer.Step(gradients); err != nil {
		t.Fatal(err)
	}
	state, err := trainer.State()
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "nested", "checkpoint.json")
	if err = SaveTrainingState(path, state); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadTrainingState(path)
	if err != nil {
		t.Fatal(err)
	}
	resumed, err := NewTinyTrainerFromState(loaded)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(state, mustTrainingState(t, resumed)) {
		t.Fatal("checkpoint round-trip changed state")
	}
	// Both paths take the exact same second deterministic step.
	_, gradientA, _ := ForwardBackwardTinyTraining(trainer.Model, batch, DefaultTinyTrainingConfig())
	_, gradientB, _ := ForwardBackwardTinyTraining(resumed.Model, batch, DefaultTinyTrainingConfig())
	if err = trainer.Step(gradientA); err != nil {
		t.Fatal(err)
	}
	if err = resumed.Step(gradientB); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(mustTrainingState(t, trainer), mustTrainingState(t, resumed)) {
		t.Fatal("resumed step differs from uninterrupted step")
	}
}

func TestTrainingRejectsMalformed(t *testing.T) {
	if _, _, err := EOSLossAndGradient(nil, nil); err == nil {
		t.Fatal("accepted empty EOS")
	}
	if _, _, err := EOSLossAndGradient([]float32{0, 0, 0}, []bool{true, false, true}); err == nil {
		t.Fatal("accepted non-prefix EOS mask")
	}
	if _, _, err := FlowMatchingLossAndGradient([]float32{1}, []float32{1}, []float32{1}, []float32{2}, 1, .001); err == nil {
		t.Fatal("accepted bad time")
	}
	if _, _, _, err := LSDDiagonalLossAndGradient([]float32{1}, []float32{1}, []float32{0}, 1, false); err == nil {
		t.Fatal("accepted unused log variance")
	}
	m := &TinyTrainingModel{HiddenDim: 1, LatentDim: 1, FlowWeight: make([]float32, 3), FlowBias: []float32{0}, EOSWeight: []float32{0}}
	if _, _, err := ForwardBackwardTinyTraining(m, TinyTrainingBatch{}, DefaultTinyTrainingConfig()); err == nil {
		t.Fatal("accepted empty tiny batch")
	}
	state := TrainingState{Version: 1, Step: 1, HiddenDim: 1, LatentDim: 1}
	if err := state.Validate(); err == nil {
		t.Fatal("accepted incomplete state")
	}
}

func TestTrainingExplicitZeroWeightsAndGradientAliases(t *testing.T) {
	model := &TinyTrainingModel{HiddenDim: 1, LatentDim: 1, FlowWeight: []float32{.2, -.3, .4}, FlowBias: []float32{.1}, EOSWeight: []float32{-.2}, EOSBias: .3}
	batch := TinyTrainingBatch{Rows: 2, HiddenDim: 1, LatentDim: 1, Hidden: []float32{.5, -.4}, Noise: []float32{.2, .1}, Target: []float32{.8, -.2}, Times: []float32{.25, .75}, Mask: []bool{true, false}}
	metrics, gradients, err := ForwardBackwardTinyTraining(model, batch, TinyTrainingConfig{})
	if err != nil {
		t.Fatal(err)
	}
	if metrics.Loss != 0 {
		t.Fatalf("zero loss weights produced %g", metrics.Loss)
	}
	config := DefaultAdamWConfig()
	config.WeightDecay = 0
	config.Beta1 = 0
	config.Beta2 = 0
	trainer, err := NewTinyTrainer(model, config, 0)
	if err != nil {
		t.Fatal(err)
	}
	// Deliberately alias one public gradient slice to a model parameter. Step
	// must consume the original gradient values, not its own mutations.
	gradients.FlowWeight = model.FlowWeight
	wantModel := &TinyTrainingModel{HiddenDim: 1, LatentDim: 1, FlowWeight: append([]float32(nil), model.FlowWeight...), FlowBias: append([]float32(nil), model.FlowBias...), FlowLogVariance: model.FlowLogVariance, EOSWeight: append([]float32(nil), model.EOSWeight...), EOSBias: model.EOSBias}
	wantTrainer, _ := NewTinyTrainer(wantModel, config, 0)
	wantGradients := gradients
	wantGradients.FlowWeight = append([]float32(nil), model.FlowWeight...)
	if err = wantTrainer.Step(wantGradients); err != nil {
		t.Fatal(err)
	}
	if err = trainer.Step(gradients); err != nil {
		t.Fatal(err)
	}
	for name, got := range model.trainingParameterMap() {
		assertSliceClose(t, "aliased "+name, got, wantModel.trainingParameterMap()[name], 0)
	}
	state, err := trainer.State()
	if err != nil || state.AdamW.WeightDecay != 0 || state.AdamW.Beta1 != 0 || state.AdamW.Beta2 != 0 {
		t.Fatalf("zero optimizer settings not retained: %+v %v", state.AdamW, err)
	}
}

func mustTrainingState(t *testing.T, trainer *TinyTrainer) TrainingState {
	t.Helper()
	state, err := trainer.State()
	if err != nil {
		t.Fatal(err)
	}
	return state
}

func centralDifferenceSlice(t *testing.T, label string, values, gradients []float32, evaluate func() float64, step, tolerance float32) {
	t.Helper()
	for i := range values {
		original := values[i]
		values[i] = original + step
		plus := evaluate()
		values[i] = original - step
		minus := evaluate()
		values[i] = original
		numeric := float32((plus - minus) / (2 * float64(step)))
		if math.Abs(float64(gradients[i]-numeric)) > float64(tolerance) {
			t.Fatalf("%s[%d] gradient=%g numeric=%g", label, i, gradients[i], numeric)
		}
	}
}

func assertSliceClose(t *testing.T, label string, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length=%d want=%d", label, len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tolerance {
			t.Fatalf("%s[%d]=%.10g want=%.10g", label, i, got[i], want[i])
		}
	}
}

func assertClose64(t *testing.T, label string, got, want, tolerance float64) {
	t.Helper()
	if math.Abs(got-want) > tolerance {
		t.Fatalf("%s=%g want=%g", label, got, want)
	}
}
