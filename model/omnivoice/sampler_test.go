package omnivoice

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strconv"
	"testing"
)

type samplerFloatTensor struct {
	Shape []int     `json:"shape"`
	Data  []float32 `json:"-"`
}

func (t *samplerFloatTensor) UnmarshalJSON(b []byte) error {
	var aux struct {
		Shape []int             `json:"shape"`
		Data  []json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(b, &aux); err != nil {
		return err
	}
	t.Shape = aux.Shape
	t.Data = make([]float32, len(aux.Data))
	for i, raw := range aux.Data {
		var num float64
		if err := json.Unmarshal(raw, &num); err == nil {
			t.Data[i] = float32(num)
			continue
		}
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return fmt.Errorf("float tensor[%d]: %w", i, err)
		}
		switch s {
		case "-inf":
			t.Data[i] = float32(math.Inf(-1))
		case "+inf":
			t.Data[i] = float32(math.Inf(1))
		case "nan":
			t.Data[i] = float32(math.NaN())
		default:
			v, err := strconv.ParseFloat(s, 32)
			if err != nil {
				return fmt.Errorf("float tensor[%d]: %w", i, err)
			}
			t.Data[i] = float32(v)
		}
	}
	return nil
}

type samplerIntTensor struct {
	Shape []int `json:"shape"`
	Data  []int `json:"data"`
}

type samplerFixture struct {
	Config struct {
		NumAudioCodebook   int     `json:"num_audio_codebook"`
		TargetLen          int     `json:"target_len"`
		AudioVocabSize     int     `json:"audio_vocab_size"`
		AudioMaskID        int     `json:"audio_mask_id"`
		NumStep            int     `json:"num_step"`
		GuidanceScale      float32 `json:"guidance_scale"`
		TShift             float32 `json:"t_shift"`
		LayerPenaltyFactor float32 `json:"layer_penalty_factor"`
		PositionTemp       float32 `json:"position_temperature"`
		ClassTemp          float32 `json:"class_temperature"`
		ClassTopKRatio     float64 `json:"class_top_k_ratio"`
		SelectionK         int     `json:"selection_k"`
	} `json:"config"`
	Schedule struct {
		TargetLens []int     `json:"target_lens"`
		Timesteps  []float32 `json:"timesteps"`
		Schedules  [][]int   `json:"schedules"`
	} `json:"schedule"`
	CondLogits           samplerFloatTensor `json:"cond_logits"`
	UncondLogits         samplerFloatTensor `json:"uncond_logits"`
	ClassUniforms        samplerFloatTensor `json:"class_uniforms"`
	GuidedLogProbs       samplerFloatTensor `json:"guided_log_probs"`
	FilteredLogProbs     samplerFloatTensor `json:"filtered_log_probs"`
	ClassGumbelScores    samplerFloatTensor `json:"class_gumbel_scores"`
	PredTokens           samplerIntTensor   `json:"pred_tokens"`
	ConfidenceScores     samplerFloatTensor `json:"confidence_scores"`
	TokensBefore         samplerIntTensor   `json:"tokens_before"`
	PositionUniforms     samplerFloatTensor `json:"position_uniforms"`
	PositionScores       samplerFloatTensor `json:"position_scores"`
	PositionGumbelScores samplerFloatTensor `json:"position_gumbel_scores"`
	SelectedFlatIndices  []int              `json:"selected_flat_indices"`
	TokensAfter          samplerIntTensor   `json:"tokens_after"`
}

func loadSamplerFixture(t testing.TB) samplerFixture {
	t.Helper()
	raw, err := os.ReadFile("../../testdata/omnivoice/sampler.json")
	if err != nil {
		t.Fatal(err)
	}
	var f samplerFixture
	if err = json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	return f
}

func TestSamplerTimeStepsAndScheduleFixture(t *testing.T) {
	f := loadSamplerFixture(t)
	timesteps := make([]float32, f.Config.NumStep+1)
	if err := TimeStepsInto(timesteps, 0, 1, f.Config.NumStep, f.Config.TShift); err != nil {
		t.Fatal(err)
	}
	assertCloseF32(t, "timesteps", timesteps, f.Schedule.Timesteps, 5e-7)
	for i, targetLen := range f.Schedule.TargetLens {
		sched := make([]int, f.Config.NumStep)
		if err := BuildUnmaskScheduleInto(sched, targetLen, f.Config.NumAudioCodebook, timesteps); err != nil {
			t.Fatalf("schedule[%d]: %v", i, err)
		}
		assertEqualInts(t, "schedule", sched, f.Schedule.Schedules[i])
	}
}

func TestSamplerGuidanceAndClassSamplingFixture(t *testing.T) {
	f := loadSamplerFixture(t)
	ws, err := NewSamplerWorkspace(f.Config.NumAudioCodebook, f.Config.TargetLen, f.Config.AudioVocabSize)
	if err != nil {
		t.Fatal(err)
	}
	logProbs := make([]float32, len(f.GuidedLogProbs.Data))
	if err = ws.GuidedLogProbsInto(logProbs, f.CondLogits.Data, f.UncondLogits.Data, f.Config.TargetLen, f.Config.GuidanceScale, f.Config.AudioMaskID); err != nil {
		t.Fatal(err)
	}
	assertCloseF32(t, "guided_log_probs", logProbs, f.GuidedLogProbs.Data, 4e-6)

	filtered := make([]float32, len(f.FilteredLogProbs.Data))
	if err = ws.FilterTopKInto(filtered, logProbs, f.Config.TargetLen, f.Config.ClassTopKRatio); err != nil {
		t.Fatal(err)
	}
	assertCloseF32(t, "filtered_log_probs", filtered, f.FilteredLogProbs.Data, 4e-6)
	inplace := append([]float32(nil), logProbs...)
	if err = ws.FilterTopKInto(inplace, inplace, f.Config.TargetLen, f.Config.ClassTopKRatio); err != nil {
		t.Fatal(err)
	}
	assertCloseF32(t, "inplace top-k", inplace, filtered, 0)

	classScores := make([]float32, len(filtered))
	if err = ws.GumbelPerturbInto(classScores, filtered, f.Config.ClassTemp, GumbelNoise{Uniforms: f.ClassUniforms.Data}); err != nil {
		t.Fatal(err)
	}
	assertCloseF32(t, "class_gumbel_scores", classScores, f.ClassGumbelScores.Data, 4e-6)

	rows := f.Config.NumAudioCodebook * f.Config.TargetLen
	pred := make([]int, rows)
	confidence := make([]float32, rows)
	if err = ws.PredictTokensWithConfidenceInto(pred, confidence, logProbs, f.Config.TargetLen, f.Config.ClassTemp, f.Config.ClassTopKRatio, GumbelNoise{Uniforms: f.ClassUniforms.Data}); err != nil {
		t.Fatal(err)
	}
	assertEqualInts(t, "pred_tokens", pred, f.PredTokens.Data)
	assertCloseF32(t, "confidence_scores", confidence, f.ConfidenceScores.Data, 4e-6)
}

func TestSamplerPositionSelectionFixture(t *testing.T) {
	f := loadSamplerFixture(t)
	ws, err := NewSamplerWorkspace(f.Config.NumAudioCodebook, f.Config.TargetLen, f.Config.AudioVocabSize)
	if err != nil {
		t.Fatal(err)
	}
	positionScores := make([]float32, len(f.PositionScores.Data))
	if err = ws.PositionScoresInto(positionScores, f.ConfidenceScores.Data, f.Config.TargetLen, f.Config.LayerPenaltyFactor); err != nil {
		t.Fatal(err)
	}
	assertCloseF32(t, "position_scores", positionScores, f.PositionScores.Data, 4e-6)

	gumbelScores := make([]float32, len(positionScores))
	if err = ws.GumbelPerturbInto(gumbelScores, positionScores, f.Config.PositionTemp, GumbelNoise{Uniforms: f.PositionUniforms.Data}); err != nil {
		t.Fatal(err)
	}
	assertCloseF32(t, "position_gumbel_scores", gumbelScores, f.PositionGumbelScores.Data, 4e-6)

	tokens := append([]int(nil), f.TokensBefore.Data...)
	selected, err := ws.ApplyConfidenceSelection(tokens, f.PredTokens.Data, f.ConfidenceScores.Data, f.Config.TargetLen, f.Config.AudioMaskID, f.Config.SelectionK, f.Config.LayerPenaltyFactor, f.Config.PositionTemp, GumbelNoise{Uniforms: f.PositionUniforms.Data})
	if err != nil {
		t.Fatal(err)
	}
	assertEqualInts(t, "selected_flat_indices", selected, f.SelectedFlatIndices)
	assertEqualInts(t, "tokens_after", tokens, f.TokensAfter.Data)
}

func TestSamplerValidation(t *testing.T) {
	ws, err := NewSamplerWorkspace(2, 3, 5)
	if err != nil {
		t.Fatal(err)
	}
	if err = TimeStepsInto(make([]float32, 0), 0, 1, 1, 0.1); err == nil {
		t.Fatal("expected timestep length validation")
	}
	if err = ws.GumbelPerturbInto(make([]float32, 2), make([]float32, 2), 1, GumbelNoise{}); err == nil {
		t.Fatal("expected empty-noise validation")
	}
	if err = ws.GumbelPerturbInto(make([]float32, 2), make([]float32, 2), 1, GumbelNoise{Uniforms: []float32{0.2, 0.3}, Gumbels: []float32{1, 2}}); err == nil {
		t.Fatal("expected dual-noise validation")
	}
	tokens := []int{4, 1, 2, 3, 4, 4}
	pred := []int{0, 0, 0, 0, 0, 0}
	conf := []float32{0, 1, 2, 3, 4, 5}
	if _, err = ws.ApplyConfidenceSelection(tokens, pred, conf, 3, 4, 4, 1, 0, GumbelNoise{}); err == nil {
		t.Fatal("expected k > remaining validation")
	}
}

func TestSamplerWorkspaceHotPathAllocs(t *testing.T) {
	f := loadSamplerFixture(t)
	ws, err := NewSamplerWorkspace(f.Config.NumAudioCodebook, f.Config.TargetLen, f.Config.AudioVocabSize)
	if err != nil {
		t.Fatal(err)
	}
	logProbs := make([]float32, len(f.GuidedLogProbs.Data))
	pred := make([]int, f.Config.NumAudioCodebook*f.Config.TargetLen)
	confidence := make([]float32, len(pred))
	tokens := make([]int, len(f.TokensBefore.Data))
	allocs := testing.AllocsPerRun(50, func() {
		if err := ws.GuidedLogProbsInto(logProbs, f.CondLogits.Data, f.UncondLogits.Data, f.Config.TargetLen, f.Config.GuidanceScale, f.Config.AudioMaskID); err != nil {
			t.Fatal(err)
		}
		if err := ws.PredictTokensWithConfidenceInto(pred, confidence, logProbs, f.Config.TargetLen, f.Config.ClassTemp, f.Config.ClassTopKRatio, GumbelNoise{Uniforms: f.ClassUniforms.Data}); err != nil {
			t.Fatal(err)
		}
		copy(tokens, f.TokensBefore.Data)
		selected, err := ws.ApplyConfidenceSelection(tokens, pred, confidence, f.Config.TargetLen, f.Config.AudioMaskID, f.Config.SelectionK, f.Config.LayerPenaltyFactor, f.Config.PositionTemp, GumbelNoise{Uniforms: f.PositionUniforms.Data})
		if err != nil {
			t.Fatal(err)
		}
		if len(selected) != f.Config.SelectionK {
			t.Fatalf("selected=%d want %d", len(selected), f.Config.SelectionK)
		}
	})
	if allocs != 0 {
		t.Fatalf("allocs/run=%g want 0", allocs)
	}
}

func assertEqualInts(t testing.TB, name string, got, want []int) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length=%d want %d", name, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s[%d]=%d want %d", name, i, got[i], want[i])
		}
	}
}

func assertCloseF32(t testing.TB, name string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length=%d want %d", name, len(got), len(want))
	}
	for i := range want {
		gv, wv := got[i], want[i]
		if math.IsInf(float64(gv), 0) || math.IsInf(float64(wv), 0) {
			if !(math.IsInf(float64(gv), -1) && math.IsInf(float64(wv), -1)) {
				t.Fatalf("%s[%d]=%v want %v", name, i, gv, wv)
			}
			continue
		}
		diff := math.Abs(float64(gv - wv))
		if diff > tol || math.IsNaN(diff) {
			t.Fatalf("%s[%d]=%.9g want %.9g diff=%g", name, i, gv, wv, diff)
		}
	}
}
