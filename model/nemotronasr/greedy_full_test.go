package nemotronasr

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/loader/safetensors"
)

// The 139 rows come from PyTorch's full JFK encoder; only native RNNT
// predictor/joint/greedy decisions are checked, not native full encoding.
func TestReleasedGreedyRNNTJFKFullProjectedDecodePyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_ASR_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_ASR_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	decoder, err := LoadRNNTDecoder(file)
	if err != nil {
		t.Fatal(err)
	}
	projection, err := LoadRNNTProjection(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("testdata/asr_jfk_full_decode.json")
	if err != nil {
		t.Fatal(err)
	}
	var want struct {
		EncoderRows    int       `json:"encoder_rows"`
		Lookahead      int       `json:"lookahead"`
		PromptID       int       `json:"prompt_id"`
		Tokens         []int     `json:"tokens"`
		Frames         []int     `json:"frames"`
		SelectedLogits []float64 `json:"selected_logits"`
		RunnerTokens   []int     `json:"runner_tokens"`
		RunnerLogits   []float64 `json:"runner_logits"`
		TopTwoMargins  []float64 `json:"top_two_margins"`
	}
	if err := json.Unmarshal(data, &want); err != nil {
		t.Fatal(err)
	}
	if want.EncoderRows != 139 || want.Lookahead != 3 || want.PromptID != 101 || len(want.Tokens) != len(want.Frames) || len(want.Tokens) != len(want.SelectedLogits) || len(want.Tokens) != len(want.RunnerTokens) || len(want.Tokens) != len(want.RunnerLogits) || len(want.Tokens) != len(want.TopTwoMargins) || len(want.Tokens) < want.EncoderRows || len(want.Tokens) > want.EncoderRows*10 {
		t.Fatal("invalid full JFK RNNT fixture")
	}
	encoder := readStemFixture(t, "asr_jfk_full_encoder", want.EncoderRows*rnntHidden)
	original := append([]float32(nil), encoder...)
	var steps, logitOutliers, unstable int
	var retainedFirst []float32
	var firstValue float32
	var maxLogitError, maxRunnerError, minMargin float64
	minMargin = math.Inf(1)
	tokens, frames, err := (&GreedyRNNT{Decoder: decoder, Projection: projection}).decode(encoder, want.EncoderRows, func(step int, logits []float32) error {
		if step >= len(want.Tokens) {
			return fmt.Errorf("unexpected extra RNNT step")
		}
		steps++
		if step == 0 {
			retainedFirst, firstValue = logits, logits[0]
		} else if step == 1 && (len(retainedFirst) != rnntVocabulary || retainedFirst[0] != firstValue || &retainedFirst[0] == &logits[0]) {
			return fmt.Errorf("observer logits aliased across steps")
		}
		selected := 0
		runner := 1
		if logits[runner] > logits[selected] {
			selected, runner = runner, selected
		}
		for i := 2; i < len(logits); i++ {
			if logits[i] > logits[selected] {
				runner, selected = selected, i
			} else if logits[i] > logits[runner] {
				runner = i
			}
		}
		delta := math.Abs(float64(logits[selected]) - want.SelectedLogits[step])
		runnerDelta := math.Abs(float64(logits[runner]) - want.RunnerLogits[step])
		maxLogitError = math.Max(maxLogitError, delta)
		maxRunnerError = math.Max(maxRunnerError, runnerDelta)
		margin := float64(logits[selected] - logits[runner])
		minMargin = math.Min(minMargin, want.TopTwoMargins[step])
		if selected != want.Tokens[step] || runner != want.RunnerTokens[step] || math.IsNaN(float64(logits[selected])) || math.IsInf(float64(logits[selected]), 0) || math.IsNaN(float64(logits[runner])) || math.IsInf(float64(logits[runner]), 0) || delta > 3e-4+2e-5*math.Abs(want.SelectedLogits[step]) || runnerDelta > 3e-4+2e-5*math.Abs(want.RunnerLogits[step]) {
			logitOutliers++
		}
		// A reference top-two gap comparable to the numerical error is
		// decision-sensitive; track it rather than declaring broad stability.
		if want.TopTwoMargins[step] < delta+runnerDelta || margin < 0 {
			unstable++
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	nonblank := 0
	for _, token := range tokens {
		if token != rnntBlank {
			nonblank++
		}
	}
	t.Logf("steps=%d nonblank=%d max_selected_logit_error=%g max_runner_logit_error=%g min_reference_margin=%g sensitive_steps=%d outliers=%d", steps, nonblank, maxLogitError, maxRunnerError, minMargin, unstable, logitOutliers)
	if steps != len(want.Tokens) || retainedFirst[0] != firstValue || unstable != 0 || logitOutliers != 0 || !reflect.DeepEqual(tokens, want.Tokens) || !reflect.DeepEqual(frames, want.Frames) || nonblank < 2 || !reflect.DeepEqual(encoder, original) {
		t.Fatal("full JFK RNNT decode differs from PyTorch")
	}
}
