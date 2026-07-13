package mosstranscribe

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/rcarmo/go-pherence/models/whisper"
)

type encoderParityFixture struct {
	Transformers string `json:"transformers"`
	Compute      struct {
		BF16 []encoderBoundaryFixture `json:"bf16"`
		F32  []encoderBoundaryFixture `json:"f32_widened"`
	} `json:"compute"`
	AudioPost struct {
		BF16 []encoderBoundaryFixture `json:"bf16"`
		F32  []encoderBoundaryFixture `json:"f32_widened"`
	} `json:"audio_post"`
}

type encoderBoundaryFixture struct {
	Name    string                 `json:"name"`
	Shape   []int                  `json:"shape"`
	Samples []encoderSampleFixture `json:"samples"`
}

type encoderSampleFixture struct {
	Row   int     `json:"row"`
	Col   int     `json:"col"`
	Value float32 `json:"value"`
}

func TestRealCheckpointWhisperEncoderParity(t *testing.T) {
	modelDir := os.Getenv("MOSS_TRANSCRIBE_MODEL_DIR")
	if modelDir == "" {
		t.Skip("set MOSS_TRANSCRIBE_MODEL_DIR for the pinned real-checkpoint gate")
	}
	data, err := os.ReadFile("testdata/whisper_encoder_transformers_4_57_1.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture encoderParityFixture
	if err := json.Unmarshal(data, &fixture); err != nil {
		t.Fatal(err)
	}
	model, err := LoadAudioBackbone(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()

	samples := make([]float32, AudioChunkSamples)
	for i := 0; i < AudioSampleRate; i++ {
		samples[i] = float32(0.2*math.Sin(2*math.Pi*440*float64(i)/AudioSampleRate) +
			0.05*math.Sin(2*math.Pi*(200+1200*float64(i)/AudioSampleRate)*float64(i)/AudioSampleRate))
	}
	features, err := (AudioChunk{Samples: samples, TokenLength: 375}).InputFeatures(model.Config.WhisperConfig())
	if err != nil {
		t.Fatal(err)
	}
	boundaryIndex := 0
	maxDiff := float64(0)
	maxWhere := ""
	maxBF16Diff := float64(0)
	maxBF16Where := ""
	finalBF16Diff := float64(0)
	encoderOutput := model.Encoder.ForwardObserved(features, 3000, func(boundary whisper.EncoderBoundary, layer, rows, cols int, values []float32) {
		if boundaryIndex >= len(fixture.Compute.F32) {
			t.Fatalf("unexpected native boundary %s layer %d", boundary, layer)
		}
		want := fixture.Compute.F32[boundaryIndex]
		name := string(boundary)
		if boundary == whisper.EncoderBoundaryLayer {
			name = fmt.Sprintf("layer.%d", layer)
		}
		if name != want.Name || len(want.Shape) != 2 || rows != want.Shape[0] || cols != want.Shape[1] {
			t.Fatalf("boundary %d got %s [%d,%d], want %s %v", boundaryIndex, name, rows, cols, want.Name, want.Shape)
		}
		actualBF16 := fixture.Compute.BF16[boundaryIndex]
		for sampleIndex, sample := range want.Samples {
			got := values[sample.Row*cols+sample.Col]
			diff := math.Abs(float64(got - sample.Value))
			if diff > maxDiff {
				maxDiff = diff
				maxWhere = fmt.Sprintf("%s[%d,%d]", name, sample.Row, sample.Col)
			}
			bf16Diff := math.Abs(float64(got - actualBF16.Samples[sampleIndex].Value))
			if bf16Diff > maxBF16Diff {
				maxBF16Diff = bf16Diff
				maxBF16Where = fmt.Sprintf("%s[%d,%d]", name, sample.Row, sample.Col)
			}
			if boundary == whisper.EncoderBoundaryFinalNorm && bf16Diff > finalBF16Diff {
				finalBF16Diff = bf16Diff
			}
		}
		boundaryIndex++
	})
	if boundaryIndex != len(fixture.Compute.F32) {
		t.Fatalf("observed %d boundaries, want %d", boundaryIndex, len(fixture.Compute.F32))
	}
	if maxDiff > 1e-4 {
		t.Fatalf("Transformers %s widened-BF16 max abs diff %.6g at %s exceeds 1e-4", fixture.Transformers, maxDiff, maxWhere)
	}
	if maxBF16Diff > 0.07 || finalBF16Diff > 0.012 {
		t.Fatalf("Transformers BF16 drift max=%.6g at %s final=%.6g", maxBF16Diff, maxBF16Where, finalBF16Diff)
	}
	t.Logf("Transformers %s widened-BF16 max=%.6g at %s; actual BF16 max=%.6g at %s final=%.6g", fixture.Transformers, maxDiff, maxWhere, maxBF16Diff, maxBF16Where, finalBF16Diff)

	merged, tokens, ok := TimeMerge(encoderOutput[:52*AudioWidth], 52)
	if !ok || tokens != 13 {
		t.Fatalf("time merge tokens=%d ok=%v", tokens, ok)
	}
	adaptorOut := make([]float32, tokens*AdaptorHiddenDim)
	scratch := make([]float32, len(adaptorOut))
	if !ForwardAdaptorTo(adaptorOut, scratch, merged, tokens, model.Adaptor) {
		t.Fatal("native adaptor failed")
	}
	postValues := [][]float32{merged, adaptorOut}
	for boundary, values := range postValues {
		want := fixture.AudioPost.F32[boundary]
		actualBF16 := fixture.AudioPost.BF16[boundary]
		cols := want.Shape[1]
		var widenedMax, bf16Max float64
		for sampleIndex, sample := range want.Samples {
			got := values[sample.Row*cols+sample.Col]
			widenedMax = math.Max(widenedMax, math.Abs(float64(got-sample.Value)))
			bf16Max = math.Max(bf16Max, math.Abs(float64(got-actualBF16.Samples[sampleIndex].Value)))
		}
		widenedLimit, bf16Limit := 1e-4, 0.02
		if boundary == 0 { // Time merge is a view of encoder output.
			bf16Limit = 0.012
		}
		if widenedMax > widenedLimit || bf16Max > bf16Limit {
			t.Fatalf("%s parity widened=%.6g/%g BF16=%.6g/%g", want.Name, widenedMax, widenedLimit, bf16Max, bf16Limit)
		}
		t.Logf("%s parity widened=%.6g BF16=%.6g", want.Name, widenedMax, bf16Max)
	}
}
