package nemotrondiarization

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/rcarmo/go-pherence/loader/audio"
	"github.com/rcarmo/go-pherence/loader/safetensors"
)

func TestReleasedSegmentsPyTorchParity(t *testing.T) {
	path := os.Getenv("GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL")
	if path == "" {
		t.Skip("set GO_PHERENCE_NEMOTRON_DIARIZATION_MODEL to pinned model.safetensors")
	}
	file, err := safetensors.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	model, err := LoadOfflineRequest(file)
	if err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	pcm, rate, err := audio.WAV(filepath.Join("..", "..", "testdata", "jfk.wav"))
	if err != nil || rate != 16000 {
		t.Fatalf("JFK input rate=%d err=%v", rate, err)
	}
	logits, frames, err := model.ForwardPCM(pcm)
	if err != nil {
		t.Fatal(err)
	}
	if frames != 1101 {
		t.Fatalf("frames=%d", frames)
	}
	mask := make([]bool, frames)
	for i := range mask {
		mask[i] = i < 1100
	}
	got, err := ExtractSegments(logits, mask)
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile("testdata/jfk_request_segments.json")
	if err != nil {
		t.Fatal(err)
	}
	var want []Segment
	if err := json.Unmarshal(b, &want); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("segments=%v want=%v", got, want)
	}
}

func TestExtractSegmentsMaskOverlapAndValidation(t *testing.T) {
	logits := make([]float32, 4*diarizationSpeakers)
	for i := range logits {
		logits[i] = -1
	}
	logits[0] = 1
	logits[1] = 1
	logits[diarizationSpeakers] = 1
	logits[2*diarizationSpeakers+1] = 1
	logits[3*diarizationSpeakers] = 1
	got, err := ExtractSegments(logits, []bool{true, true, true, false})
	if err != nil {
		t.Fatal(err)
	}
	want := []Segment{{Start: 0, End: 0.02, Speaker: 0}, {Start: 0, End: 0.01, Speaker: 1}, {Start: 0.02, End: 0.03, Speaker: 1}}
	// Sorted by start then speaker, as in the released processor.
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("segments=%v want=%v", got, want)
	}
	if _, err := ExtractSegments(logits, []bool{true}); err == nil {
		t.Fatal("accepted mismatched mask")
	}
	logits[0] = float32(math.NaN())
	if _, err := ExtractSegments(logits, []bool{true, true, true, false}); err == nil {
		t.Fatal("accepted non-finite logit")
	}
}
