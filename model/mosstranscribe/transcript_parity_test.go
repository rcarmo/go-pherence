package mosstranscribe

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"testing"
)

type transcriptParityFixture struct {
	Transformers string              `json:"transformers"`
	AudioSHA256  string              `json:"audio_sha256"`
	AudioSamples int                 `json:"audio_samples"`
	AudioTokens  int                 `json:"audio_tokens"`
	PromptTokens int                 `json:"prompt_tokens"`
	GeneratedIDs []int               `json:"generated_ids"`
	Text         string              `json:"text"`
	Segments     []TranscriptSegment `json:"segments"`
}

func TestRealCheckpointJFKTranscriptParity(t *testing.T) {
	modelDir := os.Getenv("MOSS_TRANSCRIBE_MODEL_DIR")
	if modelDir == "" {
		t.Skip("set MOSS_TRANSCRIBE_MODEL_DIR for the real-speech transcript gate")
	}
	fixtureData, err := os.ReadFile("testdata/jfk_transformers_f32.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture transcriptParityFixture
	if err := json.Unmarshal(fixtureData, &fixture); err != nil {
		t.Fatal(err)
	}
	audioPath := "../../testdata/jfk.wav"
	audioData, err := os.ReadFile(audioPath)
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(audioData)
	if got := hex.EncodeToString(digest[:]); got != fixture.AudioSHA256 {
		t.Fatalf("JFK fixture SHA256=%s want %s", got, fixture.AudioSHA256)
	}
	model, err := LoadNativeModel(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	samples, err := ReadAudioWAV(audioPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(samples) != fixture.AudioSamples {
		t.Fatalf("audio samples=%d want %d", len(samples), fixture.AudioSamples)
	}
	audioEmbeddings, audioTokens, err := model.EncodeAudio(samples)
	if err != nil {
		t.Fatal(err)
	}
	if audioTokens != fixture.AudioTokens {
		t.Fatalf("audio tokens=%d want %d", audioTokens, fixture.AudioTokens)
	}
	inputIDs, err := model.Processor.EncodePrompt(BuildTranscriptionPrompt(""), audioTokens, model.Decoder.Config.MaxSeqLen)
	if err != nil {
		t.Fatal(err)
	}
	if len(inputIDs) != fixture.PromptTokens {
		t.Fatalf("prompt tokens=%d want %d", len(inputIDs), fixture.PromptTokens)
	}
	generated, err := model.GenerateGreedy(inputIDs, audioEmbeddings, 256)
	if err != nil {
		t.Fatal(err)
	}
	if !equalTokenIDs(generated, fixture.GeneratedIDs) {
		t.Fatalf("generated token mismatch\n got=%v\nwant=%v", generated, fixture.GeneratedIDs)
	}
	text := model.Processor.Decode(generated)
	if text != fixture.Text {
		t.Fatalf("transcript mismatch\n got=%q\nwant=%q", text, fixture.Text)
	}
	segments := ParseTranscript(text)
	if len(segments) != len(fixture.Segments) {
		t.Fatalf("segments=%+v want %+v", segments, fixture.Segments)
	}
	for i := range segments {
		if segments[i] != fixture.Segments[i] {
			t.Fatalf("segment[%d]=%+v want %+v", i, segments[i], fixture.Segments[i])
		}
	}
	t.Logf("Transformers %s exact token/transcript parity: %d generated tokens, %d segments", fixture.Transformers, len(generated), len(segments))
}

func equalTokenIDs(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
