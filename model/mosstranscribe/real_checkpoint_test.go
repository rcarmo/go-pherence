package mosstranscribe

import (
	"os"
	"testing"
)

func TestRealCheckpointAudioBackbone(t *testing.T) {
	modelDir := os.Getenv("MOSS_TRANSCRIBE_MODEL_DIR")
	if modelDir == "" {
		t.Skip("set MOSS_TRANSCRIBE_MODEL_DIR for the pinned real-checkpoint gate")
	}
	model, err := LoadAudioBackbone(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	if model.Config.AudioTokenID != 151671 || len(model.Encoder.Layers) != 24 || !model.Adaptor.valid() {
		t.Fatalf("unexpected real checkpoint graph: config=%+v layers=%d adaptor=%v", model.Config, len(model.Encoder.Layers), model.Adaptor.valid())
	}
}
