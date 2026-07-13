package mosstranscribe

import (
	"os"
	"testing"

	llmmodel "github.com/rcarmo/go-pherence/model"
)

func TestRealCheckpointQwenDecoderLoads(t *testing.T) {
	modelDir := os.Getenv("MOSS_TRANSCRIBE_MODEL_DIR")
	if modelDir == "" {
		t.Skip("set MOSS_TRANSCRIBE_MODEL_DIR for the pinned real-checkpoint gate")
	}
	decoder, err := llmmodel.LoadLlama(modelDir)
	if err != nil {
		t.Fatal(err)
	}
	cfg := decoder.Config
	if cfg.ModelType != "qwen3" || cfg.HiddenSize != 1024 || cfg.NumLayers != 28 || cfg.NumHeads != 16 || cfg.NumKVHeads != 8 || cfg.HeadDim != 128 || cfg.VocabSize != 151936 {
		t.Fatalf("unexpected decoder config: %+v", cfg)
	}
	if cfg.TensorPrefix != "model.language_model." || len(decoder.Layers) != 28 || decoder.EmbedTokens == nil || decoder.LMHead != decoder.EmbedTokens {
		t.Fatalf("decoder graph prefix=%q layers=%d embed=%v tied=%v", cfg.TensorPrefix, len(decoder.Layers), decoder.EmbedTokens != nil, decoder.LMHead == decoder.EmbedTokens)
	}
}
