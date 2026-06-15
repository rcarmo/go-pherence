package diffusiongemma

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

func localDiffusionGemmaModelDir(t *testing.T) string {
	t.Helper()
	dir := filepath.Join("..", "..", "models", "diffusiongemma-26B-A4B-it-FP8")
	if _, err := os.Stat(filepath.Join(dir, "tokenizer.json")); err != nil {
		t.Skip("local FP8 DiffusionGemma tokenizer metadata not present")
	}
	return dir
}

func TestLocalDiffusionGemmaProcessorSpecialTokenIDs(t *testing.T) {
	dir := localDiffusionGemmaModelDir(t)
	meta, err := LoadMetadata(dir)
	if err != nil {
		t.Fatal(err)
	}
	if meta.Processor == nil || meta.Tokenizer == nil {
		t.Fatalf("processor/tokenizer metadata unavailable: processor=%v tokenizer=%v", meta.Processor != nil, meta.Tokenizer != nil)
	}
	if meta.Processor.ProcessorClass != "Gemma4Processor" || meta.Processor.ImageProcessor != "Gemma4ImageProcessor" || meta.Processor.VideoProcessor != "Gemma4VideoProcessor" {
		t.Fatalf("processor metadata=%+v", meta.Processor)
	}
	if meta.Processor.ImageSeqLength != meta.Shape.VisionSoftTokens || meta.Processor.PatchSize != meta.Shape.PatchSize {
		t.Fatalf("processor vision shape image_seq=%d patch=%d shape soft=%d patch=%d", meta.Processor.ImageSeqLength, meta.Processor.PatchSize, meta.Shape.VisionSoftTokens, meta.Shape.PatchSize)
	}
	specials := meta.Tokenizer.SpecialTokenIDs(meta.Processor)
	want := SpecialTokenIDs{BOS: 2, EOS: 1, PAD: 0, MASK: 4, THINK: 98, BOI: 255999, EOI: 258882, IMAGE: 258880, BOT: 105, EOT: 106, BOC: 100, EOC: 101}
	if specials != want {
		t.Fatalf("specials=%+v want %+v", specials, want)
	}
}

func TestLocalDiffusionGemmaTemplateChatPromptIDs(t *testing.T) {
	dir := localDiffusionGemmaModelDir(t)
	meta, err := LoadMetadata(dir)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	framed, err := BuildTemplateChatPromptIDs(
		[]TextChatMessage{{Role: "user", Content: "Lisbon is famous for"}},
		meta.Tokenizer.SpecialTokenIDs(meta.Processor),
		tok.Encode,
		ChatRenderOptions{AddGenerationPrompt: true},
	)
	if err != nil {
		t.Fatal(err)
	}
	// Locks down the runner/server native text chat path used for local GGUF
	// validation. This is the same prompt prefix used in bounded CLI smokes.
	want := []int{105, 2364, 107, 98357, 4142, 563, 9639, 573, 106, 107, 105, 4368, 107}
	if len(framed.InputIDs) != len(want) {
		t.Fatalf("prompt len=%d want %d ids=%v", len(framed.InputIDs), len(want), framed.InputIDs)
	}
	for i := range want {
		if framed.InputIDs[i] != want[i] {
			t.Fatalf("prompt[%d]=%d want %d ids=%v", i, framed.InputIDs[i], want[i], framed.InputIDs)
		}
	}
}
