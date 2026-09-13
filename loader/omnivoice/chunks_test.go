package omnivoice

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

func TestPlanChunksPreservesTrimmedTextAndTags(t *testing.T) {
	text := "  Dr. Smith said hello. [laughter] Then he waved to everyone. 你好世界！  "
	ref := testReference(80, "Reference voice sample.")
	cfg := Config{NumAudioCodebook: 1, AudioVocabSize: 4096, AudioMaskID: 4095}
	tok := testChunkTokenizer(ref.Transcript, text, "en", "None")

	chunks, err := PlanChunks(cfg, tok, text, ref, PreparePromptOptions{Language: "en", Denoise: true}, 45)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks, got %d", len(chunks))
	}
	if got, want := joinPromptText(chunks), strings.TrimSpace(text); got != want {
		t.Fatalf("reconstructed text mismatch\n got: %q\nwant: %q", got, want)
	}
	for i, chunk := range chunks {
		if chunk.TargetFrames <= 0 || chunk.TargetFrames > 45 {
			t.Fatalf("chunk %d target_frames=%d", i, chunk.TargetFrames)
		}
		if strings.Count(chunk.Text, "[") != strings.Count(chunk.Text, "]") {
			t.Fatalf("chunk %d split bracket span: %q", i, chunk.Text)
		}
	}
}

func TestPlanChunksRuneFallbackPreservesCJK(t *testing.T) {
	text := strings.Repeat("你好世界", 24)
	ref := testReference(64, "参考音频。")
	cfg := Config{NumAudioCodebook: 1, AudioVocabSize: 4096, AudioMaskID: 4095}
	tok := testChunkTokenizer(ref.Transcript, text, "zh", "None")

	chunks, err := PlanChunks(cfg, tok, text, ref, PreparePromptOptions{Language: "zh"}, 30)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected rune fallback chunking, got %d chunk", len(chunks))
	}
	if got := joinPromptText(chunks); got != text {
		t.Fatalf("joined text mismatch\n got: %q\nwant: %q", got, text)
	}
	for i, chunk := range chunks {
		if chunk.TargetFrames <= 0 || chunk.TargetFrames > 30 {
			t.Fatalf("chunk %d target_frames=%d", i, chunk.TargetFrames)
		}
		if chunk.Text == "" {
			t.Fatalf("chunk %d empty text", i)
		}
	}
}

func TestPlanChunksRejectsReferenceWithoutCapacity(t *testing.T) {
	text := "Hi"
	ref := testReference(500, "x")
	cfg := Config{NumAudioCodebook: 1, AudioVocabSize: 4096, AudioMaskID: 4095}
	tok := testChunkTokenizer(ref.Transcript, text, "en", "None")

	if _, err := PlanChunks(cfg, tok, text, ref, PreparePromptOptions{Language: "en"}, 10); err == nil {
		t.Fatal("expected capacity error")
	}
}

func TestPlanChunksRejectsTooManyChunks(t *testing.T) {
	text := strings.Repeat("a", 300)
	ref := testReference(25, "a")
	cfg := Config{NumAudioCodebook: 1, AudioVocabSize: 4096, AudioMaskID: 4095}
	tok := testChunkTokenizer(ref.Transcript, text, "en", "None")

	if _, err := PlanChunks(cfg, tok, text, ref, PreparePromptOptions{Language: "en"}, 1); err == nil || !strings.Contains(err.Error(), "exceeded 128 chunks") {
		t.Fatalf("want too-many-chunks error, got %v", err)
	}
}

func TestPlanChunksWithFirstLimitDefaultEquivalence(t *testing.T) {
	text := "  Dr. Smith said hello. [laughter] Then he waved to everyone. 你好世界！  "
	ref := testReference(80, "Reference voice sample.")
	cfg := Config{NumAudioCodebook: 1, AudioVocabSize: 4096, AudioMaskID: 4095}
	tok := testChunkTokenizer(ref.Transcript, text, "en", "None")
	opts := PreparePromptOptions{Language: "en", Denoise: true}

	got, err := PlanChunksWithFirstLimit(cfg, tok, text, ref, opts, 45, 0)
	if err != nil {
		t.Fatal(err)
	}
	want, err := PlanChunks(cfg, tok, text, ref, opts, 45)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("default planning mismatch\n got: %#v\nwant: %#v", got, want)
	}
}

func TestPlanChunksWithFirstLimitBoundsFirstChunkOnly(t *testing.T) {
	text := strings.Repeat("Alpha beta gamma delta. ", 30)
	ref := testReference(40, "Reference voice sample.")
	cfg := Config{NumAudioCodebook: 1, AudioVocabSize: 4096, AudioMaskID: 4095}
	tok := testChunkTokenizer(ref.Transcript, text, "en", "None")

	chunks, err := PlanChunksWithFirstLimit(cfg, tok, text, ref, PreparePromptOptions{Language: "en"}, 60, 20)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks, got %d", len(chunks))
	}
	if chunks[0].TargetFrames > 20 {
		t.Fatalf("first chunk target_frames=%d want <=20", chunks[0].TargetFrames)
	}
	seenLarger := false
	for i, chunk := range chunks {
		if chunk.TargetFrames <= 0 || chunk.TargetFrames > 60 {
			t.Fatalf("chunk %d target_frames=%d", i, chunk.TargetFrames)
		}
		if i > 0 && chunk.TargetFrames > 20 {
			seenLarger = true
		}
	}
	if !seenLarger {
		t.Fatalf("expected a later chunk to exceed first chunk limit; chunks=%v", chunkFrames(chunks))
	}
}

func TestPlanChunksWithFirstLimitPreservesExactUnicodeText(t *testing.T) {
	text := "  Olá, café — déjà vu? Niño! «Ça va?» 你好，世界！ Grüß Gott; smørrebrød...  "
	ref := testReference(50, "Referência de voz.")
	cfg := Config{NumAudioCodebook: 1, AudioVocabSize: 4096, AudioMaskID: 4095}
	tok := testChunkTokenizer(ref.Transcript, text, "pt", "None")

	chunks, err := PlanChunksWithFirstLimit(cfg, tok, text, ref, PreparePromptOptions{Language: "pt"}, 55, 18)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks, got %d", len(chunks))
	}
	if got, want := joinPromptText(chunks), strings.TrimSpace(text); got != want {
		t.Fatalf("reconstructed text mismatch\n got: %q\nwant: %q", got, want)
	}
}

func TestPlanChunksWithFirstLimitRejectsInvalidLimits(t *testing.T) {
	text := "hello world"
	ref := testReference(20, "Reference voice sample.")
	cfg := Config{NumAudioCodebook: 1, AudioVocabSize: 4096, AudioMaskID: 4095}
	tok := testChunkTokenizer(ref.Transcript, text, "en", "None")

	for _, first := range []int{-1, 21} {
		if _, err := PlanChunksWithFirstLimit(cfg, tok, text, ref, PreparePromptOptions{Language: "en"}, 20, first); err == nil || !strings.Contains(err.Error(), "first_frames") {
			t.Fatalf("first_frames=%d: want validation error, got %v", first, err)
		}
	}
}

func TestPlanChunksWithFirstLimitVeryShortText(t *testing.T) {
	text := "  Hi  "
	ref := testReference(80, "Reference voice sample.")
	cfg := Config{NumAudioCodebook: 1, AudioVocabSize: 4096, AudioMaskID: 4095}
	tok := testChunkTokenizer(ref.Transcript, text, "es", "None")

	chunks, err := PlanChunksWithFirstLimit(cfg, tok, text, ref, PreparePromptOptions{Language: "es"}, 60, 30)
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 1 {
		t.Fatalf("expected single chunk, got %d", len(chunks))
	}
	if chunks[0].TargetFrames > 30 {
		t.Fatalf("target_frames=%d want <=30", chunks[0].TargetFrames)
	}
	if chunks[0].Text != strings.TrimSpace(text) {
		t.Fatalf("text=%q want %q", chunks[0].Text, strings.TrimSpace(text))
	}
}

func TestRealPlanChunksOptIn(t *testing.T) {
	root := os.Getenv("GO_PHERENCE_REAL_OMNIVOICE")
	fixture := os.Getenv("GO_PHERENCE_REAL_PROMPT")
	if root == "" || fixture == "" {
		t.Skip("set GO_PHERENCE_REAL_OMNIVOICE and GO_PHERENCE_REAL_PROMPT")
	}
	raw, err := os.ReadFile(fixture)
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		Reference CachedReferenceTokens `json:"reference"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadConfig(root)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.Load(filepath.Join(root, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	text := "First sentence. Second sentence with [laughter]. 第三句没有空格但是有中文标点。Final sentence!"
	chunks, err := PlanChunks(cfg, tok, text, f.Reference, PreparePromptOptions{Language: "en", Denoise: true}, 60)
	if err != nil {
		t.Fatal(err)
	}
	if got := joinPromptText(chunks); got != strings.TrimSpace(text) {
		t.Fatalf("joined text mismatch\n got: %q\nwant: %q", got, strings.TrimSpace(text))
	}
	for i, chunk := range chunks {
		if chunk.TargetFrames <= 0 || chunk.TargetFrames > 60 {
			t.Fatalf("chunk %d target_frames=%d", i, chunk.TargetFrames)
		}
	}
}

func testReference(frames int, transcript string) CachedReferenceTokens {
	codes := make([]int, frames)
	return CachedReferenceTokens{Books: 1, Frames: frames, Codes: codes, Transcript: transcript}
}

func chunkFrames(prompts []PreparedPrompt) []int {
	frames := make([]int, len(prompts))
	for i, prompt := range prompts {
		frames[i] = prompt.TargetFrames
	}
	return frames
}

func joinPromptText(prompts []PreparedPrompt) string {
	var b strings.Builder
	for _, prompt := range prompts {
		b.WriteString(prompt.Text)
	}
	return b.String()
}

func testChunkTokenizer(texts ...string) *tokenizer.Tokenizer {
	vocab := map[string]int{"▁the": 1, "▁": 2}
	nextID := 3
	add := func(s string) {
		if s == "" {
			return
		}
		if _, ok := vocab[s]; !ok {
			vocab[s] = nextID
			nextID++
		}
	}
	for _, special := range []string{
		"<|denoise|>",
		"<|lang_start|>",
		"<|lang_end|>",
		"<|instruct_start|>",
		"<|instruct_end|>",
		"<|text_start|>",
		"<|text_end|>",
	} {
		add(special)
	}
	for _, text := range texts {
		for _, field := range strings.Fields(text) {
			add(field)
			add("▁" + field)
		}
		for _, r := range text {
			add(string(r))
		}
	}
	inv := make(map[int]string, len(vocab))
	addedSpecial := make(map[string]int, 7)
	for token, id := range vocab {
		inv[id] = token
	}
	for _, special := range []string{
		"<|denoise|>",
		"<|lang_start|>",
		"<|lang_end|>",
		"<|instruct_start|>",
		"<|instruct_end|>",
		"<|text_start|>",
		"<|text_end|>",
	} {
		addedSpecial[special] = vocab[special]
	}
	return &tokenizer.Tokenizer{Vocab: vocab, InvVocab: inv, AddedSpecial: addedSpecial}
}

func TestPlanChunksRejectsOversizedProtectedSpan(t *testing.T) {
	cfg := Config{NumAudioCodebook: 1, AudioVocabSize: 4096, AudioMaskID: 4095}
	tok := testChunkTokenizer("Reference. laughter")
	ref := testReference(2, "Reference.")
	if _, err := PlanChunks(cfg, tok, "["+strings.Repeat("laughter", 100)+"]", ref, PreparePromptOptions{}, 1); err == nil {
		t.Fatal("split oversized protected span")
	}
}

func TestPlanChunksRejectsInvalidUTF8(t *testing.T) {
	cfg := Config{NumAudioCodebook: 1, AudioVocabSize: 4096, AudioMaskID: 4095}
	tok := testChunkTokenizer("Reference. laughter")
	if _, err := PlanChunks(cfg, tok, string([]byte{0xff}), testReference(2, "Reference."), PreparePromptOptions{}, 75); err == nil {
		t.Fatal("accepted invalid UTF8")
	}
}

func TestProtectedControlTokenBoundary(t *testing.T) {
	s := []rune("x<|text_start|>y[laughter]z")
	blocked := buildDisallowedBoundaries(s)
	for i := 2; i <= 14; i++ {
		if !blocked[i] {
			t.Fatalf("control token boundary %d unprotected", i)
		}
	}
}
