package qwen3tts

import (
	"path/filepath"
	"testing"

	"github.com/rcarmo/go-pherence/loader/tokenizer"
)

func TestLoadReferenceFixture(t *testing.T) {
	fx, err := LoadReferenceFixture(filepath.Join("testdata", "customvoice_prompt_fixture.json"))
	if err != nil {
		t.Fatal(err)
	}
	if fx.Variant != CustomVoice || fx.ModelSize != "0b6" || fx.Speaker != Ryan || fx.Language != English {
		t.Fatalf("fixture metadata=%+v", fx)
	}
	tok := &tokenizer.Tokenizer{Vocab: map[string]int{"Hello": 9707, "Ġworld": 1879}}
	prompt, err := BuildCustomVoicePrompt(tok, fx.Text, fx.Speaker, fx.Language)
	if err != nil {
		t.Fatal(err)
	}
	if !eq(prompt.Text, fx.Prompt.Text) || !eq(prompt.Codec, fx.Prompt.Codec) {
		t.Fatalf("prompt mismatch built=%+v fixture=%+v", prompt, fx.Prompt)
	}
	cov := fx.Coverage()
	if !cov.Prompt || cov.CompleteRuntimeTrace || len(cov.Missing) != 4 {
		t.Fatalf("coverage=%+v", cov)
	}
}

func TestLoadReferencePlaceholderFixture(t *testing.T) {
	fx, err := LoadReferenceFixture(filepath.Join("testdata", "customvoice_reference_placeholder.json"))
	if err != nil {
		t.Fatal(err)
	}
	cov := fx.Coverage()
	if !cov.CompleteRuntimeTrace || !cov.SemanticToken || !cov.AcousticFrame || !cov.DecodedWAVSummary || !cov.RuntimeRequest {
		t.Fatalf("coverage=%+v", cov)
	}
	if fx.Talker.LogitChecksum != "pending-qwen3-tts-rs" || fx.Decoder12Hz.SHA256 != "pending-qwen3-tts-rs" {
		t.Fatalf("placeholder checksums not preserved: %+v", fx)
	}
	cfg, err := ParseConfig([]byte(`{"tts_model_type":"custom_voice","talker_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64,"code_predictor_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64,"vocab_size":2048,"num_code_groups":16}}}`))
	if err != nil {
		t.Fatal(err)
	}
	plan, err := NewRuntimeRequestPlan(cfg, RuntimeRequest{Conditioning: ConditioningRequest{Speaker: fx.Speaker, Language: fx.Language}, Prompt: fx.Prompt, MaxFrames: fx.Runtime.MaxFrames})
	if err != nil {
		t.Fatal(err)
	}
	if plan.MaxFrames != fx.Runtime.MaxFrames || plan.MaxSamples != fx.Runtime.MaxSamples || plan.MaxCodes != fx.Runtime.MaxCodes {
		t.Fatalf("runtime fixture summary=%+v plan=%+v", fx.Runtime, plan)
	}
}

func TestReferenceFixtureCoverageComplete(t *testing.T) {
	fx := ReferenceFixture{
		Name:          "complete",
		Variant:       CustomVoice,
		ModelSize:     "0b6",
		Speaker:       Ryan,
		Language:      English,
		Prompt:        PromptIDs{Text: []uint32{1}, Codec: []uint32{2}},
		Talker:        &TalkerReference{FirstSemanticToken: 42},
		CodePredictor: &CodePredictorReference{AcousticFrame: make([]uint32, 15)},
		Decoder12Hz:   &Decoder12HzReference{SampleRate: 24000, Samples: 2000, DurationS: 1.0 / 12.0},
		Runtime:       &RuntimeRequestReference{MaxFrames: 1, MaxSamples: 2000, MaxCodes: 15},
	}
	if err := fx.Validate(); err != nil {
		t.Fatal(err)
	}
	cov := fx.Coverage()
	if !cov.CompleteRuntimeTrace || len(cov.Missing) != 0 {
		t.Fatalf("coverage=%+v", cov)
	}
}

func TestReferenceFixtureRejectsBadAcousticFrame(t *testing.T) {
	fx := ReferenceFixture{
		Name:      "bad_frame",
		Variant:   CustomVoice,
		ModelSize: "0b6",
		Speaker:   Ryan,
		Language:  English,
		Prompt:    PromptIDs{Text: []uint32{1}, Codec: []uint32{2}},
		CodePredictor: &CodePredictorReference{
			AcousticFrame: []uint32{1, 2, 3},
		},
	}
	if err := fx.Validate(); err == nil {
		t.Fatal("expected bad acoustic frame length error")
	}
}

func TestReferenceFixtureRejectsBadSemanticToken(t *testing.T) {
	fx := ReferenceFixture{
		Name:      "bad_semantic",
		Variant:   CustomVoice,
		ModelSize: "0b6",
		Speaker:   Ryan,
		Language:  English,
		Prompt:    PromptIDs{Text: []uint32{1}, Codec: []uint32{2}},
		Talker:    &TalkerReference{FirstSemanticToken: CodecVocabSize},
	}
	if err := fx.Validate(); err == nil {
		t.Fatal("expected bad semantic token error")
	}
}
