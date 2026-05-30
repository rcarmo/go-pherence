package qwen3tts

import "testing"

func TestTalkerInputLayout(t *testing.T) {
	prefill := PrefillLayout{TextTokens: 11, CodecTokens: 7, FirstTextIndex: CustomVoiceFirstTextIndex, OverlayPosition: CustomVoiceFirstTextIndex, TalkerHiddenSize: 1024, EmbeddingFloats: 11264}
	layout, err := NewTalkerInputLayout(ParsedConfig{TalkerTextHiddenSize: 2048, TalkerHiddenSize: 1024}, prefill)
	if err != nil {
		t.Fatal(err)
	}
	if layout.ProjectionFloats != 2097152 || layout.FusedInputFloats != 11264 {
		t.Fatalf("layout=%+v", layout)
	}
}

func TestTalkerInputLayoutRejectsMalformed(t *testing.T) {
	bad := TalkerInputLayout{TextHiddenSize: 2048, TalkerHiddenSize: 1024, TextTokens: 11, CodecTokens: 7, OverlayPosition: 0, ProjectionFloats: 2097152, FusedInputFloats: 11264}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected overlay position error")
	}
	bad = TalkerInputLayout{TextHiddenSize: 2048, TalkerHiddenSize: 1024, TextTokens: 11, CodecTokens: 7, OverlayPosition: CustomVoiceFirstTextIndex, ProjectionFloats: 1, FusedInputFloats: 11264}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected projection size error")
	}
}
