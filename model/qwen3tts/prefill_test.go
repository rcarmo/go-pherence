package qwen3tts

import "testing"

func TestPrefillLayout(t *testing.T) {
	prompt := PromptIDs{Text: []uint32{IMStart, Assistant, Newline, TTSPad, TTSPad, TTSPad, TTSPad, TTSPad, TTSBOS, 9707, 1879}, Codec: []uint32{CodecThink, CodecThinkBOS, 2050, CodecThinkEOS, 3061, CodecPad, CodecBOS}}
	layout, err := NewPrefillLayout(ParsedConfig{TalkerHiddenSize: 1024}, prompt)
	if err != nil {
		t.Fatal(err)
	}
	if layout.TextTokens != 11 || layout.CodecTokens != 7 || layout.EmbeddingFloats != 11264 {
		t.Fatalf("layout=%+v", layout)
	}
	bytes, err := layout.EmbeddingBytes(4)
	if err != nil {
		t.Fatal(err)
	}
	if bytes != 45056 {
		t.Fatalf("bytes=%d", bytes)
	}
}

func TestPrefillLayoutRejectsMalformed(t *testing.T) {
	if _, err := NewPrefillLayout(ParsedConfig{TalkerHiddenSize: 1024}, PromptIDs{Text: []uint32{1}, Codec: []uint32{2}}); err == nil {
		t.Fatal("expected short text stream error")
	}
	bad := PrefillLayout{TextTokens: 11, CodecTokens: 7, FirstTextIndex: CustomVoiceFirstTextIndex, OverlayPosition: CustomVoiceFirstTextIndex, TalkerHiddenSize: 1024, EmbeddingFloats: 1}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected embedding size error")
	}
	if _, err := bad.EmbeddingBytes(0); err == nil {
		t.Fatal("expected bytes/float error")
	}
}
