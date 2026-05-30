package qwen3tts

import "testing"

func TestSemanticTokenLayout(t *testing.T) {
	layout, err := NewSemanticTokenLayout(ParsedConfig{TalkerVocabSize: CodecVocabSize})
	if err != nil {
		t.Fatal(err)
	}
	if layout.Group != 0 || layout.VocabSize != CodecVocabSize || layout.BOS != CodecBOS || layout.EOS != CodecEOS {
		t.Fatalf("layout=%+v", layout)
	}
	if err := layout.ValidateToken(0); err != nil {
		t.Fatal(err)
	}
	if err := layout.ValidateToken(CodecVocabSize - 1); err != nil {
		t.Fatal(err)
	}
	if err := layout.ValidateToken(CodecVocabSize); err == nil {
		t.Fatal("expected token range error")
	}
	if err := layout.ValidateSequence([]uint32{CodecBOS, 1, CodecEOS}); err != nil {
		t.Fatal(err)
	}
	if err := layout.ValidateSequence(nil); err == nil {
		t.Fatal("expected empty sequence error")
	}
}

func TestSemanticTokenLayoutRejectsMalformed(t *testing.T) {
	bad := SemanticTokenLayout{Group: 1, VocabSize: CodecVocabSize, BOS: CodecBOS, EOS: CodecEOS, Pad: CodecPad, NoThink: CodecNoThink, Think: CodecThink, ThinkBOS: CodecThinkBOS, ThinkEOS: CodecThinkEOS}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected group error")
	}
	bad = SemanticTokenLayout{Group: 0, VocabSize: 2, BOS: CodecBOS, EOS: CodecEOS, Pad: CodecPad, NoThink: CodecNoThink, Think: CodecThink, ThinkBOS: CodecThinkBOS, ThinkEOS: CodecThinkEOS}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected special token range error")
	}
}
