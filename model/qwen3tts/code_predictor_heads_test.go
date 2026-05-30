package qwen3tts

import "testing"

func TestCodePredictorHeadLayout(t *testing.T) {
	layout, err := NewCodePredictorHeadLayout(ParsedConfig{CPNumCodeGroups: 16, CPVocabSize: 2048})
	if err != nil {
		t.Fatal(err)
	}
	if layout.SemanticGroup != 0 || layout.Heads != 15 || len(layout.HeadGroups) != 15 || layout.VocabSize != 2048 {
		t.Fatalf("layout=%+v", layout)
	}
	if err := layout.ValidateHeadLogits(0, 2048); err != nil {
		t.Fatal(err)
	}
	if err := layout.ValidateHeadLogits(14, 2048); err != nil {
		t.Fatal(err)
	}
	if err := layout.ValidateHeadLogits(15, 2048); err == nil {
		t.Fatal("expected head range error")
	}
	if err := layout.ValidateHeadLogits(0, 1024); err == nil {
		t.Fatal("expected logits size error")
	}
}

func TestCodePredictorHeadLayoutRejectsMalformed(t *testing.T) {
	bad := CodePredictorHeadLayout{SemanticGroup: 1, Heads: 1, HeadGroups: []int{1}, VocabSize: 2048}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected semantic group error")
	}
	bad = CodePredictorHeadLayout{SemanticGroup: 0, Heads: 2, HeadGroups: []int{1, 3}, VocabSize: 2048}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected non-contiguous head group error")
	}
}
