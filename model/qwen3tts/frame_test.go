package qwen3tts

import "testing"

func TestAcousticFrameLayout(t *testing.T) {
	layout, err := NewAcousticFrameLayout(ParsedConfig{CPNumCodeGroups: 16, CPVocabSize: 2048})
	if err != nil {
		t.Fatal(err)
	}
	if layout.TotalCodeGroups != 16 || layout.SemanticGroup != 0 || layout.AcousticCodesPerFrame != 15 || len(layout.AcousticGroups) != 15 {
		t.Fatalf("layout=%+v", layout)
	}
	for i, group := range layout.AcousticGroups {
		if group != i+1 {
			t.Fatalf("groups=%v", layout.AcousticGroups)
		}
	}
	frame := make([]uint32, 15)
	for i := range frame {
		frame[i] = uint32(i)
	}
	if err := layout.ValidateFrame(frame); err != nil {
		t.Fatal(err)
	}
	if err := layout.ValidateFrame(frame[:14]); err == nil {
		t.Fatal("expected frame length error")
	}
	frame[14] = 2048
	if err := layout.ValidateFrame(frame); err == nil {
		t.Fatal("expected vocab range error")
	}
}

func TestAcousticFrameLayoutRejectsMalformed(t *testing.T) {
	if _, err := NewAcousticFrameLayout(ParsedConfig{CPNumCodeGroups: 1, CPVocabSize: 2048}); err == nil {
		t.Fatal("expected code group error")
	}
	bad := AcousticFrameLayout{TotalCodeGroups: 16, SemanticGroup: 0, AcousticGroups: []int{1, 3}, AcousticCodesPerFrame: 2, CodecVocab: 2048}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected non-contiguous acoustic group error")
	}
}
