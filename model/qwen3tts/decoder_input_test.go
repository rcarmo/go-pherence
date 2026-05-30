package qwen3tts

import "testing"

func TestDecoderInputLayout(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{"talker_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64,"code_predictor_config":{"hidden_size":1024,"num_attention_heads":16,"head_dim":64,"vocab_size":2048,"num_code_groups":16}}}`))
	if err != nil {
		t.Fatal(err)
	}
	layout, err := NewDecoderInputLayout(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if layout.FrameRateHz != 12 || layout.AcousticGroups != 15 || layout.CodesPerFrame != 15 || layout.FirstCodeGroup != 1 || layout.LastCodeGroup != 15 {
		t.Fatalf("layout=%+v", layout)
	}
	codes, err := layout.CodesForFrames(4)
	if err != nil {
		t.Fatal(err)
	}
	if codes != 60 {
		t.Fatalf("codes=%d", codes)
	}
	if err := layout.ValidateCodes(make([]uint32, 30)); err != nil {
		t.Fatal(err)
	}
}

func TestDecoderInputLayoutRejectsMalformed(t *testing.T) {
	bad := DecoderInputLayout{FrameRateHz: 24, AcousticGroups: 15, CodecVocab: 2048, CodesPerFrame: 15, SemanticGroup: 0, FirstCodeGroup: 1, LastCodeGroup: 15}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected frame rate error")
	}
	bad = DecoderInputLayout{FrameRateHz: 12, AcousticGroups: 15, CodecVocab: 2048, CodesPerFrame: 1, SemanticGroup: 0, FirstCodeGroup: 1, LastCodeGroup: 15}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected codes/frame error")
	}
	bad = DecoderInputLayout{FrameRateHz: 12, AcousticGroups: 15, CodecVocab: 2048, CodesPerFrame: 15, SemanticGroup: 0, FirstCodeGroup: 1, LastCodeGroup: 14}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected last group error")
	}
	good := DecoderInputLayout{FrameRateHz: 12, AcousticGroups: 15, CodecVocab: 2048, CodesPerFrame: 15, SemanticGroup: 0, FirstCodeGroup: 1, LastCodeGroup: 15}
	if _, err := good.CodesForFrames(-1); err == nil {
		t.Fatal("expected negative frame count error")
	}
	if err := good.ValidateCodes(make([]uint32, 14)); err == nil {
		t.Fatal("expected indivisible codes error")
	}
	codes := make([]uint32, 15)
	codes[3] = 2048
	if err := good.ValidateCodes(codes); err == nil {
		t.Fatal("expected vocab range error")
	}
}
