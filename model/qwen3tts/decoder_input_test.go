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
	if layout.FrameRateHz != 12 || layout.TotalCodeGroups != 16 || layout.AcousticGroups != 15 || layout.CodesPerFrame != 16 || layout.FirstCodeGroup != 1 || layout.LastCodeGroup != 15 {
		t.Fatalf("layout=%+v", layout)
	}
	if got, err := layout.CodesForFrames(2); err != nil || got != 32 {
		t.Fatalf("decoder codes=%d err=%v", got, err)
	}
	if got, err := layout.AcousticCodesForFrames(2); err != nil || got != 30 {
		t.Fatalf("acoustic codes=%d err=%v", got, err)
	}
	semantic := []uint32{3071, 7}
	acoustic := make([]uint32, 30)
	for i := range acoustic {
		acoustic[i] = uint32(i)
	}
	codes, err := layout.JoinFrames(semantic, acoustic)
	if err != nil {
		t.Fatal(err)
	}
	if len(codes) != 32 || codes[0] != 3071 || codes[1] != 0 || codes[16] != 7 || codes[17] != 15 {
		t.Fatalf("joined=%v", codes)
	}
	plan, err := layout.DecoderPlan()
	if err != nil {
		t.Fatal(err)
	}
	if plan != (DecoderPlan{FrameRateHz: 12, CodeGroups: 16, CodesPerFrame: 16, CodecVocab: 2048}) {
		t.Fatalf("decoder plan=%+v", plan)
	}
}

func TestDecoderInputLayoutRejectsMalformed(t *testing.T) {
	bad := DecoderInputLayout{FrameRateHz: 24, TotalCodeGroups: 16, AcousticGroups: 15, CodecVocab: 2048, CodesPerFrame: 16, SemanticGroup: 0, FirstCodeGroup: 1, LastCodeGroup: 15}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected frame rate error")
	}
	bad.FrameRateHz, bad.CodesPerFrame = 12, 15
	if err := bad.Validate(); err == nil {
		t.Fatal("expected codes/frame error")
	}
	good := DecoderInputLayout{FrameRateHz: 12, TotalCodeGroups: 16, AcousticGroups: 15, CodecVocab: 2048, CodesPerFrame: 16, SemanticGroup: 0, FirstCodeGroup: 1, LastCodeGroup: 15}
	if _, err := good.CodesForFrames(-1); err == nil {
		t.Fatal("expected negative frame count error")
	}
	if _, err := good.JoinFrames(nil, nil); err == nil {
		t.Fatal("accepted empty streams")
	}
	if _, err := good.JoinFrames([]uint32{1}, make([]uint32, 14)); err == nil {
		t.Fatal("accepted short acoustic frame")
	}
	codes := make([]uint32, 16)
	codes[0] = 3071 // semantic codes are modulo-mapped by the decoder codebook
	codes[1] = 2048
	if err := good.ValidateCodes(codes); err == nil {
		t.Fatal("accepted acoustic code outside vocab")
	}
}
