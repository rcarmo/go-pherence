package whisper

import "testing"

func TestForceAlignRejectsEmptyDecoderLayers(t *testing.T) {
	cfg := Tiny()
	cfg.DecoderLayers = 0
	dec := NewDecoder(cfg)
	state := NewDecoderState(cfg, make([]float32, cfg.EncoderDModel), 1, dec)
	if got := ForceAlign(dec, state, []int{1, 2}, cfg, 1); got != nil {
		t.Fatalf("ForceAlign with no layers=%v want nil", got)
	}
}

func TestForceAlignRejectsMissingCrossK(t *testing.T) {
	cfg := Tiny()
	dec := NewDecoder(cfg)
	state := &DecoderState{}
	if got := ForceAlign(dec, state, []int{1}, cfg, 1); got != nil {
		t.Fatalf("ForceAlign with missing cross-K=%v want nil", got)
	}
}
