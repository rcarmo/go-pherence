package qwen3tts

import "testing"

func TestWaveformLayout(t *testing.T) {
	layout, err := NewWaveformLayout(DecoderPlan{FrameRateHz: 12})
	if err != nil {
		t.Fatal(err)
	}
	if layout.SampleRateHz != 24000 || layout.Channels != 1 || layout.SamplesPerFrame != 1920 {
		t.Fatalf("layout=%+v", layout)
	}
	samples, err := layout.SamplesForFrames(3)
	if err != nil || samples != 5760 {
		t.Fatalf("samples=%d err=%v", samples, err)
	}
	frames, err := layout.FramesForSeconds(1)
	if err != nil || frames != 13 {
		t.Fatalf("one second frames=%d err=%v", frames, err)
	}
	if _, err := layout.SamplesForFrames(-1); err == nil {
		t.Fatal("expected negative frame count error")
	}
}

func TestWaveformLayoutRejectsMalformed(t *testing.T) {
	if _, err := NewWaveformLayout(DecoderPlan{FrameRateHz: 0}); err == nil {
		t.Fatal("expected frame-rate error")
	}
	bad := WaveformLayout{FrameRateHz: 12, SampleRateHz: 24000, Channels: 1, SamplesPerFrame: 2000}
	if err := bad.Validate(); err == nil {
		t.Fatal("accepted nominal-rate sizing instead of decoder topology")
	}
	if _, err := (WaveformLayout{FrameRateHz: 12, SampleRateHz: 24000, Channels: 1, SamplesPerFrame: 1920}).FramesForSeconds(-1); err == nil {
		t.Fatal("accepted negative duration")
	}
}
