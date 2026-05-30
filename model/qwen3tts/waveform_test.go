package qwen3tts

import "testing"

func TestWaveformLayout(t *testing.T) {
	layout, err := NewWaveformLayout(DecoderPlan{FrameRateHz: 12})
	if err != nil {
		t.Fatal(err)
	}
	if layout.SampleRateHz != 24000 || layout.Channels != 1 || layout.SamplesPerFrame != 2000 {
		t.Fatalf("layout=%+v", layout)
	}
	samples, err := layout.SamplesForFrames(3)
	if err != nil {
		t.Fatal(err)
	}
	if samples != 6000 {
		t.Fatalf("samples=%d", samples)
	}
	if _, err := layout.SamplesForFrames(-1); err == nil {
		t.Fatal("expected negative frame count error")
	}
}

func TestWaveformLayoutRejectsMalformed(t *testing.T) {
	if _, err := NewWaveformLayout(DecoderPlan{FrameRateHz: 0}); err == nil {
		t.Fatal("expected frame-rate error")
	}
	bad := WaveformLayout{FrameRateHz: 7, SampleRateHz: 24000, Channels: 1, SamplesPerFrame: 1}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected divisibility/samples error")
	}
	bad = WaveformLayout{FrameRateHz: 12, SampleRateHz: 24000, Channels: 1, SamplesPerFrame: 1}
	if err := bad.Validate(); err == nil {
		t.Fatal("expected samples/frame error")
	}
}
