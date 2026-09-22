package minicpmv

import (
	"errors"
	"testing"
)

var (
	_ TextBackbone = (*TextCPU)(nil)
	_ AudioEncoder = (*AudioCPU)(nil)
)

func TestPendingRuntimeInterfaces(t *testing.T) {
	rt := NewPendingRuntimeInterfaces()
	if rt.Vision == nil || rt.Resampler == nil || rt.Text == nil || rt.Audio == nil {
		t.Fatalf("nil pending runtime interface: %+v", rt)
	}
	if _, err := rt.Vision.EncodeImage(nil, [4]int{}); !errors.Is(err, ErrRuntimeNotImplemented) {
		t.Fatalf("vision err=%v", err)
	}
	if _, err := rt.Resampler.Resample(nil, 0, 0); !errors.Is(err, ErrRuntimeNotImplemented) {
		t.Fatalf("resampler err=%v", err)
	}
	if _, err := rt.Text.GenerateFromEmbeddings(nil, 0, 0, 0); !errors.Is(err, ErrRuntimeNotImplemented) {
		t.Fatalf("text err=%v", err)
	}
	if _, err := rt.Audio.EncodeAudio(nil, 0, 0); !errors.Is(err, ErrRuntimeNotImplemented) {
		t.Fatalf("audio err=%v", err)
	}
}

func TestTextAudioRuntimeInterfaces(t *testing.T) {
	cfg := tinyAudioConfig()
	audio, err := LoadAudioCPU(tinyAudioSource(cfg), cfg)
	if err != nil {
		t.Fatal(err)
	}
	textCfg := tinyTextConfig("qwen2")
	text, err := LoadTextCPU(tinyTextSource(textCfg), textCfg)
	if err != nil {
		t.Fatal(err)
	}
	rt, err := NewTextAudioRuntimeInterfaces(text, audio)
	if err != nil || rt.Text != text || rt.Audio != audio {
		t.Fatalf("runtime=%+v err=%v", rt, err)
	}
	if _, err := rt.Vision.EncodeImage(nil, [4]int{}); !errors.Is(err, ErrRuntimeNotImplemented) {
		t.Fatalf("vision err=%v", err)
	}
	if _, err := NewTextAudioRuntimeInterfaces(nil, audio); err == nil {
		t.Fatal("accepted nil text")
	}
}
