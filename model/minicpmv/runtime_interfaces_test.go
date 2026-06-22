package minicpmv

import (
	"errors"
	"testing"
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
